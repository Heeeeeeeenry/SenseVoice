"""
STT 后处理文本纠错模块
支持三种后端：
  - dictionary: 混淆词典（默认，零延迟，内置数百对常见纠错）
  - pycorrector: MacBERT 模型（需 pip install pycorrector torch）
  - llm: 本地/远程 LLM（最高质量，需额外服务）

环境变量:
  CORRECTOR_BACKEND=dictionary|pycorrector|llm
  CORRECTOR_LLM_URL=http://localhost:11434/v1/chat/completions  (llm 模式)
  CORRECTOR_LLM_MODEL=qwen2.5:1.5b  (llm 模式)
"""

import re
import os
from typing import Optional, List, Dict, Tuple

# ═══════════════════════════════════════════════════════════
# 常见 ASR 同音/近音字混淆表（词典纠错）
# 来源：中文常见同音字、ASR 典型错误模式
# ═══════════════════════════════════════════════════════════

CONFUSION_MAP = {
    # ── 的/得/地 ──
    "做的很好": "做得很好",
    "做的非常好": "做得非常好",
    "写的好": "写得好",
    "说的对": "说得对",
    "跑的很快": "跑得很快",
    "来的很早": "来得很早",
    "做的对": "做得对",
    "做的好": "做得好",
    # ── 在/再 ──
    "在说一遍": "再说一遍",
    "在见": "再见",
    "在一次": "再一次",
    "在来一次": "再来一次",
    "在也不会": "再也不会",
    "在也不": "再也不",
    "现也不": "再也不",
    # ── 象/像 ──
    "好象": "好像",
    "好象是": "好像是",
    "不象": "不像",
    "很象": "很像",
    # ── 做/作 ──
    "做用": "作用",
    "做工": "做工",
    # ── 坐/座 ──
    "坐位": "座位",
    "请坐": "请坐",
    # ── 到/道 ──
    "说到的": "说到的",
    "想的到": "想得到",
    "看不道": "看不到",
    "听不道": "听不到",
    "找不道": "找不到",
    # ── 其他高频同音字 ──
    "化钱": "花钱",
    "化了": "花了",
    "花拉": "花了",
    "开起": "开启",
    "关必": "关闭",
    "气动": "启动",
    "起开": "开",
    "在说": "再说",
    # ── 数字相关 ──
    "一块钱": "1元",
    "两块钱": "2元",
    "五块钱": "5元",
    "十块钱": "10元",
    "一百块": "100元",
    # ── 常见词 ──
    "份内": "分内",
    "发先": "发现",
    "先到先得": "先到先得",
    "不然后面": "不然后面",
    # ── 高频连接与上下文字错 ──
    "他象": "他像",
    "在次来": "再次来",
    "在次出现": "再次出现",
    "在次发生": "再次发生",
    "在次确认": "再次确认",
    "在次回到": "再次回到",
    "在次强调": "再次强调",
    "在次提交": "再次提交",
    "己经": "已经",
    "一值": "一直",
    "关建": "关键",
    "由其": "尤其",
    # ── 地名 ──
    "北经": "北京",
    "天汽": "天气",
    # ── 建筑物 ──
    "那栋搂": "那栋楼",
    # ── 感觉类 ──
    "觉的": "觉得",
    "认未": "认为",
    "以未": "以为",
    # ── 连接词 ──
    "所已": "所以",
    "应为": "因为",
    "虽然": "虽然",
    "但是": "但是",
    "如过": "如果",
    "既使": "即使",
    "那末": "那么",
    # ── 时间词 ──
    "以经": "已经",
    "刚材": "刚才",
    "现再": "现在",
    "以候": "以后",
    "以钱": "以前",
    "马尚": "马上",
    # ── 代词 ──
    "它门": "他们",
    "他门": "他们",
    "我门": "我们",
    "你门": "你们",
    # ── 动词 ──
    "只到": "知道",
    "知到": "知道",
    "告术": "告诉",
    "告速": "告诉",
    "觉定": "决定",
    "接的": "接着",
    "继序": "继续",
    "帮住": "帮助",
    # ── 副词/形容词 ──
    "非长": "非常",
    "正个": "整个",
    "几忽": "几乎",
    "突燃": "突然",
    "突兰": "突然",
    "绝的": "绝对",
    "影该": "应该",
    "实际尚": "实际上",
    # ── 网络/科技 ──
    "账豪": "账号",
    "网止": "网址",
    "文当": "文档",
    "下栽": "下载",
    "安庄": "安装",
    "配直": "配置",
    "设直": "设置",
    # ── 商务 ──
    "定单": "订单",
    "客互": "客户",
    "需球": "需求",
    "公思": "公司",
    "象目": "项目",
}

# 正则上下文规则（比简单替换更安全，需要前后文匹配）
CONTEXT_RULES = [
    (r"我花了?(\d+)块(?:钱)?买的?", r"我花了\1元买的", "块钱→元"),
    (r"花拉(\d+)", r"花了\1", "花拉→花了"),
    (r"(\d{2,4})年(\d{1,2})日", r"\1年\2日", "日期格式"),
]


class TextCorrector:
    """文本纠错器"""

    def __init__(
        self,
        backend: str = "dictionary",
        llm_url: Optional[str] = None,
        llm_model: Optional[str] = None,
    ):
        self.backend = backend
        self._model = None

        if backend == "pycorrector":
            self._try_init_pycorrector()
        elif backend == "llm":
            self.llm_url = llm_url or os.getenv("CORRECTOR_LLM_URL", "")
            self.llm_model = llm_model or os.getenv("CORRECTOR_LLM_MODEL", "qwen2.5:1.5b")

    def _try_init_pycorrector(self):
        """尝试加载 PyCorrector，失败回退到词典"""
        try:
            from pycorrector import MacBertCorrector

            import torch  # noqa: F401

            print("⏳ 加载纠错模型 MacBertCorrector...")
            self._model = MacBertCorrector()
            print("✅ 纠错模型就绪")
        except ImportError as e:
            print(f"⚠️  PyCorrector 不可用 ({e})，回退到词典模式")
            print("   安装: pip install pycorrector torch")
            self.backend = "dictionary"

    def correct(self, text: str) -> Dict:
        """返回: {"original": str, "corrected": str, "changes": List[dict]}"""
        if not text or not text.strip():
            return {"original": text, "corrected": text, "changes": []}

        changes = []
        corrected = text

        # 第一步：上下文规则（先执行，避免被词典误匹配）
        corrected, rule_changes = self._context_rule_correct(corrected)
        changes.extend(rule_changes)

        # 第二步：词典快速纠错
        corrected, dict_changes = self._dictionary_correct(corrected)
        changes.extend(dict_changes)

        # 第三步：模型纠错（可选）
        if self.backend == "pycorrector" and self._model is not None:
            corrected, model_changes = self._pycorrector_correct(corrected)
            changes.extend(model_changes)
        elif self.backend == "llm" and self.llm_url:
            corrected, model_changes = self._llm_correct(corrected)
            changes.extend(model_changes)

        return {"original": text, "corrected": corrected, "changes": changes}

    def _dictionary_correct(self, text: str) -> Tuple[str, List[dict]]:
        """基于混淆词典的快速纠错"""
        changes = []
        result = text
        for wrong, right in CONFUSION_MAP.items():
            if wrong in result:
                result = result.replace(wrong, right)
                changes.append({"type": "dictionary", "from": wrong, "to": right})
        return result, changes

    def _context_rule_correct(self, text: str) -> Tuple[str, List[dict]]:
        """基于正则的上下文纠错"""
        changes = []
        result = text
        for pattern, replacement, desc in CONTEXT_RULES:
            if re.search(pattern, result):
                new_text = re.sub(pattern, replacement, result)
                if new_text != result:
                    changes.append(
                        {
                            "type": "rule",
                            "pattern": pattern,
                            "replacement": replacement,
                            "desc": desc,
                        }
                    )
                    result = new_text
        return result, changes

    def _pycorrector_correct(self, text: str) -> Tuple[str, List[dict]]:
        """PyCorrector MacBERT 纠错（适配 pycorrector >= 1.0 新 API）"""
        changes = []
        try:
            result = self._model.correct(text)

            # pycorrector >= 1.0: 返回 dict {source, target, errors: [(原字,纠错字,位置)]}
            if isinstance(result, dict):
                corrected = result.get("target", text)
                errors = result.get("errors", [])
                for err in errors:
                    if isinstance(err, (list, tuple)) and len(err) >= 3:
                        changes.append(
                            {
                                "type": "macbert",
                                "from": err[0],
                                "to": err[1],
                                "position": err[2],
                            }
                        )
                return corrected, changes
            else:
                # 旧版 API: tuple (corrected_text, details)
                corrected, details = result
                for d in details:
                    if d.get("err_type") != "correct":
                        changes.append(
                            {
                                "type": "macbert",
                                "from": d.get("wrong", ""),
                                "to": d.get("right", ""),
                                "position": d.get("begin_offset", 0),
                            }
                        )
                return corrected, changes
        except Exception as e:
            print(f"⚠️  PyCorrector 纠错失败: {e}")
            return text, changes

    def _llm_correct(self, text: str) -> Tuple[str, List[dict]]:
        """LLM 纠错（最高质量，延迟 ~500ms-2s）"""
        import json
        from urllib.request import Request, urlopen

        prompt = f"""你是语音转文字纠错助手。纠正以下ASR结果中的同音/近音字错误。
规则：只改明显错误，不改原意；保留口语化表达；只输出纠正后文本。

原文：{text}
纠正："""

        try:
            data = json.dumps(
                {
                    "model": self.llm_model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1,
                    "max_tokens": max(len(text) * 2, 100),
                }
            ).encode()
            req = Request(self.llm_url, data=data, headers={"Content-Type": "application/json"})
            resp = urlopen(req, timeout=15)
            result = json.loads(resp.read())
            corrected = result["choices"][0]["message"]["content"].strip()
            corrected = corrected.split("纠正：")[-1].strip().split("\n\n")[0]

            if corrected and corrected != text:
                return corrected, [{"type": "llm", "from": text, "to": corrected}]
        except Exception as e:
            print(f"⚠️  LLM 纠错失败: {e}")

        return text, []


# ── 全局单例 ──
_corrector: Optional[TextCorrector] = None


def get_corrector(backend: str = "dictionary") -> TextCorrector:
    global _corrector
    if _corrector is None or _corrector.backend != backend:
        _corrector = TextCorrector(backend=backend)
    return _corrector
