"""
STT 智能纠错模块 — v3.0 双模式版

改进：
1. 乱码检测器：自动判断文本质量，选择轻症/重症模式
2. 轻症模式：外科医生，只修正同音字（防止过度纠正）
3. 重症模式：基于语境重建，适合严重不通顺的 ASR 输出
4. 政务领域术语库 + ASR 常见错误映射
5. 词典预处理 → LLM 纠错 → 编辑距离校验

用法：
    from smart_corrector import smart_correct_paragraph
    result = smart_correct_paragraph(asr_text)
"""

import re
import json
import os
import urllib.request
from typing import Optional, List, Dict, Tuple
from corrector import get_corrector

# ═══════════════════════════════════════════════════════════
# 领域知识：信访 + 政务术语
# ═══════════════════════════════════════════════════════════
DOMAIN_TERMS = {
    # ── 信访/警务 ──
    "public_security": [
        "盗窃", "抢劫", "诈骗", "扒窃", "入室盗窃", "电信诈骗",
        "打架斗殴", "噪音扰民", "赌博", "违章停车",
        "交通事故", "交通拥堵",
        "接警", "出警", "立案", "调解", "治安处罚", "行政处罚",
        "派出所", "民警", "辅警", "局长信箱", "信访", "投诉", "举报",
        "社区", "街道办", "物业", "违建", "占道", "扰民",
        "身份证", "警号", "车牌号", "监控", "摄像头",
    ],
    # ── 政务/机构 ──
    "government": [
        "经济发展局", "改革创新局", "投资与招标股", "行政审批局",
        "发改委", "住建局", "自然资源局", "财政局", "人社局",
        "行政审批", "初步设计", "项目建议书", "可行性研究", "可研",
        "立项", "审批", "核准", "备案", "招投标",
        "城市基础设施", "公共建设", "房地产开发",
        "国有资金", "预算内", "省级预算", "国家预算",
        "负责人", "局长", "主任", "科长", "股长",
    ],
}

# ASR 常见错误 → 正确 映射表（专为政务场景）
ASR_FIX_MAP = {
    # 机构名 ASR 错误
    "惊现": "经信/经济",  # 需 LLM 根据上下文确定
    "经县": "经信/经济",
    "创新剧": "创新局",
    "教改": "教改",
    # 公文术语
    "公建法": "公共建设法",
    "科研": "可研/可行性研究",
    "预算类": "预算内",  # 项目类别
    "建设书": "项目建议书",
    "初设": "初步设计",
    # 通用
    "务时": "务实/我的（因上下文而异）",
    "按，": "按照",
    "指的": "职责",
}

# ═══════════════════════════════════════════════════════════
# 乱码检测器
# ═══════════════════════════════════════════════════════════

# 常用中文高频词（简化词表，用于检测文本流畅度）
_COMMON_WORDS = set("""
的 一 是 在 不 了 有 和 人 这 中 大 为 上 个 国 我 以 要 他
时 来 用 们 生 到 作 地 于 出 就 分 对 成 会 可 主 发 年 动
同 工 也 能 下 过 子 说 产 种 面 而 方 后 多 定 行 学 法 所
民 得 经 十 三 之 进 着 等 部 度 家 电 力 里 如 水 化 高 自
二 理 起 小 物 现 实 加 量 都 两 体 制 机 当 使 点 从 业 本
去 把 性 应 开 它 合 还 因 由 其 些 然 前 外 天 政 四 日 那
社 义 事 平 形 相 全 表 间 样 与 关 各 重 新 线 内 数 正 心
反 你 明 看 原 又 么 利 比 或 但 质 气 第 向 道 命 此 变 条
只 没 结 解 问 意 建 月 公 无 系 军 很 情 者 最 立 代 想 已
通 并 提 直 题 党 程 展 五 果 料 象 员 革 位 入 常 文 总 次
品 式 活 设 及 管 特 件 长 求 老 头 基 资 边 流 路 级 少 图
山 统 接 知 较 将 组 见 计 别 她 手 角 期 根 论 运 农 指 几
九 区 强 放 决 西 被 干 做 必 战 先 回 则 任 取 据 处 队 南
给 色 光 门 即 保 治 北 造 百 规 热 领 七 海 口 东 导 器 压
志 世 金 增 争 济 阶 油 思 术 极 交 受 联 什 认 六 共 权 收
证 改 清 己 美 再 采 转 更 单 风 切 打 白 教 速 花 带 安 场
身 车 例 真 务 具 万 每 目 至 达 走 积 示 议 声 报 斗 完 类
八 离 华 名 确 科 传 张 信 马 节 话 米 整 空 元 况 今 集 温
许 步 群 广 石 记 需 段 研 界 拉 林 律 叫 且 究 观 越 织 装
影 算 低 持 音 众 书 布 复 容 儿 须 际 商 非 验 连 断 深 难
近 矿 千 周 委 素 技 备 半 办 青 省 列 习 响 约 支 般 史 感
劳 便 团 往 酸 历 市 克 何 除 消 构 府 称 太 准 精 值 号 率
族 维 划 选 标 写 存 候 毛 亲 快 效 斯 院 查 江 型 眼 王 按
格 养 易 置 派 层 片 始 却 专 状 育 厂 京 识 适 属 圆 包 火
""".split())

_PUNCT_SET = set("，。！？；：、")


def _calc_garbled_score(text: str) -> float:
    """
    计算文本乱码程度 (0=完全通顺, 1=完全乱码)

    指标：
    1. 有效词汇比例（jieba 分词后命中词典的比例）
    2. 平均句长（过短的断句=乱码标志）
    3. 虚词/标点密度（正常的口语也有一定虚词比例）
    """
    if len(text) < 10:
        return 0.0

    # 指标1: 字符级有效词汇比例
    chars = list(text)
    hits = sum(1 for c in chars if c in _COMMON_WORDS)
    word_ratio = hits / len(chars) if chars else 0

    # 指标2: 断句检测 — ASR 乱码常常是碎片式输出
    # 计算逗号之间的平均字符数，太短说明断句混乱
    fragments = re.split(r'[，,。！？\uff0c]', text)
    fragments = [f.strip() for f in fragments if len(f.strip()) >= 2]
    if fragments:
        avg_frag_len = sum(len(f) for f in fragments) / len(fragments)
        # 平均 < 5 字的片段太多 = 乱码特征
        short_frag_ratio = sum(1 for f in fragments if len(f) < 5) / len(fragments)
    else:
        avg_frag_len = 0
        short_frag_ratio = 1.0

    # 指标3: 连续单字/双字断句（非正常表达的特征）
    # 检查 "按，" "剧，" 这类不自然的断句
    unnatural_breaks = len(re.findall(r'[\u4e00-\u9fff]{1,2}[，,]', text))

    # 综合打分
    score = 0.0
    score += (1.0 - min(word_ratio / 0.5, 1.0)) * 0.4  # 常用词比例低→乱码
    score += min(short_frag_ratio / 0.5, 1.0) * 0.35       # 短片段多→乱码
    score += min(unnatural_breaks / 3.0, 1.0) * 0.25        # 不自然断句多→乱码

    return min(max(score, 0.0), 1.0)


def is_garbled(text: str, threshold: float = 0.25) -> bool:
    """判断文本是否严重乱码"""
    return _calc_garbled_score(text) > threshold


# ═══════════════════════════════════════════════════════════
# 实体锚定
# ═══════════════════════════════════════════════════════════
PROTECTED_PATTERNS = [
    (r'\d{17}[\dXx]', '身份证号'),
    (r'\d{4}年\d{1,2}月\d{1,2}日', '完整日期'),
    (r'\d{1,2}月\d{1,2}日', '月日'),
    (r'\d{1,2}时\d{1,2}分', '时间'),
    (r'[京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青川藏宁琼]'
     r'[A-Za-z]?[A-Za-z0-9]{4,6}', '车牌号'),
    (r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}', '邮箱'),
    (r'https?://[^\s]+', '链接'),
    (r'\d+元|\d+块|\d+万|\d+千', '金额'),
    (r'第\d+条', '法条引用'),
    (r'\d{7,11}', '电话号码'),
]


def _extract_entities(text: str) -> List[Dict]:
    entities = []
    seen_spans = set()
    for pattern, etype in PROTECTED_PATTERNS:
        for match in re.finditer(pattern, text):
            start, end = match.start(), match.end()
            overlapping = any(
                start < s_end and end > s_start
                for s_start, s_end in seen_spans
            )
            if overlapping:
                continue
            original = match.group(0)
            seen_spans.add((start, end))
            entities.append({
                "type": etype,
                "original": original,
                "start": start,
                "end": end,
            })
    entities.sort(key=lambda x: x["start"])
    return entities


def _anchor_entities(text: str, entities: List[Dict]) -> Tuple[str, List[Dict]]:
    result = text
    for i, ent in enumerate(reversed(entities)):
        placeholder = f"␂E{i}␃"
        pos = len(entities) - 1 - i
        entities[pos]["placeholder"] = placeholder
        result = result[:ent["start"]] + placeholder + result[ent["end"]:]
    return result, entities


def _restore_entities(text: str, entities: List[Dict]) -> str:
    result = text
    for ent in entities:
        placeholder = ent.get("placeholder", "")
        if placeholder and placeholder in result:
            result = result.replace(placeholder, ent["original"])
    return result


# ═══════════════════════════════════════════════════════════
# 双模式 LLM Prompt
# ═══════════════════════════════════════════════════════════

SYSTEM_PROMPT_LIGHT = """你是一个中文语音识别(ASR)纠错助手，专门处理信访/政务场景。

## 你的角色
"外科医生"——只修正明显的同音/近音字错误，保留原句结构和措辞。

## 规则
1. 只改明显的同音字错误（如 知道→知到，象→像，在→再）
2. 不确定的字保留原样，不要猜测
3. 不删减、不添加、不调序、不改写
4. 保留口语化表达
5. 只输出纠正后文本，不要解释

## 参考术语（理解语境用）
{domain_terms}

## 文本
{target_text}

纠正后："""

SYSTEM_PROMPT_HEAVY = """你是一位专业的政务语音转文字纠错专家。

## 背景
这是一段政府工作人员的口述自述/汇报，内容涉及行政职务、工作职责、审批流程、项目建议等。ASR（语音识别）转写结果严重不通顺，存在大量同音字错误和断句混乱。

## 核心任务
基于语境理解，将严重乱码的 ASR 转写纠正为连贯通顺的公文表达。

## 领域知识参考
{domain_terms}

### ASR 常见错误→正确映射
{asr_fix_map}

## 纠正规则
1. 保持原文语义和关键信息（姓名、职务、事项），不要添加原文没有的内容
2. 将乱码术语修正为正确的政务术语（参考上述映射表）
3. 合理断句，适度补充衔接词使表达通顺
4. 公文风格，但不需过度书面化——这是口述转写
5. 只输出纠正后的完整文本，不要任何解释

## 待纠错文本
{target_text}

## 纠正后（只输出文本）"""


def _build_domain_terms_str() -> str:
    """构建领域术语字符串（分类展示，帮助 LLM 理解场景）"""
    parts = []
    if DOMAIN_TERMS.get("government"):
        parts.append("【政务/机构】" + "、".join(DOMAIN_TERMS["government"][:25]))
    if DOMAIN_TERMS.get("public_security"):
        parts.append("【信访/警务】" + "、".join(DOMAIN_TERMS["public_security"][:20]))
    return "\n".join(parts)


def _build_asr_fix_map_str() -> str:
    """构建 ASR 错误映射字符串"""
    lines = []
    for wrong, right in ASR_FIX_MAP.items():
        lines.append(f'  "{wrong}" → "{right}"')
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════
# LLM 调用
# ═══════════════════════════════════════════════════════════

def _llm_call(system_prompt: str, user_content: str, max_tokens: int = 1000) -> Optional[str]:
    """调用 DeepSeek LLM（支持 system + user 消息）"""
    api_key = os.environ.get("LLM_API_KEY") or os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        return None

    body = json.dumps({
        "model": os.environ.get("LLM_MODEL", "deepseek-chat"),
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.1,
    }).encode("utf-8")

    req = urllib.request.Request(
        os.environ.get("LLM_API_URL", "https://api.deepseek.com/v1/chat/completions"),
        data=body,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read())
            content = result["choices"][0]["message"]["content"].strip()
            return content if content else None
    except Exception as e:
        print(f"[smart_corrector] LLM error: {e}")
        return None


# ═══════════════════════════════════════════════════════════
# 编辑距离校验
# ═══════════════════════════════════════════════════════════

def _edit_distance_ratio(a: str, b: str) -> float:
    """基于 Jaccard 的差异率 (0=相同, 1=完全不同)"""
    if not a and not b:
        return 0.0
    if not a or not b:
        return 1.0
    set_a, set_b = set(a), set(b)
    intersection = set_a & set_b
    union = set_a | set_b
    if not union:
        return 0.0
    return 1.0 - len(intersection) / len(union)


# ═══════════════════════════════════════════════════════════
# 核心纠错 Pipeline
# ═══════════════════════════════════════════════════════════

_corrector = None


def _get_corrector():
    global _corrector
    if _corrector is None:
        _corrector = get_corrector(backend="dictionary")
    return _corrector


def smart_correct(
    text: str,
    prev_text: str = "",
    next_text: str = "",
    enable_llm: bool = True,
    force_heavy: bool = False,
) -> str:
    """
    智能纠错（单句）

    Args:
        text: 待纠错文本
        prev_text: 前文上下文
        next_text: 后文上下文
        enable_llm: 是否启用 LLM
        force_heavy: 强制使用重症模式

    Returns:
        纠错后文本
    """
    if not text or len(text.strip()) < 3:
        return text

    # Step 1: 词典预处理
    corrector = _get_corrector()
    dict_result = corrector.correct(text)
    pre_corrected = dict_result["corrected"]

    if not enable_llm:
        return pre_corrected

    # Step 2: 检测文本质量 → 选择模式
    heavy_mode = force_heavy or is_garbled(pre_corrected)

    # Step 3: 实体锚定
    entities = _extract_entities(pre_corrected)
    anchored_text, entities = _anchor_entities(pre_corrected, entities)

    # 限制长度
    target = anchored_text[:500] if len(anchored_text) > 500 else anchored_text

    # Step 4: 构建 prompt + LLM 纠错
    if heavy_mode:
        system_prompt = SYSTEM_PROMPT_HEAVY.format(
            domain_terms=_build_domain_terms_str(),
            asr_fix_map=_build_asr_fix_map_str(),
            target_text=target,
        )
        user_content = "请纠正上述文本："
        max_diff = 0.55  # 重症模式允许更大改动
        max_tokens = max(len(target) * 3, 300)
    else:
        system_prompt = SYSTEM_PROMPT_LIGHT.format(
            domain_terms=_build_domain_terms_str(),
            target_text=target,
        )
        user_content = "纠正后："
        max_diff = 0.25  # 轻症模式严格限制改动
        max_tokens = max(len(target) * 2, 150)

    llm_result = _llm_call(system_prompt, user_content, max_tokens=max_tokens)

    if llm_result is None:
        return pre_corrected

    # Step 5: 还原实体
    restored = _restore_entities(llm_result, entities)

    # Step 6: 编辑距离校验
    diff = _edit_distance_ratio(pre_corrected, restored)
    if diff > max_diff:
        print(f"[smart_corrector] {'重症' if heavy_mode else '轻症'}模式 "
              f"编辑距离比率 {diff:.2%} > 阈值 {max_diff:.0%}，回退到词典结果")
        return pre_corrected

    return restored


def _join_with_punctuation(parts: List[str]) -> str:
    """智能拼接：保证相邻句之间有合理标点"""
    if not parts:
        return ""

    ENDING_PUNCT = set('。！？；')
    result_parts = []

    for i, part in enumerate(parts):
        part = part.strip()
        if not part:
            continue

        if result_parts:
            prev = result_parts[-1]
            prev_last = prev[-1] if prev else ''
            if prev_last not in ENDING_PUNCT and prev_last != '，':
                if part[0] not in '，。！？；':
                    result_parts[-1] = prev + '，'

        result_parts.append(part)

    if result_parts and result_parts[-1]:
        last = result_parts[-1]
        if last[-1] == '，':
            result_parts[-1] = last[:-1] + '。'
        elif last[-1] not in ENDING_PUNCT:
            result_parts[-1] = last + '。'

    return ''.join(result_parts)


def smart_correct_paragraph(text: str, enable_llm: bool = True) -> str:
    """
    对完整段落进行纠错（带乱码检测 + 双模式）

    流程：
    1. 预处理：去标记、换行→逗号
    2. 整段检测乱码程度
    3. 分句纠错（带上下文）
    4. 智能标点拼接
    """
    if not text or len(text.strip()) < 3:
        return text

    # 预处理
    text = re.sub(r'\[\d+\]\s*', '', text)
    text = re.sub(r'\n{2,}', '。\n', text)
    text = re.sub(r'\n', '，', text)

    # 整段乱码检测
    is_heavy = is_garbled(text)

    # 分句
    sentences = re.split(r'(?<=[。！？；])', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if len(sentences) <= 1:
        result = smart_correct(text, enable_llm=enable_llm, force_heavy=is_heavy)
        ENDING_PUNCT = set('。！？；')
        if result and result[-1] not in ENDING_PUNCT and result[-1] != '，':
            result += '。'
        return result

    corrected_parts = []
    for i, sent in enumerate(sentences):
        prev_s = sentences[i - 1] if i > 0 else ""
        next_s = sentences[i + 1] if i < len(sentences) - 1 else ""
        corrected = smart_correct(
            sent, prev_text=prev_s, next_text=next_s,
            enable_llm=enable_llm, force_heavy=is_heavy,
        )
        corrected_parts.append(corrected)

    return _join_with_punctuation(corrected_parts)
