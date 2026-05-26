"""
纠错模块测试脚本
用法: python test_corrector.py
"""

import sys
sys.path.insert(0, ".")

from corrector import TextCorrector

# 模拟 ASR 常见错误
test_cases = [
    # 同音字错误
    "我今天去了一趟北京天汽很好",
    "他在说一遍那个地址我妹记住",
    "这个东西做的很好我很喜欢",
    "好象是这样的对吧",
    "我化了两百块钱买的",
    "请帮我开起那个设置",
    "先到先得不然后面就没有了",
    # 正常文本（不应被改）
    "今天天气很好适合出门",
    "会议室在三楼左手边",
    # 近音字错误（zh/z, sh/s, n/l 混淆）
    "我住在那栋搂的十三层",
    "这是他的份内之事",
    # 混合错误
    "他象往常一样在次来到了老地方发先人已经不在这了",
]

print("=" * 60)
print("纠错模块测试")
print("=" * 60)

# 测试词典纠错（默认，最轻量）
print("\n📖 词典模式 (dictionary)")
print("-" * 60)
c = TextCorrector(backend="dictionary")
for text in test_cases:
    result = c.correct(text)
    if result["original"] != result["corrected"]:
        print(f"  ❌ {result['original']}")
        print(f"  ✅ {result['corrected']}")
        for change in result["changes"]:
            print(f"     └─ {change}")
    else:
        print(f"  ✓  {text} (未修改)")
    print()

# 测试 PyCorrector（如果安装了）
print("\n🔬 模型模式 (pycorrector)")
print("-" * 60)
try:
    c2 = TextCorrector(backend="pycorrector")
    if c2._model is not None:
        for text in test_cases[:5]:  # 只测前5个，省时间
            result = c2.correct(text)
            if result["original"] != result["corrected"]:
                print(f"  ❌ {result['original']}")
                print(f"  ✅ {result['corrected']}")
                for change in result["changes"]:
                    print(f"     └─ {change}")
            else:
                print(f"  ✓  {text} (未修改)")
            print()
except ImportError:
    print("  ⚠️  pycorrector 未安装，跳过。安装: pip install pycorrector")
