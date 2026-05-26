#!/bin/bash
# 衡水方言识别测试脚本
# 用法: bash test_hengshui.sh [B站视频链接]

AUDIO_DIR="./hengshui_audio"
mkdir -p "$AUDIO_DIR"

# ---- 方式1: 从B站下载 ----
if [ -n "$1" ]; then
    echo "📥 从B站下载: $1"
    yt-dlp -x --audio-format wav --audio-quality 0 \
        -o "$AUDIO_DIR/%(title)s.%(ext)s" \
        --postprocessor-args "ffmpeg: -ar 16000 -ac 1" \
        "$1"
    echo "✅ 下载完成 → $AUDIO_DIR/"
fi

# ---- 方式2: 直接用麦克风录制 ----
record() {
    echo "🎙️ 录制5秒衡水话..."
    ffmpeg -f avfoundation -i ":0" -t 5 -ar 16000 -ac 1 "$AUDIO_DIR/record.wav" -y 2>/dev/null
    echo "✅ 录制完成 → $AUDIO_DIR/record.wav"
}

# ---- 方式3: 已有音频直接识别 ----
if [ "$1" = "record" ]; then
    record
fi

# 批量测试
echo ""
echo "🔍 开始识别..."
for f in "$AUDIO_DIR"/*.wav; do
    [ -f "$f" ] || continue
    echo "--- $(basename "$f") ---"
    source ~/work/conda/anaconda3/bin/activate base
    python -c "
from funasr import AutoModel
m = AutoModel(model='iic/SenseVoiceSmall', device='cpu', trust_remote_code=True)
r = m.generate(input='$f', language='zh')
for x in r:
    print(f\"  语言: {x.get('key','?')}  |  文本: {x.get('text','?')}\")
"
done

echo ""
echo "💡 示例B站链接（搜索「衡水话」）："
echo "   yt-dlp 'ytsearch5:衡水方言' --get-url"
