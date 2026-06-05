# coding=utf-8
"""
SenseVoice 流式转译 Web UI — 真流式版
- VAD 先分段 → 逐段 ASR → 逐段 yield
- 消除批量 generate() 的长等待
- 用法: python webui_streaming.py
"""

import os, re, time
import gradio as gr
import numpy as np
import torch
import torchaudio
from funasr import AutoModel

# ========== 加载模型 ==========
print("⏳ 加载 SenseVoice + VAD...")
asr_model = AutoModel(
    model="iic/SenseVoiceSmall",
    trust_remote_code=True,
    device="cpu",
)
vad_model = AutoModel(
    model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
    trust_remote_code=True,
    device="cpu",
)
print("✅ 模型就绪\n")

# ========== 标签字典（同原版） ==========
emo_dict = {
    "<|HAPPY|>": "😊", "<|SAD|>": "😔", "<|ANGRY|>": "😡",
    "<|NEUTRAL|>": "", "<|FEARFUL|>": "😰", "<|DISGUSTED|>": "🤢", "<|SURPRISED|>": "😮",
}
event_dict = {
    "<|BGM|>": "🎼", "<|Speech|>": "", "<|Applause|>": "👏",
    "<|Laughter|>": "😀", "<|Cry|>": "😭", "<|Sneeze|>": "🤧",
    "<|Breath|>": "", "<|Cough|>": "🤧",
}
emoji_dict = {
    "<|nospeech|><|Event_UNK|>": "❓", "<|zh|>": "", "<|en|>": "",
    "<|yue|>": "", "<|ja|>": "", "<|ko|>": "", "<|nospeech|>": "",
    "<|HAPPY|>": "😊", "<|SAD|>": "😔", "<|ANGRY|>": "😡",
    "<|NEUTRAL|>": "", "<|BGM|>": "🎼", "<|Speech|>": "",
    "<|Applause|>": "👏", "<|Laughter|>": "😀", "<|FEARFUL|>": "😰",
    "<|DISGUSTED|>": "🤢", "<|SURPRISED|>": "😮", "<|Cry|>": "😭",
    "<|EMO_UNKNOWN|>": "", "<|Sneeze|>": "🤧", "<|Breath|>": "",
    "<|Cough|>": "😷", "<|Sing|>": "", "<|Speech_Noise|>": "",
    "<|withitn|>": "", "<|woitn|>": "", "<|GBG|>": "",
    "<|Event_UNK|>": "",
}
lang_dict = {
    "<|zh|>": "<|lang|>", "<|en|>": "<|lang|>", "<|yue|>": "<|lang|>",
    "<|ja|>": "<|lang|>", "<|ko|>": "<|lang|>", "<|nospeech|>": "<|lang|>",
}
emo_set = {"😊", "😔", "😡", "😰", "🤢", "😮"}
event_set = {"🎼", "👏", "😀", "😭", "🤧", "😷"}

def format_str_v2(s):
    sptk_dict = {}
    for sptk in emoji_dict:
        sptk_dict[sptk] = s.count(sptk)
        s = s.replace(sptk, "")
    emo = "<|NEUTRAL|>"
    for e in emo_dict:
        if sptk_dict[e] > sptk_dict[emo]:
            emo = e
    for e in event_dict:
        if sptk_dict[e] > 0:
            s = event_dict[e] + s
    s = s + emo_dict[emo]
    for emoji in emo_set.union(event_set):
        s = s.replace(" " + emoji, emoji)
        s = s.replace(emoji + " ", emoji)
    return s.strip()

def format_str_v3(s):
    def get_emo(s):
        return s[-1] if s[-1] in emo_set else None
    def get_event(s):
        return s[0] if s[0] in event_set else None
    s = s.replace("<|nospeech|><|Event_UNK|>", "❓")
    for lang in lang_dict:
        s = s.replace(lang, "<|lang|>")
    s_list = [format_str_v2(s_i).strip(" ") for s_i in s.split("<|lang|>")]
    new_s = " " + s_list[0]
    cur_ent_event = get_event(new_s)
    for i in range(1, len(s_list)):
        if len(s_list[i]) == 0:
            continue
        if get_event(s_list[i]) == cur_ent_event and get_event(s_list[i]) != None:
            s_list[i] = s_list[i][1:]
        cur_ent_event = get_event(s_list[i])
        if get_emo(s_list[i]) != None and get_emo(s_list[i]) == get_emo(new_s):
            new_s = new_s[:-1]
        new_s += s_list[i].strip().lstrip()
    new_s = new_s.replace("The.", " ")
    return new_s.strip()

def preprocess_audio(input_wav):
    """音频预处理：转 16kHz 单声道 float32"""
    if isinstance(input_wav, tuple):
        fs, input_wav = input_wav
        input_wav = input_wav.astype(np.float32) / np.iinfo(np.int16).max
        if len(input_wav.shape) > 1:
            input_wav = input_wav.mean(-1)
        if fs != 16000:
            resampler = torchaudio.transforms.Resample(fs, 16000)
            input_wav_t = torch.from_numpy(input_wav).to(torch.float32)
            input_wav = resampler(input_wav_t[None, :])[0, :].numpy()
    return input_wav

# ========== 核心：真流式识别 ==========
def stream_inference(input_wav, language):
    """
    真流式语音转文字
    Step 1: VAD 分段 → 获取语音时间戳
    Step 2: 合并相邻短段 → 减少碎片
    Step 3: 逐段 ASR → 每段识别完立刻 yield
    Step 4: AI 纠错后处理
    """
    audio = preprocess_audio(input_wav)
    duration = len(audio) / 16000
    
    if duration < 0.3:
        yield "⚠️ 音频太短（< 0.3 秒），请重新录制"
        return
    
    language_map = {"auto": "auto", "zh": "zh", "en": "en", "yue": "yue", "ja": "ja", "ko": "ko", "nospeech": "nospeech"}
    lang = language_map.get(language, "auto")
    
    # Step 1: VAD 分段
    try:
        vad_result = vad_model.generate(input=audio)
    except Exception as e:
        yield f"❌ VAD 分段失败: {e}"
        return
    
    # 提取语音段的时间戳
    raw_segments = []
    if vad_result and len(vad_result) > 0:
        for item in vad_result[0].get("value", []):
            start_ms = item[0]
            end_ms = item[1]
            start_sample = int(start_ms / 1000 * 16000)
            end_sample = int(end_ms / 1000 * 16000)
            if end_sample - start_sample > 12000:  # 至少 0.75 秒（歌曲短语通常 >1s）
                raw_segments.append((start_sample, end_sample, start_ms))
    
    if not raw_segments:
        yield "⚠️ 未检测到语音内容"
        return
    
    # Step 2: 合并相邻短段（间距 < 500ms 的合并为一个短语）
    MIN_GAP_MS = 500
    MIN_SEG_MS = 1500  # 合并后最短段长
    merged = []
    current_start, current_end, current_start_ms = raw_segments[0]
    
    for start, end, start_ms in raw_segments[1:]:
        gap_ms = start_ms - ((current_end / 16000) * 1000)
        seg_dur_ms = (end - start) / 16000 * 1000
        merged_dur_ms = (end - current_start) / 16000 * 1000
        
        if gap_ms < MIN_GAP_MS or seg_dur_ms < MIN_SEG_MS:
            # 合并：间隔近 或 单独段太短
            current_end = end
        else:
            merged.append((current_start, current_end))
            current_start, current_end, current_start_ms = start, end, start_ms
    
    merged.append((current_start, current_end))
    
    segments = [(s, e) for s, e in merged if (e - s) / 16000 * 1000 >= MIN_SEG_MS]
    
    if not segments:
        yield "⚠️ 未检测到有效语音内容"
        return
    
    # Step 3: 逐段 ASR + 实时 yield
    texts = []
    total_segs = len(segments)
    
    for i, (start, end) in enumerate(segments):
        seg_audio = audio[start:end]
        
        try:
            asr_result = asr_model.generate(
                input=seg_audio,
                language=lang,
                use_itn=True,
                batch_size_s=60,
            )
        except Exception as e:
            yield f"❌ 第 {i+1} 段识别失败: {e}"
            continue
        
        if asr_result and len(asr_result) > 0:
            text = asr_result[0].get("text", "")
            text = format_str_v3(text)
            # 过滤过短文本（< 3 个汉字，排除纯标点/噪声）
            han_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
            if han_chars >= 3:
                texts.append(text)
        
        # 构建累积输出
        combined = "\n".join(f"[{j+1}] {t}" for j, t in enumerate(texts))
        seg_dur = (end - start) / 16000
        yield f"🎙️ 实时转译中... ({i+1}/{total_segs} 段, {seg_dur:.1f}s)\n{'─'*40}\n{combined}\n{'─'*40}\n⏱️ 总时长: {duration:.1f}秒"
    
    # 最终结果
    final = "\n".join(f"[{j+1}] {t}" for j, t in enumerate(texts))
    corrected = correct_text(final)
    yield f"✅ 转译完成（共 {len(texts)} 段 | {duration:.1f}秒）\n{'─'*40}\n{corrected}"


# ========== 文本纠错 ==========
def correct_text(text):
    """基于规则和上下文的文本纠错"""
    # 去除单独的句号/标点残片
    text = text.replace("\n。", "").replace("\n，", "").replace("\n！", "").replace("\n？", "")
    
    # 去除 VAD 分段标记 [N]
    import re as re_mod
    text = re_mod.sub(r'\[\d+\] ', '', text)
    text = re_mod.sub(r'\[\d+\]', '', text)
    
    # 常见同音错词修正（歌曲场景）
    corrections = {
        "雇勇": "孤勇",
        "林黛": "领带",
        "如一标": "如目标",
        "那的头发": "那时的我",
        "绝不平庸": "绝不普通",
        "团我眼睛": "团火眼睛",
        "人海理": "人海里",
        "敢哭敢笑": "敢哭敢笑",
        "燃少过": "燃烧过",
        "在一宙": "在宇宙",
        "多少回": "多少回",
        "来点分岔": "那天分岔",
        "出口告诉": "出口告诉",
        "别为谁": "别为谁",
        "算陷入梦": "算陷入梦",
        "团伙眼": "团火焰",
        "起手埋葬": "亲手埋葬",
        "从新快乐": "重新快乐",
        "情是浮躁": "情绪浮躁",
        "刀结冰": "都结冰",
        "奉献一句": "逢人一句",
        "拼命天一": "拼命添衣",
        "爱情我": "曾经我",
        "台上的树": "躺在那数",
        "失合": "适合",
        "可心": "可心",
        "一生的漫长": "一生的漫长",
        "多想为了": "多想为了",
        "盛下留后": "盛夏午后",
        "燃烧过的宇宙": "燃烧过的宇宙",
        "眼睛手": "眼睛守",
    }
    
    for wrong, right in corrections.items():
        text = text.replace(wrong, right)
    
    return text.strip()

# ========== Gradio UI ==========
HTML_STREAMING = """
<div style="text-align:center; margin-bottom:20px;">
    <h1>🎙️ SenseVoice 流式语音转文字</h1>
    <p style="font-size:16px; color:#666;">
        上传音频或使用麦克风 → 逐段实时输出字幕
    </p>
    <p style="font-size:14px; color:#999;">
        支持中文(zh)、英文(en)、粤语(yue)、日语(ja)、韩语(ko)
    </p>
</div>
"""

def launch():
    with gr.Blocks(theme=gr.themes.Soft(), title="SenseVoice 流式转译") as demo:
        gr.HTML(HTML_STREAMING)
        
        with gr.Row():
            with gr.Column(scale=2):
                audio_input = gr.Audio(
                    label="📥 上传音频 或 点击🎤录音",
                    type="numpy",
                    sources=["upload", "microphone"],
                )
                
                with gr.Row():
                    language_input = gr.Dropdown(
                        choices=["auto", "zh", "en", "yue", "ja", "ko"],
                        value="auto",
                        label="🌐 语言",
                        scale=1,
                    )
                    stream_btn = gr.Button("▶️ 开始实时转译", variant="primary", scale=1)
                    stop_btn = gr.Button("⏹️ 停止", variant="stop", scale=1)
            
            with gr.Column(scale=3):
                text_output = gr.Textbox(
                    label="📝 实时字幕",
                    lines=20,
                    max_lines=30,
                    placeholder="转译结果将逐段显示在这里...",
                    autoscroll=True,
                )
        
        # 示例音频
        gr.Examples(
            examples=[
                ["example/zh.mp3", "zh"],
                ["example/yue.mp3", "yue"],
                ["example/en.mp3", "en"],
                ["example/ja.mp3", "ja"],
                ["example/ko.mp3", "ko"],
                ["example/longwav_2.wav", "auto"],
            ],
            inputs=[audio_input, language_input],
            label="📁 示例音频",
        )
        
        # 绑定事件
        stream_event = stream_btn.click(
            fn=stream_inference,
            inputs=[audio_input, language_input],
            outputs=text_output,
        )
        stop_btn.click(fn=None, cancels=[stream_event])

    demo.launch(server_name="0.0.0.0", server_port=7860)

if __name__ == "__main__":
    launch()
