# coding=utf-8
"""
SenseVoice 流式转译 Web UI
- 支持上传音频文件或麦克风输入
- 逐段实时输出字幕（相当于实时转译效果）
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
model = AutoModel(
    model="iic/SenseVoiceSmall",
    vad_model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
    vad_kwargs={"max_single_segment_time": 30000},
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

# ========== 核心：流式识别 ==========
def stream_inference(input_wav, language):
    """
    流式语音转文字
    用 SenseVoice + VAD 精准分段 → 逐段输出
    """
    audio = preprocess_audio(input_wav)
    duration = len(audio) / 16000
    
    if duration < 0.3:
        yield "⚠️ 音频太短（< 0.3 秒），请重新录制"
        return
    
    language_map = {"auto": "auto", "zh": "zh", "en": "en", "yue": "yue", "ja": "ja", "ko": "ko", "nospeech": "nospeech"}
    lang = language_map.get(language, "auto")
    
    # 用 VAD 分段识别
    try:
        raw_results = model.generate(
            input=audio,
            language=lang,
            use_itn=True,
            batch_size_s=60,
            merge_vad=True,
        )
    except Exception as e:
        yield f"❌ 识别失败: {e}"
        return
    
    if not raw_results:
        yield "⚠️ 未检测到语音内容"
        return
    
    # 逐段输出（模拟实时字幕）
    segments = []
    total_segs = len(raw_results)
    
    for i, seg in enumerate(raw_results):
        text = seg.get("text", "")
        text = format_str_v3(text)
        if text.strip():
            segments.append(text)
        
        # 构建累积文本
        combined = "\n".join(f"[{j+1}] {t}" for j, t in enumerate(segments))
        
        # 添加进度和时长信息
        header = f"🎙️ 实时转译中... ({i+1}/{total_segs} 段)"
        yield f"{header}\n{'─'*40}\n{combined}\n{'─'*40}\n⏱️ 总时长: {duration:.1f}秒"
        
        # 模拟流式延迟（可选，让字幕更有"实时"感）
        time.sleep(0.15)
    
    # 最终结果
    final = "\n".join(f"[{j+1}] {t}" for j, t in enumerate(segments))
    yield f"✅ 转译完成（共 {total_segs} 段 | {duration:.1f}秒）\n{'─'*40}\n{final}"

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
