# coding=utf-8
"""
SenseVoice 流式转译 Web UI
- VAD 先分段 → 逐段 ASR → AI纠错
- 用法: python webui_streaming.py
"""

import os, re, time, tempfile
import gradio as gr
import numpy as np
import soundfile as sf
import torch
import torchaudio
from funasr import AutoModel
from smart_corrector import smart_correct_paragraph

print("Loading SenseVoice + VAD...")
asr_model = AutoModel(model="iic/SenseVoiceSmall", trust_remote_code=True, device="cpu", disable_update=True)
vad_model = AutoModel(model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch", trust_remote_code=True, device="cpu", disable_update=True)
print("Models ready\n")

def format_str_v3(s):
    emoji_dict = {"<|nospeech|><|Event_UNK|>":"?","<|zh|>":"","<|en|>":"","<|yue|>":"","<|ja|>":"","<|ko|>":"","<|nospeech|>":"","<|HAPPY|>":"😊","<|SAD|>":"😔","<|ANGRY|>":"😡","<|NEUTRAL|>":"","<|BGM|>":"🎼","<|Speech|>":"","<|Applause|>":"👏","<|Laughter|>":"😀","<|FEARFUL|>":"😰","<|DISGUSTED|>":"🤢","<|SURPRISED|>":"😮","<|Cry|>":"😭","<|EMO_UNKNOWN|>":"","<|Sneeze|>":"🤧","<|Breath|>":"","<|Cough|>":"😷","<|Sing|>":"","<|Speech_Noise|>":"","<|withitn|>":"","<|woitn|>":"","<|GBG|>":"","<|Event_UNK|>":""}
    emo_dict = {"<|HAPPY|>":"😊","<|SAD|>":"😔","<|ANGRY|>":"😡","<|NEUTRAL|>":"","<|FEARFUL|>":"😰","<|DISGUSTED|>":"🤢","<|SURPRISED|>":"😮"}
    event_dict = {"<|BGM|>":"🎼","<|Speech|>":"","<|Applause|>":"👏","<|Laughter|>":"😀","<|Cry|>":"😭","<|Sneeze|>":"🤧","<|Breath|>":"","<|Cough|>":"🤧"}
    emo_set = {"😊","😔","😡","😰","🤢","😮"}
    event_set = {"🎼","👏","😀","😭","🤧","😷"}
    lang_dict = {"<|zh|>":"<|lang|>","<|en|>":"<|lang|>","<|yue|>":"<|lang|>","<|ja|>":"<|lang|>","<|ko|>":"<|lang|>","<|nospeech|>":"<|lang|>"}
    def f2(t):
        d={}
        for k in emoji_dict: d[k]=t.count(k); t=t.replace(k,"")
        e="<|NEUTRAL|>"
        for k in emo_dict:
            if d[k]>d[e]: e=k
        for k in event_dict:
            if d[k]>0: t=event_dict[k]+t
        t=t+emo_dict[e]
        for em in emo_set|event_set: t=t.replace(" "+em,em).replace(em+" ",em)
        return t.strip()
    def e2(t): return t[-1] if t[-1] in emo_set else None
    def ev(t): return t[0] if t[0] in event_set else None
    s=s.replace("<|nospeech|><|Event_UNK|>","?")
    for lang in lang_dict: s=s.replace(lang,"<|lang|>")
    ps=[f2(p.strip(" ")) for p in s.split("<|lang|>")]
    r=" "+ps[0]; c=ev(r)
    for i in range(1,len(ps)):
        if not ps[i]: continue
        if ev(ps[i])==c and ev(ps[i]) is not None: ps[i]=ps[i][1:]
        c=ev(ps[i])
        if e2(ps[i]) is not None and e2(ps[i])==e2(r): r=r[:-1]
        r+=ps[i].strip().lstrip()
    return r.strip().replace("The."," ")

def preprocess(input_wav):
    if isinstance(input_wav,tuple):
        fs,input_wav=input_wav
        input_wav=input_wav.astype(np.float32)/np.iinfo(np.int16).max
        if len(input_wav.shape)>1: input_wav=input_wav.mean(-1)
        if fs!=16000:
            r=torchaudio.transforms.Resample(fs,16000)
            input_wav=r(torch.from_numpy(input_wav).float()[None,:])[0,:].numpy()
    return input_wav

def stream_inference(input_wav, language):
    """同步转写（非生成器），返回完整文本。Go 后端通过 /gradio_api/run/ 直接调用。"""
    audio = preprocess(input_wav)
    dur = len(audio)/16000
    if dur < 0.3: return "音频太短"

    lang = {"auto":"auto","zh":"zh","en":"en","yue":"yue","ja":"ja","ko":"ko"}.get(language,"auto")

    # VAD
    try: vr = vad_model.generate(input=audio)
    except Exception as e: return f"VAD failed: {e}"

    raw = []
    if vr and vr[0].get("value"):
        for item in vr[0]["value"]:
            ss = int(item[0]*16000/1000)
            es = int(item[1]*16000/1000)
            if es-ss > 12000: raw.append((ss,es,item[0]))

    if not raw: return "未检测到语音"

    # Merge
    mg = []; cs,ce,cm = raw[0]
    for s,e,m in raw[1:]:
        g = m-((ce/16000)*1000)
        if g<300 or (e-s)/16000*1000<800: ce=e
        else: mg.append((cs,ce)); cs,ce,cm=s,e,m
    mg.append((cs,ce))
    segs = [(s,e) for s,e in mg if (e-s)/16000*1000>=800]
    if not segs: return "未检测到有效语音"

    # ASR
    lines = []
    for s,e in segs:
        try: r = asr_model.generate(input=audio[s:e], language=lang, use_itn=True, batch_size_s=60)
        except: continue
        if r and r[0].get("text"):
            t = format_str_v3(r[0]["text"]).strip().rstrip("。！？；，")
            t = re.sub(r'<\|[^>]+\|>','',t)
            if sum(1 for c in t if '\u4e00'<=c<='\u9fff')>=3: lines.append(t)

    raw = "\n".join(lines)
    if not raw: return "未识别出有效文本"
    return smart_correct_paragraph(raw, enable_llm=True)

HTML = """<div style="text-align:center"><h1>🎙️ SenseVoice 流式语音转文字</h1><p>上传音频 → AI识别+纠错</p></div>"""

def launch():
    with gr.Blocks(theme=gr.themes.Soft(), title="SenseVoice") as demo:
        gr.HTML(HTML)
        with gr.Row():
            with gr.Column(scale=2):
                ai = gr.Audio(label="上传音频", type="numpy", sources=["upload","microphone"])
                with gr.Row():
                    li = gr.Dropdown(["auto","zh","en","yue","ja","ko"], value="auto", label="语言", scale=1)
                    btn = gr.Button("开始转译", variant="primary", scale=1)
            with gr.Column(scale=3):
                to = gr.Textbox(label="转译结果", lines=20, autoscroll=True)
        btn.click(fn=stream_inference, inputs=[ai,li], outputs=to)
    demo.launch(server_name="0.0.0.0", server_port=7860)

if __name__ == "__main__":
    launch()
