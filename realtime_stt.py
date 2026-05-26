#!/usr/bin/env python3
"""
实时语音转文字 v3 - SenseVoice 内置 VAD + 连续录音
用法: python realtime_stt.py
"""

import sys, os, re, time, threading, tempfile
import numpy as np
import sounddevice as sd
import soundfile as sf
from funasr import AutoModel

SAMPLE_RATE = 16000
CHUNK_SECONDS = 3

print("⏳ 加载 SenseVoice + VAD...")
model = AutoModel(
    model="iic/SenseVoiceSmall",
    vad_model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
    vad_kwargs={"max_single_segment_time": 30000},
    trust_remote_code=True,
    device="cpu",
)
print("✅ 模型就绪\n")

audio_buffer = []
lock = threading.Lock()
running = True
seen = set()
last_text = ""

def clean(text):
    return re.sub(r'<\|[^|]+\|>', '', text).strip()

def audio_callback(indata, frames, t, status):
    with lock:
        audio_buffer.append(indata.copy().flatten())

def total_length(buffer_list):
    return sum(len(a) for a in buffer_list)

def worker():
    global audio_buffer, running, last_text, seen
    while running:
        time.sleep(CHUNK_SECONDS)
        
        with lock:
            total = total_length(audio_buffer)
            if total < SAMPLE_RATE * CHUNK_SECONDS:
                continue
            chunk = np.concatenate(audio_buffer)
            audio_buffer = []
        
        # 保存 → 识别
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(tmp.name, chunk, SAMPLE_RATE)
        
        try:
            results = model.generate(input=tmp.name, language="zh", use_itn=True)
            os.unlink(tmp.name)
            if results:
                for seg in results:
                    text = clean(seg.get("text", ""))
                    if text and text != last_text and text not in seen:
                        seen.add(text)
                        last_text = text
                        sys.stdout.write(f"\n📝 {text}\n{'─'*50}\n")
                        sys.stdout.flush()
        except Exception as e:
            print(f"\n⚠️ {e}")

print("🎤 开始监听 (Ctrl+C 停止)\n" + "="*50)

try:
    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                             callback=audio_callback, dtype=np.float32)
    stream.start()
    t = threading.Thread(target=worker, daemon=True)
    t.start()
    
    while running:
        time.sleep(0.3)
        with lock:
            dur = total_length(audio_buffer) / SAMPLE_RATE
        bar = "█" * min(int(dur * 8), 25)
        sys.stdout.write(f"\r🔊 [{bar:<25}] {dur:.1f}s")
        sys.stdout.flush()
except KeyboardInterrupt:
    running = False
    print("\n\n🛑 停止")
    stream.stop()
    stream.close()
