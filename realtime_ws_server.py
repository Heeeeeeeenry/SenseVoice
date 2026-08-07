"""
实时流式 ASR WebSocket 服务
Silero VAD + StreamASREngine → WebSocket JSON 事件
"""

import os, sys, io, uuid, json, time, logging
import numpy as np
import soundfile as sf
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from silero_vad import load_silero_vad, VADIterator
from stream_asr import StreamASREngine

# Load VAD model at startup
_vad_model = None

def get_vad_iterator(threshold=0.5, min_silence_ms=550):
    global _vad_model
    if _vad_model is None:
        _vad_model = load_silero_vad()
    return VADIterator(_vad_model, threshold=threshold, sampling_rate=16000,
                       min_silence_duration_ms=min_silence_ms)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("realtime-asr")

HOST = os.getenv("ASR_HOST", "0.0.0.0")
PORT = int(os.getenv("ASR_PORT", "9100"))
DEVICE = os.getenv("SENSEVOICE_DEVICE", "cpu")
MODEL = os.getenv("SENSEVOICE_MODEL", "iic/SenseVoiceSmall")
CHUNK_DURATION = float(os.getenv("CHUNK_DURATION", "0.1"))
SAMPLE_RATE = 16000
VAD_THRESHOLD = float(os.getenv("VAD_THRESHOLD", "0.5"))
VAD_MIN_SILENCE_MS = int(os.getenv("VAD_MIN_SILENCE_MS", "550"))

app = FastAPI(title="Realtime STT")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])

@app.on_event("startup")
async def startup():
    """预加载模型，避免首次请求 OOM"""
    logger.info("Preloading SenseVoice model...")
    try:
        engine = StreamASREngine(model=MODEL, device=DEVICE, language="zh")
        # 做一次假推理来初始化所有组件
        import numpy as np
        dummy = np.zeros(1600, dtype=np.float32)
        list(engine.process_chunk(dummy, True))
        logger.info("Model preloaded successfully")
    except Exception as e:
        logger.warning(f"Model preload warning: {e}")

@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL, "device": DEVICE}

@app.websocket("/asr/stream")
async def asr_stream(ws: WebSocket):
    await ws.accept()
    sid = str(uuid.uuid4())[:8]
    logger.info(f"[{sid}] connected")
    try:
        await _handle(ws, sid)
    except WebSocketDisconnect:
        logger.info(f"[{sid}] disconnected")
    except Exception as e:
        logger.error(f"[{sid}] {e}")

async def _handle(ws, sid):
    engine = StreamASREngine(model=MODEL, device=DEVICE, language="zh",
                             chunk_size=10, beam_size=1, stability_window=3)
    vad = get_vad_iterator(threshold=VAD_THRESHOLD, min_silence_ms=VAD_MIN_SILENCE_MS)

    buf = np.array([], dtype=np.float32)
    # silero-vad 要求正好 512 采样点/帧 (32ms @ 16kHz)
    VAD_FRAME = 512
    ASR_CHUNK = int(CHUNK_DURATION * SAMPLE_RATE)  # 1600 samples for 100ms
    asr_buf = np.array([], dtype=np.float32)
    begin = 0.0
    has_speech = False

    while True:
        data = await ws.receive_bytes()
        try:
            bio = io.BytesIO(data)
            bio.name = "a.mp3"
            samples, sr = sf.read(bio, dtype="float32")
            if sr != SAMPLE_RATE:
                continue
            buf = np.concatenate([buf, samples])
        except Exception:
            try:
                samples = np.frombuffer(data, dtype=np.float32)
                if len(samples) > 0:
                    buf = np.concatenate([buf, samples])
            except Exception:
                continue

        # Process VAD in 512-sample frames
        while len(buf) >= VAD_FRAME:
            frame = buf[:VAD_FRAME]; buf = buf[VAD_FRAME:]
            import torch
            speech_dict = vad(torch.from_numpy(frame), return_seconds=True)

            if speech_dict:
                if "start" in speech_dict:
                    engine.on_speech_start()
                    begin = speech_dict["start"]
                    has_speech = True
                    asr_buf = np.array([], dtype=np.float32)
                    await ws.send_json({"type": "vad", "is_active": True})

                if "end" in speech_dict and has_speech:
                    # Flush remaining ASR buffer
                    if len(asr_buf) > 0:
                        for r in engine.process_chunk(asr_buf, True):
                            await ws.send_json(r)
                    final = engine.on_speech_end()
                    final["begin_at"] = begin
                    final["end_at"] = speech_dict["end"]
                    await ws.send_json(final)
                    await ws.send_json({"type": "vad", "is_active": False})
                    has_speech = False
                    asr_buf = np.array([], dtype=np.float32)
                    continue

            # If speech active, accumulate for ASR
            if has_speech:
                asr_buf = np.concatenate([asr_buf, frame])
                while len(asr_buf) >= ASR_CHUNK:
                    chunk = asr_buf[:ASR_CHUNK]; asr_buf = asr_buf[ASR_CHUNK:]
                    is_last = (not has_speech) and len(asr_buf) < ASR_CHUNK
                    for r in engine.process_chunk(chunk, is_last):
                        await ws.send_json(r)

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting on {HOST}:{PORT} model={MODEL} device={DEVICE}")
    uvicorn.run(app, host=HOST, port=PORT, log_level="info")
