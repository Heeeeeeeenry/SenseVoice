#!/usr/bin/env python3
"""
SenseVoice 独立转录服务 — 同步 + 流式逐句输出
端点:
  POST /transcribe          → JSON {success, text}   (同步)
  POST /transcribe_stream   → SSE 逐句推送           (流式)
"""

import os, re, sys, tempfile, json, cgi, io
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from threading import Condition

import numpy as np
import soundfile as sf
import torch
import torchaudio
from funasr import AutoModel
from smart_corrector import smart_correct_paragraph

# ========== 加载模型 ==========
print("Loading SenseVoice + VAD...", file=sys.stderr, flush=True)
asr_model = AutoModel(
    model="iic/SenseVoiceSmall",
    trust_remote_code=True,
    device="cpu",
    disable_update=True,
)
vad_model = AutoModel(
    model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
    trust_remote_code=True,
    device="cpu",
    disable_update=True,
)
print("Models ready", file=sys.stderr, flush=True)

# ========== Resource guard ==========
_CGROUP_LIMIT_PATHS = (
    "/sys/fs/cgroup/memory.max",
    "/sys/fs/cgroup/memory/memory.limit_in_bytes",
)
_CGROUP_USAGE_PATHS = (
    "/sys/fs/cgroup/memory.current",
    "/sys/fs/cgroup/memory/memory.usage_in_bytes",
)

def _read_int(path):
    try:
        value = open(path).read().strip()
        if value == "max": return None
        return int(value)
    except Exception:
        return None

def _read_mem_available_bytes():
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except Exception:
        pass
    return None

def _memory_snapshot():
    limit = next((v for v in (_read_int(p) for p in _CGROUP_LIMIT_PATHS) if v), None)
    usage = next((v for v in (_read_int(p) for p in _CGROUP_USAGE_PATHS) if v is not None), None)
    host_available = _read_mem_available_bytes()
    cgroup_available = None
    if limit and usage is not None and limit < (1 << 60):
        cgroup_available = max(0, limit - usage)
    available_values = [v for v in (cgroup_available, host_available) if v is not None]
    return {
        "limit": limit,
        "usage": usage,
        "host_available": host_available,
        "available": min(available_values) if available_values else None,
    }

def _bytes_to_mb(value):
    if value is None: return None
    return round(value / 1024 / 1024, 1)

def _env_int(name, default):
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return default

def calculate_max_concurrent(snapshot=None):
    """Calculate startup STT inference concurrency.

    STT_MAX_CONCURRENT can be:
      - a positive integer: fixed limit, used as-is
      - "auto"/empty: derive from available memory
    """
    raw = os.environ.get("STT_MAX_CONCURRENT", "auto").strip().lower()
    if raw and raw != "auto":
        try:
            fixed = int(raw)
            if fixed > 0:
                return fixed
        except Exception:
            pass

    snapshot = snapshot or _memory_snapshot()
    available = snapshot.get("available")
    mem_per_job_mb = _env_int("STT_MEM_PER_JOB_MB", 1536)
    reserved_mb = _env_int("STT_RESERVED_MB", 2048)
    hard_max = _env_int("STT_HARD_MAX_CONCURRENT", 8)
    min_concurrent = _env_int("STT_MIN_CONCURRENT", 1)
    fallback = _env_int("STT_FALLBACK_CONCURRENT", 3)

    if available is None or mem_per_job_mb <= 0:
        return max(min_concurrent, min(fallback, hard_max))

    usable_mb = max(0, int(available / 1024 / 1024) - reserved_mb)
    calculated = usable_mb // mem_per_job_mb
    calculated = max(min_concurrent, calculated)
    calculated = min(calculated, hard_max)
    return calculated

def capacity_snapshot():
    snapshot = _memory_snapshot()
    suggested = calculate_max_concurrent(snapshot)
    limiter_status = _inference_limiter.status() if _inference_limiter else {"active": 0, "limit": suggested}
    return {
        "success": True,
        "suggested_concurrent": suggested,
        "active": limiter_status["active"],
        "limit": limiter_status["limit"],
        "memory": {k: _bytes_to_mb(v) for k, v in snapshot.items()},
        "policy": {
            "max_concurrent": os.environ.get("STT_MAX_CONCURRENT", "auto"),
            "mem_per_job_mb": _env_int("STT_MEM_PER_JOB_MB", 1536),
            "reserved_mb": _env_int("STT_RESERVED_MB", 2048),
            "hard_max_concurrent": _env_int("STT_HARD_MAX_CONCURRENT", 8),
        },
    }

def refresh_inference_limit():
    suggested = calculate_max_concurrent()
    if _inference_limiter:
        _inference_limiter.resize(suggested)
    return suggested

def ensure_resource_available():
    """Return (ok, snapshot, message). Reject before OOM instead of crashing."""
    snapshot = _memory_snapshot()
    min_available_mb = int(os.environ.get("STT_MIN_AVAILABLE_MB", "2048"))
    max_usage_pct = float(os.environ.get("STT_MAX_MEMORY_PCT", "85"))
    available = snapshot.get("available")
    if available is not None and available < min_available_mb * 1024 * 1024:
        return False, snapshot, f"STT资源不足：可用内存约 {_bytes_to_mb(available)}MB，低于安全阈值 {min_available_mb}MB"
    limit, usage = snapshot.get("limit"), snapshot.get("usage")
    if limit and usage is not None and limit < (1 << 60):
        usage_pct = usage / limit * 100
        if usage_pct >= max_usage_pct:
            return False, snapshot, f"STT资源不足：容器内存使用率 {usage_pct:.1f}% >= {max_usage_pct:.1f}%"
    return True, snapshot, ""

def send_overloaded(handler, stream=False):
    ok, snapshot, message = ensure_resource_available()
    if ok: return False
    payload = {
        "success": False,
        "error": message,
        "retry_after": 30,
        "memory": {k: _bytes_to_mb(v) for k, v in snapshot.items()},
    }
    print(f"[STT] reject overload: {payload}", file=sys.stderr, flush=True)
    if stream:
        handler.send_response(503)
        handler.send_header("Content-Type", "text/event-stream; charset=utf-8")
        handler.send_header("Retry-After", "30")
        handler.send_header("Cache-Control", "no-cache")
        handler.end_headers()
        handler.wfile.write(sse_event("error", {"message": message, "retry_after": 30}))
        handler.wfile.write(sse_event("done", {}))
    else:
        resp = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        handler.send_response(503)
        handler.send_header("Content-Type", "application/json; charset=utf-8")
        handler.send_header("Retry-After", "30")
        handler.send_header("Content-Length", str(len(resp)))
        handler.end_headers()
        handler.wfile.write(resp)
    return True

# ========== ASR helpers ==========
def format_str_v3(s):
    emoji_dict = {"<|nospeech|><|Event_UNK|>": "?", "<|zh|>": "", "<|en|>": "", "<|yue|>": "", "<|ja|>": "", "<|ko|>": "", "<|nospeech|>": "", "<|HAPPY|>": "😊", "<|SAD|>": "😔", "<|ANGRY|>": "😡", "<|NEUTRAL|>": "", "<|BGM|>": "🎼", "<|Speech|>": "", "<|Applause|>": "👏", "<|Laughter|>": "😀", "<|FEARFUL|>": "😰", "<|DISGUSTED|>": "🤢", "<|SURPRISED|>": "😮", "<|Cry|>": "😭", "<|EMO_UNKNOWN|>": "", "<|Sneeze|>": "🤧", "<|Breath|>": "", "<|Cough|>": "😷", "<|Sing|>": "", "<|Speech_Noise|>": "", "<|withitn|>": "", "<|woitn|>": "", "<|GBG|>": "", "<|Event_UNK|>": ""}
    emo_dict = {"<|HAPPY|>": "😊", "<|SAD|>": "😔", "<|ANGRY|>": "😡", "<|NEUTRAL|>": "", "<|FEARFUL|>": "😰", "<|DISGUSTED|>": "🤢", "<|SURPRISED|>": "😮"}
    event_dict = {"<|BGM|>": "🎼", "<|Speech|>": "", "<|Applause|>": "👏", "<|Laughter|>": "😀", "<|Cry|>": "😭", "<|Sneeze|>": "🤧", "<|Breath|>": "", "<|Cough|>": "🤧"}
    emo_set = {"😊", "😔", "😡", "😰", "🤢", "😮"}
    event_set = {"🎼", "👏", "😀", "😭", "🤧", "😷"}
    lang_dict = {"<|zh|>": "<|lang|>", "<|en|>": "<|lang|>", "<|yue|>": "<|lang|>", "<|ja|>": "<|lang|>", "<|ko|>": "<|lang|>", "<|nospeech|>": "<|lang|>"}
    def _fmt2(t):
        d = {}
        for k in emoji_dict: d[k] = t.count(k); t = t.replace(k, "")
        e = "<|NEUTRAL|>"
        for k in emo_dict:
            if d[k] > d[e]: e = k
        for k in event_dict:
            if d[k] > 0: t = event_dict[k] + t
        t = t + emo_dict[e]
        for em in emo_set | event_set: t = t.replace(" " + em, em).replace(em + " ", em)
        return t.strip()
    def _emo(t): return t[-1] if t[-1] in emo_set else None
    def _evt(t): return t[0] if t[0] in event_set else None
    s = s.replace("<|nospeech|><|Event_UNK|>", "?")
    for lang in lang_dict: s = s.replace(lang, "<|lang|>")
    parts = [_fmt2(p.strip(" ")) for p in s.split("<|lang|>")]
    result = " " + parts[0]
    cur = _evt(result)
    for i in range(1, len(parts)):
        if not parts[i]: continue
        if _evt(parts[i]) == cur and _evt(parts[i]) is not None: parts[i] = parts[i][1:]
        cur = _evt(parts[i])
        if _emo(parts[i]) is not None and _emo(parts[i]) == _emo(result): result = result[:-1]
        result += parts[i].strip().lstrip()
    return result.strip().replace("The.", " ")

def load_audio(audio_path):
    audio, sr = sf.read(audio_path, dtype="float32")
    if len(audio.shape) > 1: audio = audio.mean(axis=1)
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        audio = resampler(torch.from_numpy(audio).float()[None, :])[0, :].numpy()
    return audio

def vad_segment(audio):
    vad_result = vad_model.generate(input=audio)
    if not vad_result or not vad_result[0].get("value"): return []
    raw_segments = []
    for item in vad_result[0]["value"]:
        start_ms, end_ms = int(item[0]), int(item[1])
        start_sample = int(start_ms * 16000 / 1000)
        end_sample = int(end_ms * 16000 / 1000)
        if end_sample - start_sample > 12000: raw_segments.append((start_sample, end_sample, start_ms))
    if not raw_segments: return []
    merged = []
    cur_start, cur_end, cur_ms = raw_segments[0]
    for start, end, start_ms in raw_segments[1:]:
        gap_ms = start_ms - ((cur_end / 16000) * 1000)
        dur_ms = (end - start) / 16000 * 1000
        if gap_ms < 300 or dur_ms < 800: cur_end = end
        else: merged.append((cur_start, cur_end)); cur_start, cur_end, cur_ms = start, end, start_ms
    merged.append((cur_start, cur_end))
    return [(s, e) for s, e in merged if (e - s) / 16000 * 1000 >= 800]

def sse_event(event_type, data):
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event_type}\ndata: {payload}\n\n".encode("utf-8")

def transcribe_sync(audio_path):
    audio = load_audio(audio_path)
    if len(audio) / 16000 < 0.3: return "音频太短"
    segments = vad_segment(audio)
    if not segments: return "未检测到有效语音"
    lines = []
    # 每次推理前刷新并发窗口，运行中可随内存水位动态收缩/扩张
    refresh_inference_limit()
    with _inference_limiter:
        for start, end in segments:
            res = asr_model.generate(input=audio[start:end], language="auto", use_itn=True, batch_size_s=60)
            if res and res[0].get("text"):
                text = format_str_v3(res[0]["text"]).strip().rstrip("。！？；，")
                text = re.sub(r'<\|[^>]+\|>', '', text)
                if sum(1 for c in text if '\u4e00' <= c <= '\u9fff') >= 3: lines.append(text)
    raw_text = "\n".join(lines)
    if not raw_text: return "未识别出有效文本"
    return smart_correct_paragraph(raw_text, enable_llm=True)

def transcribe_streaming(audio, wfile):
    if len(audio) / 16000 < 0.3:
        wfile.write(sse_event("error", {"message": "音频太短"}))
        wfile.write(sse_event("done", {})); return
    segments = vad_segment(audio)
    if not segments:
        wfile.write(sse_event("error", {"message": "未检测到有效语音"}))
        wfile.write(sse_event("done", {})); return
    total = len(segments)
    lines = []
    # 每次推理前刷新并发窗口，运行中可随内存水位动态收缩/扩张
    refresh_inference_limit()
    with _inference_limiter:
        for i, (start, end) in enumerate(segments):
            res = asr_model.generate(input=audio[start:end], language="auto", use_itn=True, batch_size_s=60)
            if res and res[0].get("text"):
                text = format_str_v3(res[0]["text"]).strip().rstrip("。！？；，")
                text = re.sub(r'<\|[^>]+\|>', '', text)
                if sum(1 for c in text if '\u4e00' <= c <= '\u9fff') >= 3:
                    lines.append(text)
                    wfile.write(sse_event("chunk", {"text": text, "index": i, "total": total}))
                    wfile.flush()
    if not lines:
        wfile.write(sse_event("error", {"message": "未识别出有效文本"}))
        wfile.write(sse_event("done", {})); return
    corrected = smart_correct_paragraph("\n".join(lines), enable_llm=True)
    wfile.write(sse_event("result", {"text": corrected}))
    wfile.write(sse_event("done", {}))

# ========== Threading server with concurrency limit ==========
# ThreadingMixIn 让每个请求在独立线程中处理
# _inference_limiter 限制同时进行模型推理的线程数，防止 CPU/内存打爆
class ThreadingTranscribeServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True  # 主进程退出时自动清理线程


class DynamicLimiter:
    def __init__(self, limit):
        self.limit = max(1, int(limit))
        self.active = 0
        self.cond = Condition()

    def resize(self, limit):
        limit = max(1, int(limit))
        with self.cond:
            self.limit = limit
            self.cond.notify_all()

    def acquire(self):
        with self.cond:
            while self.active >= self.limit:
                self.cond.wait()
            self.active += 1

    def release(self):
        with self.cond:
            self.active = max(0, self.active - 1)
            self.cond.notify_all()

    def status(self):
        with self.cond:
            return {"active": self.active, "limit": self.limit}

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.release()
        return False


_inference_limiter = None  # 在 main() 中初始化


class TranscribeHandler(BaseHTTPRequestHandler):
    def _read_file(self):
        """Parse multipart upload, save to temp file, return path"""
        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in content_type:
            self.send_error(400, "Expecting multipart/form-data")
            return None
        # Read raw body into BytesIO for cgi.FieldStorage
        length = int(self.headers.get("Content-Length", "0"))
        body = io.BytesIO(self.rfile.read(length))
        form = cgi.FieldStorage(fp=body, headers=self.headers, environ={"REQUEST_METHOD": "POST", "CONTENT_TYPE": content_type})
        file_item = form["file"]
        if file_item is None or file_item.file is None:
            self.send_error(400, "No file uploaded")
            return None
        suffix = os.path.splitext(file_item.filename or "audio.wav")[1] or ".wav"
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        tmp.write(file_item.file.read())
        tmp.close()
        return tmp.name

    def do_GET(self):
        if self.path in ("/capacity", "/healthz"):
            if self.path == "/capacity":
                refresh_inference_limit()
            payload = capacity_snapshot()
            resp = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(resp)))
            self.end_headers()
            self.wfile.write(resp)
            return
        self.send_error(404)

    def do_POST(self):
        if self.path == "/transcribe":
            if send_overloaded(self): return
            tmp_path = self._read_file()
            if tmp_path is None: return
            try:
                text = transcribe_sync(tmp_path)
                resp = json.dumps({"success": True, "text": text}, ensure_ascii=False).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(resp)))
                self.end_headers()
                self.wfile.write(resp)
            except Exception as e:
                resp = json.dumps({"success": False, "error": str(e)}, ensure_ascii=False).encode("utf-8")
                self.send_response(500)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.end_headers()
                self.wfile.write(resp)
            finally:
                os.unlink(tmp_path)

        elif self.path == "/transcribe_stream":
            if send_overloaded(self, stream=True): return
            tmp_path = self._read_file()
            if tmp_path is None: return
            try:
                audio = load_audio(tmp_path)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("X-Accel-Buffering", "no")
                self.end_headers()
                self.wfile.flush()
                transcribe_streaming(audio, self.wfile)
            except Exception as e:
                try:
                    self.wfile.write(sse_event("error", {"message": str(e)}))
                    self.wfile.write(sse_event("done", {}))
                    self.wfile.flush()
                except: pass
            finally:
                os.unlink(tmp_path)
        else:
            self.send_error(404)

    def log_message(self, format, *args):
        print(f"[{self.log_date_time_string()}] {args[0]}", file=sys.stderr, flush=True)

def main():
    global _inference_limiter

    port = int(os.environ.get("STT_API_PORT", "7861"))

    # 并发推理数：STT_MAX_CONCURRENT=auto 时根据可用内存自动估算；
    # 设置为数字时使用固定值，便于运维强制覆盖。
    max_concurrent = calculate_max_concurrent()
    _inference_limiter = DynamicLimiter(max_concurrent)

    server = ThreadingTranscribeServer(("0.0.0.0", port), TranscribeHandler)
    print(f"STT API server listening on port {port} (max_concurrent={max_concurrent})",
          file=sys.stderr, flush=True)
    server.serve_forever()

if __name__ == "__main__":
    main()
