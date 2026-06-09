#!/usr/bin/env python3
"""
SenseVoice 独立转录服务 — 同步 + 流式逐句输出
端点:
  POST /transcribe          → JSON {success, text}   (同步)
  POST /transcribe_stream   → SSE 逐句推送           (流式)
"""

import os, re, sys, tempfile, json, cgi, io
from http.server import HTTPServer, BaseHTTPRequestHandler

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

# ========== HTTP Server ==========
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

    def do_POST(self):
        if self.path == "/transcribe":
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
    port = int(os.environ.get("STT_API_PORT", "7861"))
    server = HTTPServer(("0.0.0.0", port), TranscribeHandler)
    print(f"STT API server listening on port {port}", file=sys.stderr, flush=True)
    server.serve_forever()

if __name__ == "__main__":
    main()
