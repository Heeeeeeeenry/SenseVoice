#!/usr/bin/env python3
"""
SenseVoice Web UI — 逐字流式输出（ChatGPT/DeepSeek 同款效果）

流程:
  ASR → 将文本喂给 AppendEngine.start_typing()
  → 每 30ms yield 一次，每次多显示一个字
  → 打完一行后 commit，后台纠错 → 原地替换

效果: 字一个接一个蹦出来，不是整段弹出
"""

import difflib
import os, re, time, queue, threading, tempfile
import numpy as np
import soundfile
import gradio as gr
import torch
import torchaudio
from funasr import AutoModel

from corrector import get_corrector

CORRECTOR_ENABLED = True
corrector = get_corrector(backend="dictionary")

# ═══════════════════════════════════════════════════════════════
print("⏳ 加载 SenseVoice + VAD 模型...")
model = AutoModel(
    model="iic/SenseVoiceSmall",
    vad_model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
    vad_kwargs={"max_single_segment_time": 30000},
    trust_remote_code=True,
    device="cpu",
)
print("✅ 模型就绪\n")

SAMPLE_RATE = 16000
TYPING_SPEED = 0.03  # 每字间隔 30ms（≈33 字/秒）

# ═══════════════════════════════════════════════════════════════
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
        if get_event(s_list[i]) == cur_ent_event and get_event(s_list[i]) is not None:
            s_list[i] = s_list[i][1:]
        cur_ent_event = get_event(s_list[i])
        if get_emo(s_list[i]) is not None and get_emo(s_list[i]) == get_emo(new_s):
            new_s = new_s[:-1]
        new_s += s_list[i].strip().lstrip()
    new_s = new_s.replace("The.", " ")
    return new_s.strip()

def preprocess(input_wav):
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


# ═══════════════════════════════════════════════════════════════
# 逐字打字引擎
# ═══════════════════════════════════════════════════════════════
class AppendEngine:
    """
    统一输出区 + 逐字打字动画:
    - lines: 已完成的行 [(text, corrected_or_None, seg_id)]
    - _typing_target: 当前正在打的完整文本
    - _typing_pos: 已打出的字符数
    - tick() 每次前进一个字符
    """
    def __init__(self):
        self.lock = threading.Lock()
        self.lines = []            # [(orig, corr_or_None, seg_id)]
        self._next_id = 0
        self._typing_target = ""   # 正在打字的完整文本
        self._typing_pos = 0       # 已打出字符数
        self._typing_seg_id = -1   # 当前打字行的 seg_id

    def start_typing(self, text: str) -> int:
        """开始逐字打出一行"""
        seg_id = self._next_id
        self._next_id += 1
        with self.lock:
            self._typing_target = text
            self._typing_pos = 0
            self._typing_seg_id = seg_id
        return seg_id

    def tick(self) -> bool:
        """前进一个字符。返回 True 表示这行已打完"""
        with self.lock:
            if self._typing_pos < len(self._typing_target):
                self._typing_pos += 1
            return self._typing_pos >= len(self._typing_target)

    def commit_current(self):
        """当前打字行打完 → 挪到 completed lines"""
        with self.lock:
            if self._typing_target:
                self.lines.append([self._typing_target, None, self._typing_seg_id])
            self._typing_target = ""
            self._typing_pos = 0
            self._typing_seg_id = -1

    def is_typing_done(self) -> bool:
        with self.lock:
            return self._typing_pos >= len(self._typing_target)

    def apply_correction(self, seg_id: int, corrected: str):
        with self.lock:
            for entry in self.lines:
                if entry[2] == seg_id:
                    entry[1] = corrected
                    break

    def snapshot_html(self) -> str:
        """生成 HTML"""
        with self.lock:
            parts = []
            # 已完成行
            for orig, corr, seg_id in self.lines:
                if corr is not None and corr != orig:
                    diff = _diff_html(orig, corr)
                    parts.append(f'<div class="line corrected">{diff}</div>')
                elif corr == orig:
                    parts.append(f'<div class="line">{_escape(orig)}</div>')
                else:
                    parts.append(f'<div class="line pending">{_escape(orig)}</div>')
            # 正在打字行
            if self._typing_target:
                partial = self._typing_target[:self._typing_pos]
                cursor = '<span class="cursor">|</span>'
                parts.append(f'<div class="line typing">{_escape(partial)}{cursor}</div>')
            inner = "\n".join(parts) if parts else '<div class="line typing"><span class="cursor">|</span></div>'
            return f'<div class="output-area" id="output-area">{inner}</div>'


def _escape(t):
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def _diff_html(original, corrected):
    matcher = difflib.SequenceMatcher(None, original, corrected)
    parts = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            parts.append(_escape(original[i1:i2]))
        elif tag == "replace":
            parts.append(f'<span class="wrong">{_escape(original[i1:i2])}</span>')
            parts.append(f'<span class="fixed">{_escape(corrected[j1:j2])}</span>')
        elif tag == "delete":
            parts.append(f'<span class="wrong">{_escape(original[i1:i2])}</span>')
        elif tag == "insert":
            parts.append(f'<span class="fixed">{_escape(corrected[j1:j2])}</span>')
    return "".join(parts)


def _correction_worker(engine: AppendEngine, seg_id: int, raw_text: str):
    if not CORRECTOR_ENABLED or len(raw_text.strip()) < 3:
        engine.apply_correction(seg_id, raw_text)
        return
    try:
        result = corrector.correct(raw_text)
        engine.apply_correction(seg_id, result["corrected"])
    except Exception:
        engine.apply_correction(seg_id, raw_text)


def _type_out(engine, text, speed=TYPING_SPEED):
    """逐字打出 text 的生成器"""
    engine.start_typing(text)
    # 纠错线程（打字期间同步开启）
    seg_id = engine._typing_seg_id
    threading.Thread(target=_correction_worker, args=(engine, seg_id, text), daemon=True).start()
    while not engine.is_typing_done():
        engine.tick()
        yield
        time.sleep(speed)
    engine.commit_current()


# ═══════════════════════════════════════════════════════════════
# 麦克风实时流式
# ═══════════════════════════════════════════════════════════════
try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except ImportError:
    HAS_SOUNDDEVICE = False

def mic_stream_generator(language: str, chunk_seconds: int = 3):
    if not HAS_SOUNDDEVICE:
        yield "❌ 麦克风不可用: pip install sounddevice"
        return

    engine = AppendEngine()
    audio_queue = queue.Queue()
    result_queue = queue.Queue()
    stop_event = threading.Event()
    status = {"total_seconds": 0, "segments_found": 0}
    seen_texts = set()
    lang_map = {"auto": "auto", "zh": "zh", "en": "en", "yue": "yue", "ja": "ja", "ko": "ko"}
    lang = lang_map.get(language, "auto")

    # ── 滑动窗口参数 ──
    window_samples = SAMPLE_RATE * chunk_seconds          # 窗口大小（3s）
    step_samples = SAMPLE_RATE * max(1, chunk_seconds // 2)  # 步长（1.5s），窗口的一半
    ring_buffer = np.array([], dtype=np.float32)
    asr_busy = threading.Lock()  # 防止 ASR 线程堆积

    def audio_callback(indata, frames, time_info, status_flags):
        if stop_event.is_set():
            raise sd.CallbackStop
        audio_queue.put(indata.copy().flatten())

    def run_asr(audio_block):
        """独立线程跑 ASR，不阻塞音频采集"""
        if not asr_busy.acquire(blocking=False):
            return  # 上一轮还在跑，跳过
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        try:
            soundfile.write(tmp.name, audio_block, SAMPLE_RATE)
            results = model.generate(input=tmp.name, language=lang, use_itn=True, batch_size_s=60, merge_vad=True)
        except Exception as e:
            result_queue.put(f"__ERR__{e}")
            return
        finally:
            try: os.unlink(tmp.name)
            except OSError: pass
            asr_busy.release()
        if results:
            for seg in results:
                text = format_str_v3(seg.get("text", ""))
                if text and text not in seen_texts:
                    seen_texts.add(text)
                    status["segments_found"] += 1
                    result_queue.put(text)

    def asr_orchestrator():
        """滑动窗口调度：每 step_samples 触发一次 ASR"""
        nonlocal ring_buffer
        last_trigger_samples = 0
        while not stop_event.is_set():
            # 收集音频到环形缓冲区
            new_chunks = []
            while True:
                try:
                    new_chunks.append(audio_queue.get(timeout=0.1))
                except queue.Empty:
                    break
            if new_chunks:
                ring_buffer = np.concatenate([ring_buffer] + new_chunks)
                status["total_seconds"] = len(ring_buffer) / SAMPLE_RATE

            # 达到步长触发 ASR（滑动窗口：取最近 window_samples）
            if len(ring_buffer) >= last_trigger_samples + step_samples:
                start = max(0, len(ring_buffer) - window_samples)
                window = ring_buffer[start:].copy()
                last_trigger_samples = len(ring_buffer)
                threading.Thread(target=run_asr, args=(window,), daemon=True).start()

            time.sleep(0.05)

        result_queue.put("__DONE__")

    threading.Thread(target=asr_orchestrator, daemon=True).start()
    try:
        stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=audio_callback, dtype=np.float32)
        stream.start()
    except Exception as e:
        yield f"❌ 麦克风错误: {e}"
        return

    # 初始状态
    yield engine.snapshot_html() or "🎤 监听中..."

    while True:
        try:
            msg = result_queue.get_nowait()
        except queue.Empty:
            yield engine.snapshot_html()
            time.sleep(0.05)
            continue

        if isinstance(msg, str) and msg.startswith("__ERR__"):
            # 错误消息也逐字打出
            err_text = f"❌ {msg[7:]}"
            for _ in _type_out(engine, err_text):
                yield engine.snapshot_html()
        elif msg == "__DONE__":
            break
        else:
            # 逐字打出！
            for _ in _type_out(engine, msg):
                yield engine.snapshot_html()

    stop_event.set()
    stream.stop()
    stream.close()

    # 等纠错
    for _ in range(15):
        still_pending = any(e[1] is None for e in engine.lines)
        if not still_pending:
            break
        yield engine.snapshot_html()
        time.sleep(0.2)

    final = engine.snapshot_html()
    if status["segments_found"] > 0:
        final += f"\n\n---\n✅ 完成 — {status['segments_found']} 句 | {status['total_seconds']:.0f}s"
    yield final


# ═══════════════════════════════════════════════════════════════
# 音频文件流式处理
# ═══════════════════════════════════════════════════════════════
def file_stream_generator(input_wav, language: str):
    audio = preprocess(input_wav)
    if audio is None or len(audio) == 0:
        yield "⚠️ 无效音频"
        return
    duration = len(audio) / SAMPLE_RATE
    if duration < 0.5:
        yield "⚠️ 音频太短"
        return

    lang_map = {"auto": "auto", "zh": "zh", "en": "en", "yue": "yue", "ja": "ja", "ko": "ko"}
    lang = lang_map.get(language, "auto")
    seen_texts = set()
    engine = AppendEngine()

    yield engine.snapshot_html() if engine.snapshot_html() else "📂 准备中..."

    WINDOW_SAMPLES = SAMPLE_RATE * 30
    chunks = [audio] if duration <= 60 else [
        audio[i:i + WINDOW_SAMPLES] for i in range(0, len(audio), WINDOW_SAMPLES)
    ]

    def _process_chunk(chunk_wav, result_box):
        """处理单个音频块 → 写入 result_box"""
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        try:
            soundfile.write(tmp.name, chunk_wav, SAMPLE_RATE)
            result_box["results"] = model.generate(
                input=tmp.name, language=lang, use_itn=True, batch_size_s=60, merge_vad=True)
        except Exception as e:
            result_box["error"] = str(e)
        finally:
            try: os.unlink(tmp.name)
            except OSError: pass
            result_box["done"] = True

    # ── 流水线：在上一个窗口打字时预加载下一个 ──
    next_box = {"results": None, "error": None, "done": False}
    # 启动第一个窗口
    if chunks:
        threading.Thread(target=_process_chunk, args=(chunks[0], next_box), daemon=True).start()

    try:
        for i, chunk_wav in enumerate(chunks):
            cur_box = next_box

            # 预启动下一个窗口（如果还有）
            if i + 1 < len(chunks):
                next_box = {"results": None, "error": None, "done": False}
                threading.Thread(target=_process_chunk, args=(chunks[i + 1], next_box), daemon=True).start()

            # 等当前窗口完成
            while not cur_box["done"]:
                yield engine.snapshot_html()
                time.sleep(0.1)

            if cur_box["error"]:
                err_text = f"❌ {cur_box['error']}"
                for _ in _type_out(engine, err_text):
                    yield engine.snapshot_html()
                break

            if cur_box["results"]:
                for seg in cur_box["results"]:
                    text = format_str_v3(seg.get("text", ""))
                    if text and text not in seen_texts:
                        seen_texts.add(text)
                        for _ in _type_out(engine, text):
                            yield engine.snapshot_html()
    except Exception as e:
        for _ in _type_out(engine, f"❌ {e}"):
            yield engine.snapshot_html()

    # 等纠错
    for _ in range(15):
        still_pending = any(e[1] is None for e in engine.lines)
        if not still_pending:
            break
        yield engine.snapshot_html()
        time.sleep(0.2)

    final = engine.snapshot_html()
    if seen_texts:
        final += f"\n\n---\n✅ 完成 — {duration:.0f}s | {len(seen_texts)} 句"
    else:
        final += "\n\n⚠️ 未检测到语音"
    yield final


# ═══════════════════════════════════════════════════════════════
def transcribe(audio_input, mode: str, language: str, chunk_seconds: int, enable_correction: bool = True):
    global CORRECTOR_ENABLED
    CORRECTOR_ENABLED = enable_correction
    if audio_input is None:
        yield "⚠️ 请先录音或上传音频"
        return
    if mode == "mic_stream":
        yield from mic_stream_generator(language, chunk_seconds)
    else:
        yield from file_stream_generator(audio_input, language)


# ═══════════════════════════════════════════════════════════════
CSS = """
.output-area { 
    font-family: 'SF Mono', 'Menlo', 'Monaco', 'Courier New', monospace;
    font-size: 16px; line-height: 1.9; white-space: pre-wrap;
    padding: 16px 20px; background: #ffffff; color: #1a1a1a;
    border-radius: 8px; min-height: 400px;
}
#output-box {
    max-height: 600px;
    overflow-y: auto !important;
    border: 1px solid #ddd;
    border-radius: 8px;
}
.line { padding: 2px 0; color: #1a1a1a; }
.line.typing { color: #1a1a1a; }
.line.pending { color: #1a1a1a; }

/* 光标闪烁 */
@keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: 0; } }
span.cursor { animation: blink 0.8s infinite; color: #58a6ff; font-weight: bold; }

/* 纠错动画 */
@keyframes fix-wrong { 
    0% { background: #da363388; color: #1a1a1a; text-decoration: line-through; }
    100% { text-decoration: line-through; color: #8b949e; background: transparent; }
}
@keyframes fix-correct {
    0% { background: #3fb95088; color: #1a1a1a; font-weight: bold; }
    100% { color: #7ee787; background: transparent; }
}
span.wrong { animation: fix-wrong 2s ease-out forwards; display: inline; }
span.fixed { animation: fix-correct 2s ease-out forwards; display: inline; }
"""

HEADER_HTML = """
<div style="text-align:center; padding:8px 0;">
    <h2 style="margin:0;">🎙️ SenseVoice · 逐字流式转译</h2>
    <p style="color:#888; font-size:0.85em; margin:4px 0;">
        像 ChatGPT 一样逐字输出 | <span style="color:#ff7b72;">错字高亮</span> → <span style="color:#7ee787;">纠正</span>
    </p>
</div>
"""


def launch():
    with gr.Blocks(
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate"),
        css=CSS,
        title="SenseVoice 逐字流式转译",
        head="""<script>
(function() {
    var state = window._autoScrollState || { userScrolledUp: false };
    window._autoScrollState = state;
    var TICK = 50;
    var boxRef = null;

    function getBox() {
        if (boxRef) return boxRef;
        boxRef = document.getElementById("output-box");
        if (boxRef) {
            boxRef.addEventListener("scroll", function() {
                var atBottom = boxRef.scrollHeight - boxRef.scrollTop - boxRef.clientHeight < 30;
                state.userScrolledUp = !atBottom;
            });
        }
        return boxRef;
    }

    setInterval(function() {
        var box = getBox();
        if (!box) return;
        if (!state.userScrolledUp) {
            box.scrollTop = box.scrollHeight;
        }
    }, TICK);
})();
</script>""",
    ) as demo:
        gr.HTML(HEADER_HTML)

        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=280):
                mode_select = gr.Radio(
                    choices=[("🎤 麦克风流式", "mic_stream"), ("📂 音频文件", "file")],
                    value="file", label="📌 输入模式",
                )
                audio_input = gr.Audio(
                    label="📥 音频输入", type="numpy", sources=["upload", "microphone"],
                )
                with gr.Accordion("⚙️ 设置", open=False):
                    language_select = gr.Dropdown(
                        choices=["auto", "zh", "en", "yue", "ja", "ko"],
                        value="auto", label="🌐 语言",
                    )
                    chunk_slider = gr.Slider(
                        minimum=2, maximum=5, value=3, step=1,
                        label="⏱ 麦克风处理间隔 (秒)",
                    )
                    correct_checkbox = gr.Checkbox(value=True, label="🔧 启用纠错")
                transcribe_btn = gr.Button("▶️ 开始转译", variant="primary", size="lg")

            with gr.Column(scale=2):
                output_box = gr.HTML(
                    value="<div class='output-area' id='output-area'><span class='cursor'>|</span></div>",
                    elem_id="output-box",
                )

        transcribe_btn.click(
            fn=transcribe,
            inputs=[audio_input, mode_select, language_select, chunk_slider, correct_checkbox],
            outputs=output_box,
        )

    demo.queue(default_concurrency_limit=2)
    demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True)


if __name__ == "__main__":
    launch()
