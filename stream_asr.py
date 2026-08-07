"""
StreamASREngine — 流式 ASR 引擎，用于 realtime_ws_server.py
基于 SenseVoiceSmall 实现增量识别
"""
import numpy as np
import torch
import time
from funasr import AutoModel

class StreamASREngine:
    """流式 ASR 引擎：分块处理音频，逐句输出中间结果"""

    def __init__(self, model="iic/SenseVoiceSmall", device="cpu", language="zh",
                 chunk_size=10, beam_size=1, stability_window=3):
        self.model = AutoModel(
            model=model,
            trust_remote_code=True,
            device=device,
            disable_update=True,
        )
        self.language = language
        self.chunk_size = chunk_size
        self.beam_size = beam_size
        self.stability_window = stability_window

        self._buffer = np.array([], dtype=np.float32)
        self._prev_output = ""
        self._stable_count = 0
        self._speech_started = False
        self._segment_samples = 0

    def on_speech_start(self):
        """VAD 检测到语音开始"""
        self._speech_started = True
        self._buffer = np.array([], dtype=np.float32)
        self._prev_output = ""
        self._stable_count = 0
        self._segment_samples = 0

    def process_chunk(self, audio_chunk, is_last=False):
        """处理一段音频，yield 增量识别结果

        audio_chunk: float32 numpy array, 16kHz
        """
        self._buffer = np.concatenate([self._buffer, audio_chunk])
        self._segment_samples += len(audio_chunk)

        duration_s = len(self._buffer) / 16000
        if duration_s < self.chunk_size and not is_last:
            return

        try:
            res = self.model.generate(
                input=self._buffer,
                language=self.language,
                use_itn=True,
                batch_size_s=60,
            )
        except Exception as e:
            yield {"type": "error", "message": str(e)}
            return

        if res and res[0].get("text"):
            text = res[0]["text"].strip()

            # 移除 SenseVoice 特殊标记
            import re
            text = re.sub(r"<\|[^>]+\|>", "", text)

            if text != self._prev_output:
                self._prev_output = text
                self._stable_count = 0
                yield {"type": "partial", "text": text, "is_final": False}
            else:
                self._stable_count += 1
                if self._stable_count >= self.stability_window or is_last:
                    yield {"type": "partial", "text": text, "is_final": is_last}

    def on_speech_end(self):
        """VAD 检测到语音结束，返回最终结果"""
        self._speech_started = False
        return {
            "type": "result",
            "text": self._prev_output or "",
            "duration_s": self._segment_samples / 16000,
        }
