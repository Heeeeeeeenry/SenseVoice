"""
流式 ASR 引擎 — 整合 SenseVoice + Stability + Revision
"""

import sys
import numpy as np
from typing import Generator, Dict

from streaming_sensevoice import StreamingSenseVoice

from .stability import StabilityDetector
from .revision import TextReviser


class StreamASREngine:
    def __init__(
        self,
        model: str = "iic/SenseVoiceSmall",
        device: str = "cpu",
        language: str = "zh",
        textnorm: bool = True,
        chunk_size: int = 10,
        padding: int = 8,
        beam_size: int = 1,
        stability_window: int = 3,
    ):
        self.asr = StreamingSenseVoice(
            model=model, device=device, language=language,
            textnorm=textnorm, chunk_size=chunk_size,
            padding=padding, beam_size=beam_size,
        )
        self.stability = StabilityDetector(window_size=stability_window)
        self.reviser = TextReviser()
        self.speech_id = 0
        self.accumulated_text = ""

    def process_chunk(self, audio: np.ndarray, is_last: bool) -> Generator[Dict, None, None]:
        for result in self.asr.streaming_inference(audio, is_last):
            text = result.get("text", "")
            tokens = result.get("tokens", [])
            ts = result.get("timestamps", [])

            if not text:
                continue

            if tokens:
                stable_text, unstable_text = self.stability.update(tokens, text, self.asr.tokenizer)
            else:
                stable_text = self.stability.stable_text
                unstable_text = text[len(stable_text):] if text.startswith(stable_text) else text

            revision = self.reviser.revise(text, stable_text)
            if revision["is_changed"]:
                self.accumulated_text = text
                yield {
                    "type": "transcription",
                    "id": self.speech_id,
                    "stable": revision["stable"],
                    "unstable": revision["unstable"],
                    "is_final": is_last,
                    "text": text,
                    "timestamps": ts,
                }

    def on_speech_start(self):
        self.speech_id += 1
        self.asr.reset()
        self.stability.reset()
        self.reviser.reset()
        self.accumulated_text = ""

    def on_speech_end(self) -> Dict:
        return {
            "type": "transcription", "id": self.speech_id,
            "stable": self.accumulated_text, "unstable": "",
            "is_final": True, "text": self.accumulated_text,
            "timestamps": [],
        }

    def reset(self):
        self.asr.reset()
        self.stability.reset()
        self.reviser.reset()
        self.speech_id = 0
        self.accumulated_text = ""
