"""
Token 稳定性检测器
连续 3 次相同的 token → 标记为 STABLE，冻结不再修改
"""

from collections import deque
from typing import List, Tuple


class StabilityDetector:
    def __init__(self, window_size: int = 3):
        self.window_size = window_size
        self.history: deque = deque(maxlen=window_size)
        self.stable_count: List[int] = []
        self.stable_text: str = ""

    def update(self, tokens: List[int], text: str, tokenizer=None) -> Tuple[str, str]:
        self.history.append(tokens)
        if len(self.history) < 2:
            return self.stable_text, text[len(self.stable_text):]

        prev_tokens = self.history[-2]
        curr_tokens = tokens
        min_len = min(len(prev_tokens), len(curr_tokens))
        while len(self.stable_count) < min_len:
            self.stable_count.append(0)

        for i in range(min_len):
            if prev_tokens[i] == curr_tokens[i]:
                self.stable_count[i] += 1
            else:
                self.stable_count[i] = 1

        stable_idx = -1
        for i in range(min_len):
            if self.stable_count[i] >= self.window_size - 1:
                stable_idx = i
            else:
                break

        if stable_idx >= 0 and tokenizer and stable_idx + 1 < len(curr_tokens):
            try:
                stable_subset = curr_tokens[:stable_idx + 1]
                new_stable = tokenizer.decode(stable_subset)
                if len(new_stable) > len(self.stable_text):
                    self.stable_text = new_stable
            except Exception:
                pass

        if text.startswith(self.stable_text):
            unstable = text[len(self.stable_text):]
        else:
            unstable = text
        return self.stable_text, unstable

    def reset(self):
        self.history.clear()
        self.stable_count = []
        self.stable_text = ""
