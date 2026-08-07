"""
文本修订器 — LCP 增量修订
"""


class TextReviser:
    def __init__(self):
        self.prev_text = ""
        self.prev_stable = ""

    def revise(self, new_text: str, stable_text: str) -> dict:
        result = {"stable": stable_text, "unstable": "", "is_changed": False}
        if new_text.startswith(stable_text):
            unstable = new_text[len(stable_text):]
        else:
            lcp = self._lcp(self.prev_text, new_text)
            result["stable"] = new_text[:lcp]
            unstable = new_text[lcp:]
        result["unstable"] = unstable
        if result["stable"] != self.prev_stable or result["unstable"]:
            result["is_changed"] = True
        self.prev_text = new_text
        self.prev_stable = stable_text
        return result

    @staticmethod
    def _lcp(a: str, b: str) -> int:
        n = min(len(a), len(b))
        for i in range(n):
            if a[i] != b[i]:
                return i
        return n

    def reset(self):
        self.prev_text = ""
        self.prev_stable = ""
