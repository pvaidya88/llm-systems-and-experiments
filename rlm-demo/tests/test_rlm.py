import unittest

from rlm_demo import RLM, LLMClient


class ScriptedLLM(LLMClient):
    def __init__(self, replies):
        self._replies = list(replies)

    def complete(self, messages):
        if not self._replies:
            raise RuntimeError("ScriptedLLM ran out of replies")
        return self._replies.pop(0)


class RLMTests(unittest.TestCase):
    def test_final_direct(self):
        llm = ScriptedLLM(["FINAL(hello)"])
        rlm = RLM(llm)
        result = rlm.answer("q", "ctx")
        self.assertEqual(result, "hello")

    def test_repl_then_final_var(self):
        llm = ScriptedLLM(
            [
                """```repl\nanswer = 'ok'\nprint(answer)\n```""",
                "FINAL_VAR(answer)",
            ]
        )
        rlm = RLM(llm)
        result = rlm.answer("q", "ctx")
        self.assertEqual(result, "ok")


if __name__ == "__main__":
    unittest.main()