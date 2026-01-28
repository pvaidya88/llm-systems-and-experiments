import unittest

from rlm_demo import RLM, LLMClient, RLMOptions


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

    def test_final_must_be_full_line(self):
        llm = ScriptedLLM(["Here is your answer FINAL(nope) inline."])
        rlm = RLM(llm)
        with self.assertRaises(RuntimeError):
            rlm.answer("q", "ctx")

    def test_final_var_must_be_full_line(self):
        llm = ScriptedLLM(
            [
                """```repl\nanswer = 'ok'\n```""",
                "Inline FINAL_VAR(answer) should be ignored.",
            ]
        )
        rlm = RLM(llm)
        with self.assertRaises(RuntimeError):
            rlm.answer("q", "ctx")

    def test_json_protocol_round_trip(self):
        reply = (
            '{"repl":[{"code":"answer = \\"ok\\""}],"final_var":"answer"}'
        )
        llm = ScriptedLLM([reply])
        options = RLMOptions(protocol="json")
        rlm = RLM(llm, options=options)
        result = rlm.answer("q", "ctx")
        self.assertEqual(result, "ok")


if __name__ == "__main__":
    unittest.main()
