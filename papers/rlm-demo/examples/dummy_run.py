from rlm_demo import RLM, LLMClient


class ScriptedLLM(LLMClient):
    def __init__(self, replies):
        self._replies = list(replies)

    def complete(self, messages):
        if not self._replies:
            raise RuntimeError("ScriptedLLM ran out of replies")
        return self._replies.pop(0)


def main():
    context = "Alice has a cat named Snowball. She also has a dog named Rex."
    query = "What is the cat's name?"

    replies = [
        """```repl\nanswer = 'Snowball'\nprint(answer)\n```""",
        "FINAL_VAR(answer)",
    ]

    rlm = RLM(ScriptedLLM(replies))
    result = rlm.answer(query, context)
    print(result)


if __name__ == "__main__":
    main()