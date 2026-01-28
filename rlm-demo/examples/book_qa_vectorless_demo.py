import os
import re
import time

from rlm_demo import LLMClient, OpenAIResponsesClient, RLM, RLMOptions
from rlm_demo.trace import RLMTrace


class ScriptedLLM(LLMClient):
    def __init__(self, replies):
        self._replies = list(replies)

    def complete(self, messages):
        if not self._replies:
            raise RuntimeError("ScriptedLLM ran out of replies")
        return self._replies.pop(0)


def load_book_text():
    base = os.path.dirname(__file__)
    corpus_path = os.environ.get("BOOK_PATH")
    corpus_name = os.environ.get("BOOK_CORPUS", "book_sample.txt")
    if corpus_path:
        path = corpus_path
    else:
        if os.path.isabs(corpus_name):
            path = corpus_name
        else:
            path = os.path.join(base, "data", corpus_name)
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def make_scripted_replies(question):
    repl_block = f"""```repl
import re
question = {question!r}
hits = ctx.bm25_search(question, k=3)
expanded = ctx.expand(hits[0], radius=220) if hits else {{"text": ""}}
snippet = expanded.get("text", "")
m = re.search(r"most common failure was ([^\\(\\.]+)\\((\\d+)%\\)", snippet, re.IGNORECASE)
if not m:
    m = re.search(r"most common failure was ([^\\.]+)", snippet, re.IGNORECASE)
if m:
    label = m.group(1).strip()
    pct = m.group(2).strip() if m.lastindex and m.lastindex >= 2 else None
    answer = f"{{label}} ({{pct}}%)" if pct else label
else:
    answer = snippet.strip()[:200] or "no answer"
print(answer)
```"""
    return [repl_block, "FINAL_VAR(answer)"]


def print_trace(trace):
    print("Trace steps:", trace.steps)
    print("Subcalls:", len(trace.subcalls))
    print("Retrieval calls:", len(trace.retrieval))
    print("Surfaced chars:", trace.surfaced_chars)
    if trace.retrieval:
        print("Top retrieval hit:")
        first = trace.retrieval[0]
        if first.hits:
            hit = first.hits[0]
            print(f"- {first.op}: {hit.ref_id} score={hit.score}")


def build_live_client(model_name):
    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_BASE_URL")
    reasoning_effort = os.environ.get("OPENAI_REASONING_EFFORT")
    text_verbosity = os.environ.get("OPENAI_TEXT_VERBOSITY", "medium")
    return OpenAIResponsesClient(
        model=model_name,
        api_key=api_key,
        base_url=base_url,
        reasoning_effort=reasoning_effort,
        text_verbosity=text_verbosity,
    )


def main():
    book_text = load_book_text()
    question = "Which failure mode was most common and what share?"
    expected = "silent truncation (43%)"

    trace = RLMTrace()
    options = RLMOptions(require_repl=True, retry_on_invalid=True, trace=trace)

    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        model = os.environ.get("OPENAI_MODEL", "gpt-5.2")
        root = build_live_client(model)
        rlm = RLM(root_llm=root, sub_llm=root, options=options)
    else:
        replies = make_scripted_replies(question)
        rlm = RLM(root_llm=ScriptedLLM(replies), options=options)

    start = time.perf_counter()
    answer = rlm.answer(question, book_text)
    elapsed = time.perf_counter() - start

    print("Question:", question)
    print("Answer:", answer)
    print("Expected:", expected)
    print(f"Elapsed: {elapsed:.2f}s")
    print_trace(rlm.last_trace or trace)

    if expected.lower() in answer.lower():
        print("Result: OK (matched expected phrase).")
    else:
        print("Result: MISMATCH (did not match expected phrase).")


if __name__ == "__main__":
    main()
