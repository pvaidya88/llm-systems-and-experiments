import json
import os

from rlm_demo import RLM, RLMOptions
from rlm_demo.trace import RLMTrace
from examples import rlm_selector_sweep as sweep


def main():
    task_id = os.environ.get("DEBUG_TASK_ID", "t1")
    selector = os.environ.get("DEBUG_SELECTOR", "grep")
    policy = os.environ.get("DEBUG_POLICY", "iterative")
    depth = int(os.environ.get("DEBUG_DEPTH", "2"))

    task = next(t for t in sweep.TASKS if t["id"] == task_id)
    book_text = sweep.load_book_text()

    query = sweep.build_strict_json_query(task, selector, policy, depth)
    root = sweep.build_live_client(os.environ.get("SWEEP_ROOT_MODEL", "gpt-5.2"))
    sub = sweep.build_live_client(os.environ.get("SWEEP_SUB_MODEL", "gpt-5.2"))

    trace = RLMTrace()
    options = RLMOptions(
        require_repl=True,
        retry_on_invalid=True,
        trace=trace,
        protocol="json",
        repl_timeout_s=float(os.environ.get("RLM_REPL_TIMEOUT_S", "30")),
    )
    if os.environ.get("DEBUG_LOG_REPL") == "1":
        options.log_repl_outputs = True
    rlm = RLM(root_llm=root, sub_llm=sub, options=options)

    answer = rlm.answer(query, book_text)
    print("Answer:", answer)
    print("Expected phrases:", task["expected_phrases"])
    print("Termination:", trace.termination_reason)
    print("Steps:", trace.steps, "Repl blocks:", len(trace.repl_blocks))
    print("Model input chars:", trace.model_input_chars)
    print("RLM queries:", trace.rlm_queries)

    if trace.retrieval:
        first = trace.retrieval[0]
        print("First retrieval op:", first.op, "hits:", len(first.hits))
        if first.hits:
            print("Top snippet:", first.hits[0].snippet)

    print("Assistant messages (depth + first 200 chars):")
    for msg in trace.assistant_messages:
        content = msg.get("content", "")
        print(f"- depth {msg.get('depth')}: {content[:200]!r}")

    dump = {
        "answer": answer,
        "trace": {
            "steps": trace.steps,
            "termination_reason": trace.termination_reason,
            "model_input_chars": trace.model_input_chars,
            "rlm_queries": trace.rlm_queries,
            "assistant_messages": trace.assistant_messages,
            "retrieval": [
                {
                    "op": evt.op,
                    "query": evt.query,
                    "params": evt.params,
                    "hits": [
                        {"ref_id": h.ref_id, "score": h.score, "snippet": h.snippet}
                        for h in evt.hits
                    ],
                }
                for evt in trace.retrieval
            ],
        },
    }
    out_path = os.environ.get("DEBUG_TRACE_OUTPUT", "debug_depth2_trace.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(dump, handle, indent=2)
    print("Wrote trace to", out_path)


if __name__ == "__main__":
    main()
