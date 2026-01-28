import csv
import os
import time

from rlm_demo import RLM, RLMOptions
from rlm_demo.trace import RLMTrace

import examples.rlm_selector_sweep as sweep


TASKS = [
    {
        "id": "s1",
        "question": "What internal nickname was used for the stabilization routine?",
        "expected_phrases": ["silver hinge"],
        "grep_pattern": "stabilization routine",
        "answer_regex": r"nicknamed it the \"([^\"]+)\"",
        "anchor": "nicknamed it",
        "extract": "phrase_after_anchor",
    },
    {
        "id": "s2",
        "question": "Which internal label was used for the discrepancy-sealing audit?",
        "expected_phrases": ["blue string"],
        "grep_pattern": "internal label",
        "answer_regex": r"internal label was \"([^\"]+)\"",
        "anchor": "internal label was",
        "extract": "phrase_after_anchor",
    },
    {
        "id": "s3",
        "question": "Which record system squared off inconsistent lots in Helios?",
        "expected_phrases": ["sunset ledger"],
        "grep_pattern": "Helios run",
        "answer_regex": r"relied on the ([^\\.]+)",
        "anchor": "relied on the",
        "extract": "phrase_after_anchor",
    },
    {
        "id": "s4",
        "question": "What nickname was used for the privacy sampler?",
        "expected_phrases": ["moth filter"],
        "grep_pattern": "privacy sampler",
        "answer_regex": r"code-named \"([^\"]+)\"",
        "anchor": "code-named",
        "extract": "phrase_after_anchor",
    },
    {
        "id": "s5",
        "question": "What informal name appears in notes for the sunset ledger?",
        "expected_phrases": ["night log"],
        "grep_pattern": "sunset ledger",
        "answer_regex": r"referred to as the \"([^\"]+)\"",
        "anchor": "referred to as",
        "extract": "phrase_after_anchor",
    },
]


def load_semantic_text():
    base = os.path.dirname(__file__)
    path = os.path.join(base, "data", "book_semantic.txt")
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def main() -> None:
    text = load_semantic_text()
    selectors = [s.strip() for s in os.environ.get("SEMANTIC_SELECTORS", "bm25,embed").split(",") if s.strip()]
    depths = [int(d.strip()) for d in os.environ.get("SEMANTIC_DEPTHS", "1,2").split(",") if d.strip()]
    policy = os.environ.get("SEMANTIC_POLICY", "single")
    output_path = os.environ.get("SEMANTIC_OUTPUT", "semantic_embed_probe.csv")
    resume = sweep._env_bool("SEMANTIC_RESUME", False)
    existing = set()

    if resume and os.path.exists(output_path):
        with open(output_path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                existing.add(
                    (
                        row.get("task_id"),
                        row.get("selector"),
                        row.get("depth"),
                        row.get("delegation"),
                    )
                )
        handle = open(output_path, "a", newline="", encoding="utf-8")
        writer = csv.writer(handle)
    else:
        handle = open(output_path, "w", newline="", encoding="utf-8")
        writer = csv.writer(handle)
        writer = csv.writer(handle)
        writer.writerow(
            [
                "task_id",
                "selector",
                "policy",
                "depth",
                "delegation",
                "correct",
                "latency_s",
                "subcalls",
                "model_input_chars",
                "surfaced_chars",
            ]
        )
    try:
        for selector in selectors:
            for depth in depths:
                delegation = "rlm" if depth > 1 else "none"
                params = sweep.build_params(delegation)
                for task in TASKS:
                    key = (task["id"], selector, str(depth), delegation)
                    if existing and key in existing:
                        print(task["id"], selector, f"d{depth}", "skipped (resume)")
                        continue
                    trace = RLMTrace()
                    options = RLMOptions(
                        require_repl=True,
                        retry_on_invalid=True,
                        trace=trace,
                        protocol="json",
                        repl_timeout_s=sweep._env_float("RLM_REPL_TIMEOUT_S", 10.0),
                        repl_memory_mb=sweep._env_int("RLM_REPL_MEMORY_MB", 256),
                        repl_cpu_seconds=sweep._env_int("RLM_REPL_CPU_SECONDS", 8),
                        max_total_repl_output_chars=sweep._env_int(
                            "RLM_MAX_TOTAL_REPL_OUTPUT_CHARS", None
                        ),
                        max_total_surfaced_chars=sweep._env_int(
                            "RLM_MAX_TOTAL_SURFACED_CHARS", None
                        ),
                        max_total_subcall_input_chars=sweep._env_int(
                            "RLM_MAX_TOTAL_SUBCALL_INPUT_CHARS", None
                        ),
                        min_rlm_queries=1 if delegation == "rlm" else 0,
                    )
                    root = sweep.build_live_client(os.environ.get("SWEEP_ROOT_MODEL", "gpt-5.2"))
                    rlm = RLM(root_llm=root, sub_llm=root, options=options)
                    query = sweep.build_strict_json_query(task, selector, policy, depth, params)
                    start = time.perf_counter()
                    answer = rlm.answer(query, text)
                    latency = time.perf_counter() - start
                    correct = sweep.match_expected(answer, task["expected_phrases"])
                    writer.writerow(
                        [
                            task["id"],
                            selector,
                            policy,
                            depth,
                            delegation,
                            int(correct),
                            f"{latency:.2f}",
                            len(trace.subcalls),
                            trace.model_input_chars,
                            trace.surfaced_chars,
                        ]
                    )
                    print(
                        task["id"],
                        selector,
                        f"d{depth}",
                        "correct",
                        correct,
                        "lat",
                        f"{latency:.2f}s",
                    )
    finally:
        handle.close()
    print("wrote", output_path)


if __name__ == "__main__":
    main()
