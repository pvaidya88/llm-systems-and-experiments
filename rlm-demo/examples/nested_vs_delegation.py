import csv
import os
import time

from rlm_demo import RLM, RLMOptions
from rlm_demo.trace import RLMTrace

import examples.rlm_selector_sweep as sweep


def get_tasks():
    base_tasks = {t["id"]: t for t in sweep.TASKS}
    tasks = [base_tasks["t4"], base_tasks["t5"]]
    tasks.append(
        {
            "id": "t9",
            "question": "What was the internal nickname for the emergency rollback?",
            "expected_phrases": ["amber latch"],
            "grep_pattern": "emergency rollback",
            "answer_regex": r"internally dubbed the \"([^\"]+)\"",
            "anchor": "internally dubbed",
            "extract": "phrase_after_anchor",
        }
    )
    return tasks


def main() -> None:
    selector = os.environ.get("HEADTOHEAD_SELECTOR", "grep")
    policy = os.environ.get("HEADTOHEAD_POLICY", "single")
    output_path = os.environ.get("HEADTOHEAD_OUTPUT", "nested_vs_delegation.csv")

    tasks = get_tasks()
    corpus = sweep.load_book_text()
    modes = [
        ("depth2_rlm", 2, "rlm"),
        ("depth1_llm", 1, "llm"),
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "task_id",
                "mode",
                "selector",
                "policy",
                "depth",
                "delegation",
                "correct",
                "latency_s",
                "subcalls",
                "subcall_input_chars",
                "subcall_output_chars",
                "model_input_chars",
                "surfaced_chars",
            ]
        )
        for mode_name, depth, delegation in modes:
            params = sweep.build_params(delegation)
            for task in tasks:
                trace = RLMTrace()
                options = RLMOptions(
                    require_repl=True,
                    retry_on_invalid=True,
                    trace=trace,
                    protocol="json",
                    max_steps=4,
                    max_output_chars=1200,
                    max_total_repl_output_chars=5000,
                    max_total_surfaced_chars=6000,
                    max_total_subcall_input_chars=5000,
                    repl_timeout_s=sweep._env_float("RLM_REPL_TIMEOUT_S", 15.0),
                    repl_memory_mb=sweep._env_int("RLM_REPL_MEMORY_MB", 256),
                    repl_cpu_seconds=sweep._env_int("RLM_REPL_CPU_SECONDS", 10),
                    min_rlm_queries=1 if delegation == "rlm" else 0,
                    min_sub_calls=2 if delegation == "llm" else 0,
                )
                root = sweep.build_live_client(os.environ.get("SWEEP_ROOT_MODEL", "gpt-5.2"))
                rlm = RLM(root_llm=root, sub_llm=root, options=options)
                query = sweep.build_strict_json_query(task, selector, policy, depth, params)
                start = time.perf_counter()
                answer = rlm.answer(query, corpus)
                latency = time.perf_counter() - start
                correct = sweep.match_expected(answer, task["expected_phrases"])
                writer.writerow(
                    [
                        task["id"],
                        mode_name,
                        selector,
                        policy,
                        depth,
                        delegation,
                        int(correct),
                        f"{latency:.2f}",
                        len(trace.subcalls),
                        trace.subcall_input_chars,
                        trace.subcall_output_chars,
                        trace.model_input_chars,
                        trace.surfaced_chars,
                    ]
                )
                print(
                    task["id"],
                    mode_name,
                    "correct",
                    correct,
                    "lat",
                    f"{latency:.2f}s",
                    "subcalls",
                    len(trace.subcalls),
                )
    print("wrote", output_path)


if __name__ == "__main__":
    main()
