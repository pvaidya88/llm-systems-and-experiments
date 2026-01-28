import csv
import os
import time

from rlm_demo import OpenAIResponsesClient, RLM, RLMOptions
from rlm_demo.trace import RLMTrace

import examples.rlm_selector_sweep as sweep


def main() -> None:
    selector = "grep"
    policies = [p.strip() for p in os.environ.get("MINI_POLICIES", "single,expand").split(",") if p.strip()]
    tasks = [t for t in sweep.TASKS if t["id"] in ("t4", "t5")]
    mode_map = {
        "depth2_rlm": ("depth2_rlm", 2, "rlm"),
        "depth1_llm": ("depth1_llm", 1, "llm"),
    }
    mode_names = [
        m.strip()
        for m in os.environ.get("MINI_MODES", "depth2_rlm,depth1_llm").split(",")
        if m.strip()
    ]
    modes = [mode_map[name] for name in mode_names if name in mode_map]

    output_path = "llm_subcall_mini.csv"
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "task_id",
                "policy",
                "mode",
                "depth",
                "delegation",
                "correct",
                "latency_s",
                "subcalls",
                "subcall_input_chars",
                "subcall_output_chars",
                "surfaced_chars",
                "model_input_chars",
            ]
        )
        for mode_name, depth, delegation in modes:
            params = sweep.build_params(delegation)
            for policy in policies:
                for task in tasks:
                    trace = RLMTrace()
                    options = RLMOptions(
                        require_repl=True,
                        retry_on_invalid=True,
                        trace=trace,
                        protocol="json",
                        max_steps=3,
                        max_output_chars=1000,
                        max_total_repl_output_chars=4000,
                        max_total_surfaced_chars=5000,
                        max_total_subcall_input_chars=3000,
                        repl_timeout_s=20.0,
                        repl_cpu_seconds=8,
                        min_rlm_queries=1 if delegation == "rlm" else 0,
                        min_sub_calls=1 if delegation == "llm" else 0,
                    )
                    root = sweep.build_live_client(
                        os.environ.get("SWEEP_ROOT_MODEL", "gpt-5.2")
                    )
                    rlm = RLM(root_llm=root, sub_llm=root, options=options)
                    query = sweep.build_strict_json_query(
                        task, selector, policy, depth, params
                    )
                    start = time.perf_counter()
                    answer = rlm.answer(query, sweep.load_book_text())
                    latency = time.perf_counter() - start
                    correct = sweep.match_expected(answer, task["expected_phrases"])
                    writer.writerow(
                        [
                            task["id"],
                            policy,
                            mode_name,
                            depth,
                            delegation,
                            int(correct),
                            f"{latency:.2f}",
                            len(trace.subcalls),
                            trace.subcall_input_chars,
                            trace.subcall_output_chars,
                            trace.surfaced_chars,
                            trace.model_input_chars,
                        ]
                    )
                    print(
                        task["id"],
                        policy,
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
