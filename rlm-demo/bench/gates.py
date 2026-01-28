import argparse
import json
from typing import Dict, Any


DEFAULT_THRESHOLDS = {
    "accuracy_diff": 0.02,
    "low_overlap_accuracy_diff": 0.05,
    "recall_diff": 0.03,
    "low_overlap_recall_diff": 0.05,
    "latency_ratio": 2.0,
    "cost_ratio": 1.5,
    "hallucination_diff": 0.0,
}


def load_summary(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def evaluate(summary: Dict[str, Any], thresholds: Dict[str, float]) -> Dict[str, Any]:
    baseline = summary.get("baseline") or {}
    candidate = summary.get("candidate") or {}

    def diff(metric: str) -> float:
        return (candidate.get(metric, 0) - baseline.get(metric, 0))

    results = {}
    results["accuracy"] = diff("accuracy") >= -thresholds["accuracy_diff"]
    results["low_overlap_accuracy"] = diff("low_overlap_accuracy") >= -thresholds["low_overlap_accuracy_diff"]
    results["recall"] = diff("recall_at_10") >= -thresholds["recall_diff"]
    results["low_overlap_recall"] = diff("low_overlap_recall_at_10") >= -thresholds["low_overlap_recall_diff"]

    latency_ratio = (candidate.get("p95_latency", 0) / max(baseline.get("p95_latency", 1e-6), 1e-6))
    cost_ratio = (candidate.get("cost_per_query", 0) / max(baseline.get("cost_per_query", 1e-6), 1e-6))
    results["latency"] = latency_ratio <= thresholds["latency_ratio"]
    results["cost"] = cost_ratio <= thresholds["cost_ratio"]

    results["hallucination"] = diff("hallucination_rate") <= thresholds["hallucination_diff"]
    passed = all(results.values())
    return {
        "passed": passed,
        "checks": results,
        "latency_ratio": latency_ratio,
        "cost_ratio": cost_ratio,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    args = parser.parse_args()

    summary = load_summary(args.run)
    thresholds = summary.get("thresholds") or DEFAULT_THRESHOLDS
    result = evaluate(summary, thresholds)
    print("PASS" if result["passed"] else "FAIL")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
