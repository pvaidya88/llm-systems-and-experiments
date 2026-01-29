import argparse
import json
import os
import time
from typing import Any, Dict, Optional

from .gates import evaluate, DEFAULT_THRESHOLDS


def _load_summary(path: str) -> Dict[str, Any]:
    if os.path.isdir(path):
        path = os.path.join(path, "summary.json")
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_thresholds(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return dict(DEFAULT_THRESHOLDS)
    with open(path, "r", encoding="utf-8") as handle:
        text = handle.read()
    try:
        import yaml

        return yaml.safe_load(text)
    except Exception:
        return json.loads(text)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--thresholds", default=None)
    parser.add_argument("--run-gates", action="store_true")
    args = parser.parse_args()

    baseline = _load_summary(args.baseline)
    candidate = _load_summary(args.candidate)
    thresholds = _load_thresholds(args.thresholds)

    compare_id = time.strftime("%Y%m%d_%H%M%S")
    out_dir = args.out or os.path.join("artifacts", f"compare_{compare_id}")
    _ensure_dir(out_dir)

    merged = {
        "compare_id": compare_id,
        "baseline": baseline,
        "candidate": candidate,
        "thresholds": thresholds,
    }

    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(merged, handle, ensure_ascii=False, indent=2)

    print(f"Wrote {summary_path}")

    if args.run_gates:
        result = evaluate(merged, thresholds)
        gates_path = os.path.join(out_dir, "gates.json")
        with open(gates_path, "w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
        print(f"Wrote {gates_path}")
        print(f"Gate pass: {result.get('passed')}")
        if not result.get("passed"):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
