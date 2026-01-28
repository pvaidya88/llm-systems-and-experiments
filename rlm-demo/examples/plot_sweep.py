import csv
import os
import statistics
import sys
from collections import defaultdict


def _median(values):
    if not values:
        return float("nan")
    return statistics.median(values)


def _to_float(value):
    try:
        return float(value)
    except Exception:
        return None


def _to_int(value):
    try:
        return int(float(value))
    except Exception:
        return None


def load_rows(path):
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            yield row


def group_key(row):
    selector = row.get("selector", "")
    policy = row.get("policy", "")
    depth = row.get("depth", "")
    delegation = row.get("delegation") or row.get("delegate") or "none"
    return (selector, policy, depth, delegation)


def aggregate(rows):
    grouped = defaultdict(lambda: {"correct": [], "lat": [], "model": [], "surfaced": []})
    for row in rows:
        key = group_key(row)
        grouped[key]["correct"].append(_to_int(row.get("correct", 0)) or 0)
        lat = _to_float(row.get("latency_s"))
        if lat is not None:
            grouped[key]["lat"].append(lat)
        model_input = _to_float(row.get("model_input_chars"))
        if model_input is not None:
            grouped[key]["model"].append(model_input)
        surfaced = _to_float(row.get("surfaced_chars"))
        if surfaced is not None:
            grouped[key]["surfaced"].append(surfaced)

    aggregated = []
    for key, vals in grouped.items():
        selector, policy, depth, delegation = key
        total = len(vals["correct"])
        accuracy = sum(vals["correct"]) / total if total else 0.0
        aggregated.append(
            {
                "selector": selector,
                "policy": policy,
                "depth": depth,
                "delegation": delegation,
                "n": total,
                "accuracy": accuracy,
                "p50_latency_s": _median(vals["lat"]),
                "p50_model_input_chars": _median(vals["model"]),
                "p50_surfaced_chars": _median(vals["surfaced"]),
            }
        )
    return aggregated


def dominates(a, b):
    if a["accuracy"] < b["accuracy"]:
        return False
    if a["p50_latency_s"] > b["p50_latency_s"]:
        return False
    if a["p50_model_input_chars"] > b["p50_model_input_chars"]:
        return False
    if a["p50_surfaced_chars"] > b["p50_surfaced_chars"]:
        return False
    return (
        a["accuracy"] > b["accuracy"]
        or a["p50_latency_s"] < b["p50_latency_s"]
        or a["p50_model_input_chars"] < b["p50_model_input_chars"]
        or a["p50_surfaced_chars"] < b["p50_surfaced_chars"]
    )


def pareto_front(rows):
    pareto = []
    for row in rows:
        dominated = False
        for other in rows:
            if other is row:
                continue
            if dominates(other, row):
                dominated = True
                break
        if not dominated:
            pareto.append(row)
    return pareto


def write_csv(rows, path, fields):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def label(row):
    return f"{row['selector']}/{row['policy']}/d{row['depth']}/{row['delegation']}"


def plot_scatter(rows, pareto_set, out_dir):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not available; skipping plots.")
        return

    metrics = [
        ("p50_latency_s", "P50 Latency (s)", "accuracy_vs_latency.png"),
        ("p50_model_input_chars", "P50 Model Input Chars", "accuracy_vs_model_input.png"),
        ("p50_surfaced_chars", "P50 Surfaced Chars", "accuracy_vs_surfaced.png"),
    ]

    for metric, xlabel, filename in metrics:
        plt.figure(figsize=(9, 6))
        xs = [row[metric] for row in rows]
        ys = [row["accuracy"] for row in rows]
        plt.scatter(xs, ys, alpha=0.6)
        for row in rows:
            key = label(row)
            if key in pareto_set:
                plt.annotate(
                    key,
                    (row[metric], row["accuracy"]),
                    textcoords="offset points",
                    xytext=(4, 4),
                    fontsize=8,
                )
        plt.xlabel(xlabel)
        plt.ylabel("Accuracy")
        plt.title(f"Accuracy vs {xlabel}")
        plt.grid(True, alpha=0.3)
        out_path = os.path.join(out_dir, filename)
        plt.tight_layout()
        plt.savefig(out_path, dpi=160)
        plt.close()


def main():
    path = os.environ.get("SWEEP_INPUT", "sweep_live_v2.csv")
    if not os.path.isabs(path):
        path = os.path.join(os.getcwd(), path)
    if not os.path.exists(path):
        print(f"CSV not found: {path}")
        sys.exit(1)

    rows = list(load_rows(path))
    aggregated = aggregate(rows)
    pareto = pareto_front(aggregated)
    pareto_labels = {label(row) for row in pareto}

    out_dir = os.environ.get("SWEEP_PLOT_DIR")
    if not out_dir:
        out_dir = os.path.dirname(path) or "."
    os.makedirs(out_dir, exist_ok=True)

    fields = [
        "selector",
        "policy",
        "depth",
        "delegation",
        "n",
        "accuracy",
        "p50_latency_s",
        "p50_model_input_chars",
        "p50_surfaced_chars",
    ]
    write_csv(aggregated, os.path.join(out_dir, "sweep_agg.csv"), fields)
    write_csv(pareto, os.path.join(out_dir, "sweep_pareto.csv"), fields)

    plot_scatter(aggregated, pareto_labels, out_dir)

    print(f"Wrote {os.path.join(out_dir, 'sweep_agg.csv')}")
    print(f"Wrote {os.path.join(out_dir, 'sweep_pareto.csv')}")


if __name__ == "__main__":
    main()
