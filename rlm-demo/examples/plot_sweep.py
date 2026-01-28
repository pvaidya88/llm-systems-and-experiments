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


def _svg_escape(text):
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\"", "&quot;")
        .replace("'", "&apos;")
    )


def _scale(value, vmin, vmax, out_min, out_max):
    if vmax == vmin:
        return (out_min + out_max) / 2
    return out_min + (value - vmin) * (out_max - out_min) / (vmax - vmin)


def plot_scatter_svg(rows, pareto_set, out_dir):
    metrics = [
        ("p50_latency_s", "P50 Latency (s)", "accuracy_vs_latency.svg"),
        ("p50_model_input_chars", "P50 Model Input Chars", "accuracy_vs_model_input.svg"),
        ("p50_surfaced_chars", "P50 Surfaced Chars", "accuracy_vs_surfaced.svg"),
    ]
    width = 960
    height = 640
    pad = 70
    plot_w = width - pad * 2
    plot_h = height - pad * 2

    for metric, xlabel, filename in metrics:
        xs = [row[metric] for row in rows]
        ys = [row["accuracy"] for row in rows]
        x_min = min(xs) if xs else 0.0
        x_max = max(xs) if xs else 1.0
        y_min = min(ys) if ys else 0.0
        y_max = max(ys) if ys else 1.0
        if x_min == x_max:
            x_min -= 1
            x_max += 1
        if y_min == y_max:
            y_min = max(0.0, y_min - 0.1)
            y_max = min(1.0, y_max + 0.1)

        parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
            "<rect width='100%' height='100%' fill='white'/>",
        ]
        # Axes
        parts.append(
            f"<line x1='{pad}' y1='{height - pad}' x2='{width - pad}' y2='{height - pad}' stroke='#333'/>"
        )
        parts.append(
            f"<line x1='{pad}' y1='{pad}' x2='{pad}' y2='{height - pad}' stroke='#333'/>"
        )
        # Ticks
        for i in range(6):
            frac = i / 5
            x = pad + plot_w * frac
            y = height - pad
            tick_val = x_min + (x_max - x_min) * frac
            parts.append(f"<line x1='{x}' y1='{y}' x2='{x}' y2='{y+6}' stroke='#333'/>")
            parts.append(
                f"<text x='{x}' y='{y+24}' font-size='11' text-anchor='middle' fill='#444'>{tick_val:.2f}</text>"
            )
        for i in range(6):
            frac = i / 5
            y = height - pad - plot_h * frac
            x = pad
            tick_val = y_min + (y_max - y_min) * frac
            parts.append(f"<line x1='{x-6}' y1='{y}' x2='{x}' y2='{y}' stroke='#333'/>")
            parts.append(
                f"<text x='{x-10}' y='{y+4}' font-size='11' text-anchor='end' fill='#444'>{tick_val:.2f}</text>"
            )

        # Labels
        parts.append(
            f"<text x='{width/2}' y='{height-20}' font-size='13' text-anchor='middle' fill='#111'>{_svg_escape(xlabel)}</text>"
        )
        parts.append(
            f"<text x='20' y='{height/2}' font-size='13' text-anchor='middle' fill='#111' transform='rotate(-90 20,{height/2})'>Accuracy</text>"
        )
        parts.append(
            f"<text x='{width/2}' y='28' font-size='14' text-anchor='middle' fill='#111'>Accuracy vs {_svg_escape(xlabel)}</text>"
        )

        # Points
        for row in rows:
            x_val = row[metric]
            y_val = row["accuracy"]
            x = _scale(x_val, x_min, x_max, pad, width - pad)
            y = _scale(y_val, y_min, y_max, height - pad, pad)
            key = label(row)
            color = "#E53935" if key in pareto_set else "#4C78A8"
            radius = 4 if key in pareto_set else 3
            parts.append(f"<circle cx='{x:.2f}' cy='{y:.2f}' r='{radius}' fill='{color}' opacity='0.8'/>")
            if key in pareto_set:
                parts.append(
                    f"<text x='{x+6:.2f}' y='{y-6:.2f}' font-size='10' fill='#222'>{_svg_escape(key)}</text>"
                )

        parts.append("</svg>")
        out_path = os.path.join(out_dir, filename)
        with open(out_path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(parts))


def plot_scatter(rows, pareto_set, out_dir):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not available; using SVG fallback.")
        plot_scatter_svg(rows, pareto_set, out_dir)
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
