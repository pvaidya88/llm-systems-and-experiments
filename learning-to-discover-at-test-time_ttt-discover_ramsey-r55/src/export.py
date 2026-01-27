"""Export utilities for graphs and logs."""

from __future__ import annotations

import json
import os
from typing import Iterable, List, Tuple


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_graph(path: str, adj: List[int], n: int) -> None:
    """Write an edge list (0-based)."""
    lines = [f"# n={n}"]
    for i in range(n):
        row = adj[i]
        for j in range(i + 1, n):
            if row & (1 << j):
                lines.append(f"{i} {j}")
    with open(path, "w", encoding="ascii") as f:
        f.write("\n".join(lines) + "\n")


def read_graph(path: str) -> Tuple[int, List[Tuple[int, int]]]:
    n = None
    edges: List[Tuple[int, int]] = []
    with open(path, "r", encoding="ascii") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("#"):
                if line.startswith("# n="):
                    n = int(line.split("=")[1])
                continue
            parts = line.split()
            if len(parts) != 2:
                continue
            i, j = int(parts[0]), int(parts[1])
            edges.append((i, j))
    if n is None:
        raise ValueError("Missing '# n=' header in graph file")
    return n, edges


def write_json(path: str, data: dict) -> None:
    with open(path, "w", encoding="ascii") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def append_log(path: str, line: str) -> None:
    with open(path, "a", encoding="ascii") as f:
        f.write(line.rstrip() + "\n")
