"""Graph utilities for Ramsey R(5,5) search."""

from __future__ import annotations

from typing import Iterable, List, Tuple


def edges_list(n: int) -> List[Tuple[int, int]]:
    """Deterministic list of undirected edges (i < j)."""
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            edges.append((i, j))
    return edges


class Graph:
    def __init__(self, n: int, adj: Iterable[int] | None = None) -> None:
        self.n = n
        self.full_mask = (1 << n) - 1
        if adj is None:
            self.adj = [0] * n
        else:
            self.adj = list(adj)

    def clone(self) -> "Graph":
        return Graph(self.n, self.adj)

    def to_key(self) -> Tuple[int, ...]:
        return tuple(self.adj)

    def toggle_edge(self, i: int, j: int) -> None:
        if i == j:
            return
        mi = 1 << j
        mj = 1 << i
        if self.adj[i] & mi:
            self.adj[i] &= ~mi
            self.adj[j] &= ~mj
        else:
            self.adj[i] |= mi
            self.adj[j] |= mj

    def apply_flips(self, edge_indices: Iterable[int], edges: List[Tuple[int, int]]) -> None:
        for idx in edge_indices:
            i, j = edges[idx]
            self.toggle_edge(i, j)

    @staticmethod
    def random(n: int, rng, p: float = 0.5) -> "Graph":
        adj = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < p:
                    adj[i] |= 1 << j
                    adj[j] |= 1 << i
        return Graph(n, adj)

    @staticmethod
    def from_edge_list(n: int, edges: Iterable[Tuple[int, int]]) -> "Graph":
        g = Graph(n)
        for i, j in edges:
            g.toggle_edge(i, j)
        return g
