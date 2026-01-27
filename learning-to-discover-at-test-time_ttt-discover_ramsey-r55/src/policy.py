"""Learnable edge-flip policy."""

from __future__ import annotations

from typing import Iterable, List

import numpy as np

from graph import edges_list


class EdgeFlipPolicy:
    def __init__(self, n: int, rng, init_scale: float = 0.01) -> None:
        self.n = n
        self.edges = edges_list(n)
        self.m = len(self.edges)
        self.rng = rng
        self.logits = rng.normal(0.0, init_scale, size=self.m)

    def sample_action(self, k: int) -> List[int]:
        k = int(max(1, min(k, self.m)))
        probs = self._softmax(self.logits)
        idx = self.rng.choice(self.m, size=k, replace=False, p=probs)
        return idx.tolist()

    def update(self, actions: Iterable[List[int]], weights: np.ndarray, lr: float = 0.1, l2: float = 1e-3) -> None:
        actions = list(actions)
        if not actions:
            return
        baseline = 1.0 / len(actions)
        deltas = weights - baseline
        for act, delta in zip(actions, deltas):
            for idx in act:
                self.logits[idx] += lr * float(delta)
        self.logits *= (1.0 - lr * l2)
        np.clip(self.logits, -10.0, 10.0, out=self.logits)

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        z = x - x.max()
        exp_z = np.exp(z)
        return exp_z / exp_z.sum()
