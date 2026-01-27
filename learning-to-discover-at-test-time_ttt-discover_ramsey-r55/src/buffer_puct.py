"""PUCT-style state buffer with max-reward Q(s)."""

from __future__ import annotations

import math
from typing import Dict, List, Tuple


class BufferPUCT:
    def __init__(self, capacity: int = 128, c_puct: float = 1.5) -> None:
        self.capacity = capacity
        self.c_puct = c_puct
        self.records: List[Dict[str, object]] = []
        self.index: Dict[Tuple[int, ...], int] = {}
        self.total_visits = 0

    def __len__(self) -> int:
        return len(self.records)

    def add_state(self, key: Tuple[int, ...], adj: Tuple[int, ...], prior: float = 1.0) -> None:
        if key in self.index:
            idx = self.index[key]
            rec = self.records[idx]
            rec["prior"] = max(float(rec["prior"]), float(prior))
            return
        if len(self.records) >= self.capacity:
            self._evict_one()
        rec = {
            "key": key,
            "adj": adj,
            "N": 0,
            "Q": -1.0,
            "prior": float(prior),
            "last_reward": None,
        }
        self.index[key] = len(self.records)
        self.records.append(rec)

    def update(self, key: Tuple[int, ...], reward: float) -> None:
        idx = self.index.get(key)
        if idx is None:
            return
        rec = self.records[idx]
        rec["N"] = int(rec["N"]) + 1
        rec["Q"] = max(float(rec["Q"]), float(reward))
        rec["last_reward"] = float(reward)
        self.total_visits += 1

    def select(self) -> Dict[str, object]:
        if not self.records:
            raise RuntimeError("Buffer is empty")
        total = max(1, self.total_visits)
        best_idx = 0
        best_score = -1e9
        for idx, rec in enumerate(self.records):
            q_val = float(rec["Q"])
            prior = float(rec["prior"])
            n_visits = int(rec["N"])
            score = q_val + self.c_puct * prior * math.sqrt(total) / (1 + n_visits)
            if score > best_score:
                best_score = score
                best_idx = idx
        return self.records[best_idx]

    def _evict_one(self) -> None:
        worst_idx = 0
        worst_score = 1e9
        for idx, rec in enumerate(self.records):
            score = float(rec["Q"]) + 0.1 * float(rec["prior"])
            if score < worst_score:
                worst_score = score
                worst_idx = idx
        self._remove_index(worst_idx)

    def _remove_index(self, idx: int) -> None:
        rec = self.records[idx]
        key = rec["key"]
        del self.index[key]
        self.records.pop(idx)
        for i in range(idx, len(self.records)):
            self.index[self.records[i]["key"]] = i
