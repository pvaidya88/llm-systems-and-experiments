"""Adaptive entropic weights with KL(q_beta || uniform) = gamma."""

from __future__ import annotations

import math
from typing import Iterable, Tuple

import numpy as np


def _softmax_beta(rewards: np.ndarray, beta: float) -> np.ndarray:
    if beta <= 0.0:
        return np.ones_like(rewards) / rewards.size
    z = beta * (rewards - rewards.max())
    exp_z = np.exp(z - z.max())
    return exp_z / exp_z.sum()


def _kl_to_uniform(q: np.ndarray) -> float:
    m = q.size
    return float(np.sum(q * (np.log(q + 1e-12) + math.log(m))))


def find_beta(rewards: Iterable[float], gamma: float, max_iter: int = 50) -> Tuple[float, np.ndarray]:
    rewards = np.asarray(list(rewards), dtype=float)
    m = rewards.size
    if m == 1:
        return 0.0, np.array([1.0], dtype=float)
    if gamma <= 0.0:
        return 0.0, np.ones(m, dtype=float) / m
    if np.allclose(rewards, rewards[0]):
        return 0.0, np.ones(m, dtype=float) / m

    low = 0.0
    high = 1.0
    q_high = _softmax_beta(rewards, high)
    while _kl_to_uniform(q_high) < gamma and high < 256.0:
        high *= 2.0
        q_high = _softmax_beta(rewards, high)

    if _kl_to_uniform(q_high) < gamma:
        return high, q_high

    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        q_mid = _softmax_beta(rewards, mid)
        kl_mid = _kl_to_uniform(q_mid)
        if kl_mid < gamma:
            low = mid
        else:
            high = mid
    q_final = _softmax_beta(rewards, high)
    return high, q_final


def entropic_weights(rewards: Iterable[float], gamma: float) -> Tuple[float, np.ndarray]:
    beta, q = find_beta(rewards, gamma)
    return beta, q
