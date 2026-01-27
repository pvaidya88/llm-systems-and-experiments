"""Reward shaping for Ramsey R(5,5) search."""

from __future__ import annotations

from typing import Dict, List, Tuple

from verifier import complement_adj, count_k5_witnesses, find_k5


def compute_reward(
    adj: List[int],
    n: int,
    witness_cap: int,
    shaped_scale: float = 0.05,
    max_checks: int | None = 200000,
) -> Tuple[float, Dict[str, object]]:
    """Return (reward, info). Valid graphs get reward 1.0."""
    witness_g = find_k5(adj, n)
    if witness_g is None:
        comp = complement_adj(adj, n)
        witness_c = find_k5(comp, n)
    else:
        comp = None
        witness_c = None

    if witness_g is None and witness_c is None:
        return 1.0, {
            "valid": True,
            "witness_g": None,
            "witness_c": None,
            "count_g": 0,
            "count_c": 0,
            "truncated": False,
        }

    count_g, _, trunc_g, _ = count_k5_witnesses(
        adj, n, witness_cap, max_checks=max_checks
    )
    if comp is None:
        comp = complement_adj(adj, n)
    count_c, _, trunc_c, _ = count_k5_witnesses(
        comp, n, witness_cap, max_checks=max_checks
    )
    penalty = (count_g + count_c) / (2.0 * witness_cap)
    shaped = -shaped_scale * penalty
    return shaped, {
        "valid": False,
        "witness_g": witness_g,
        "witness_c": witness_c,
        "count_g": count_g,
        "count_c": count_c,
        "truncated": trunc_g or trunc_c,
    }
