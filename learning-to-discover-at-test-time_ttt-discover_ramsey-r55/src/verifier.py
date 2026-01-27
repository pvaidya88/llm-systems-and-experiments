"""Exact K5 verifier using bitset intersections."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

_MASK_CACHE: Dict[int, List[int]] = {}


def _higher_masks(n: int) -> List[int]:
    cached = _MASK_CACHE.get(n)
    if cached is not None:
        return cached
    masks = [0] * n
    full = (1 << n) - 1
    for i in range(n):
        masks[i] = full & (~((1 << (i + 1)) - 1))
    _MASK_CACHE[n] = masks
    return masks


def _lsb_index(bit: int) -> int:
    return (bit.bit_length() - 1)


def complement_adj(adj: List[int], n: int) -> List[int]:
    full = (1 << n) - 1
    comp = []
    for i in range(n):
        comp.append((~adj[i]) & (full ^ (1 << i)))
    return comp


def find_k5(adj: List[int], n: int) -> Optional[List[int]]:
    """Return a witness 5-clique if it exists, else None."""
    masks = _higher_masks(n)
    for a in range(n):
        nbrs_a = adj[a] & masks[a]
        while nbrs_a:
            b_bit = nbrs_a & -nbrs_a
            nbrs_a -= b_bit
            b = _lsb_index(b_bit)
            nbrs_ab = adj[b] & adj[a] & masks[b]
            while nbrs_ab:
                c_bit = nbrs_ab & -nbrs_ab
                nbrs_ab -= c_bit
                c = _lsb_index(c_bit)
                nbrs_abc = adj[c] & nbrs_ab & masks[c]
                while nbrs_abc:
                    d_bit = nbrs_abc & -nbrs_abc
                    nbrs_abc -= d_bit
                    d = _lsb_index(d_bit)
                    nbrs_abcd = adj[d] & nbrs_abc & masks[d]
                    if nbrs_abcd:
                        e_bit = nbrs_abcd & -nbrs_abcd
                        e = _lsb_index(e_bit)
                        return [a, b, c, d, e]
    return None


def count_k5_witnesses(
    adj: List[int],
    n: int,
    cap: int,
    max_checks: int | None = None,
) -> Tuple[int, List[List[int]], bool, int]:
    """Count up to cap 5-cliques. Returns (count, witnesses, truncated, checks)."""
    masks = _higher_masks(n)
    witnesses: List[List[int]] = []
    count = 0
    checks = 0
    for a in range(n):
        nbrs_a = adj[a] & masks[a]
        while nbrs_a:
            b_bit = nbrs_a & -nbrs_a
            nbrs_a -= b_bit
            b = _lsb_index(b_bit)
            nbrs_ab = adj[b] & adj[a] & masks[b]
            while nbrs_ab:
                c_bit = nbrs_ab & -nbrs_ab
                nbrs_ab -= c_bit
                c = _lsb_index(c_bit)
                nbrs_abc = adj[c] & nbrs_ab & masks[c]
                while nbrs_abc:
                    d_bit = nbrs_abc & -nbrs_abc
                    nbrs_abc -= d_bit
                    d = _lsb_index(d_bit)
                    checks += 1
                    if max_checks is not None and checks > max_checks:
                        return count, witnesses, True, checks
                    nbrs_abcd = adj[d] & nbrs_abc & masks[d]
                    while nbrs_abcd:
                        e_bit = nbrs_abcd & -nbrs_abcd
                        nbrs_abcd -= e_bit
                        e = _lsb_index(e_bit)
                        witnesses.append([a, b, c, d, e])
                        count += 1
                        if count >= cap:
                            return count, witnesses, False, checks
    return count, witnesses, False, checks


def verify_graph(adj: List[int], n: int) -> Dict[str, object]:
    """Exact verification of no K5 in G and complement(G)."""
    witness_g = find_k5(adj, n)
    if witness_g is not None:
        return {
            "valid": False,
            "witness_g": witness_g,
            "witness_c": None,
        }
    comp = complement_adj(adj, n)
    witness_c = find_k5(comp, n)
    if witness_c is not None:
        return {
            "valid": False,
            "witness_g": None,
            "witness_c": witness_c,
        }
    return {
        "valid": True,
        "witness_g": None,
        "witness_c": None,
    }
