import re
from typing import Dict, List, Optional, Any


def _tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def jaccard(a: List[str], b: List[str]) -> float:
    if not a or not b:
        return 0.0
    set_a = set(a)
    set_b = set(b)
    return len(set_a & set_b) / max(1, len(set_a | set_b))


def infer_buckets(
    question: str,
    gold_text: str,
    gold_doc_ids: Optional[List[str]] = None,
    df_stats: Optional[Dict[str, int]] = None,
    total_docs: Optional[int] = None,
    low_overlap_threshold: float = 0.15,
) -> List[str]:
    buckets = []
    q_tokens = _tokens(question)
    g_tokens = _tokens(gold_text)
    overlap = jaccard(q_tokens, g_tokens)
    if overlap < low_overlap_threshold:
        buckets.append("low_overlap")
    else:
        buckets.append("high_overlap")

    # acronym/alias bucket heuristic
    if any(tok.isupper() and len(tok) >= 2 for tok in question.split()):
        buckets.append("acronym_alias")

    # needle-in-haystack: very low document frequency tokens
    if df_stats and total_docs:
        rare = 0
        for tok in set(g_tokens):
            if df_stats.get(tok, total_docs) <= max(1, total_docs // 1000):
                rare += 1
        if rare >= 2:
            buckets.append("needle")

    # multi-hop
    if gold_doc_ids and len(set(gold_doc_ids)) >= 2:
        buckets.append("multihop")

    return buckets
