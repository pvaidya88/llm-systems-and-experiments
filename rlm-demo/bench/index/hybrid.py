from typing import List, Dict, Any, Tuple


def _normalize_scores(hits: List[Dict[str, Any]], key: str = "score") -> Dict[str, float]:
    if not hits:
        return {}
    scores = [float(hit.get(key, 0.0)) for hit in hits]
    min_s = min(scores)
    max_s = max(scores)
    norm = {}
    for hit in hits:
        score = float(hit.get(key, 0.0))
        if max_s == min_s:
            norm_val = 1.0
        else:
            norm_val = (score - min_s) / (max_s - min_s)
        norm[str(hit.get("chunk_id"))] = norm_val
    return norm


def hybrid_search(
    bm25_hits: List[Dict[str, Any]],
    vector_hits: List[Dict[str, Any]],
    k_retrieval: int,
    weights: Tuple[float, float] = (0.5, 0.5),
) -> List[Dict[str, Any]]:
    w_bm25, w_vec = weights
    bm25_norm = _normalize_scores(bm25_hits)
    vec_norm = _normalize_scores(vector_hits)

    combined: Dict[str, Dict[str, Any]] = {}
    for hit in bm25_hits + vector_hits:
        cid = str(hit.get("chunk_id"))
        if cid not in combined:
            combined[cid] = dict(hit)
        combined[cid].setdefault("score", 0.0)

    for cid, hit in combined.items():
        score = w_bm25 * bm25_norm.get(cid, 0.0) + w_vec * vec_norm.get(cid, 0.0)
        hit["score"] = score

    ranked = sorted(combined.values(), key=lambda h: h.get("score", 0.0), reverse=True)
    return ranked[: int(k_retrieval)]
