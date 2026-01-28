import re
import statistics
from typing import List, Dict, Any, Optional, Tuple

CITE_PATTERN = re.compile(
    r"CITE\(doc_id=(?P<doc_id>[^,]+),\s*chunk_id=(?P<chunk_id>[^,]+),\s*start=(?P<start>\d+),\s*end=(?P<end>\d+)\)"
)


def _normalize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def exact_match(pred: str, gold: str) -> float:
    return 1.0 if (pred or "").strip().lower() == (gold or "").strip().lower() else 0.0


def f1_score(pred: str, gold: str) -> float:
    pred_tokens = _normalize(pred)
    gold_tokens = _normalize(gold)
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = {}
    for tok in pred_tokens:
        common[tok] = common.get(tok, 0) + 1
    overlap = 0
    for tok in gold_tokens:
        if common.get(tok, 0) > 0:
            overlap += 1
            common[tok] -= 1
    precision = overlap / max(1, len(pred_tokens))
    recall = overlap / max(1, len(gold_tokens))
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def recall_at_k(hits: List[Dict[str, Any]], gold_doc_ids: List[str], k: int) -> float:
    if not gold_doc_ids:
        return 0.0
    hit_ids = {str(h.get("doc_id")) for h in hits[:k]}
    gold = {str(x) for x in gold_doc_ids}
    return 1.0 if gold.intersection(hit_ids) else 0.0


def parse_citations(text: str) -> List[Dict[str, Any]]:
    citations = []
    for match in CITE_PATTERN.finditer(text or ""):
        citations.append(
            {
                "doc_id": match.group("doc_id"),
                "chunk_id": match.group("chunk_id"),
                "start": int(match.group("start")),
                "end": int(match.group("end")),
            }
        )
    return citations


def citation_precision(answer: str, chunk_lookup: Dict[str, Dict[str, Any]]) -> Optional[float]:
    cites = parse_citations(answer)
    if not cites:
        return None
    valid = 0
    for cite in cites:
        chunk = chunk_lookup.get(str(cite["chunk_id"]))
        if not chunk:
            continue
        text = chunk.get("text", "")
        start = max(0, min(len(text), cite["start"]))
        end = max(0, min(len(text), cite["end"]))
        if start < end and text[start:end].strip():
            valid += 1
    return valid / len(cites) if cites else None


def hallucination_proxy(answer: str, chunk_lookup: Dict[str, Dict[str, Any]]) -> bool:
    cites = parse_citations(answer)
    if not cites:
        return True
    for cite in cites:
        if str(cite["chunk_id"]) in chunk_lookup:
            return False
    return True


def bootstrap_ci(values: List[float], n: int = 1000, alpha: float = 0.05) -> Tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    import random

    random.seed(42)
    samples = []
    for _ in range(n):
        sample = [random.choice(values) for _ in range(len(values))]
        samples.append(statistics.mean(sample))
    samples.sort()
    lower = samples[int((alpha / 2) * n)]
    upper = samples[int((1 - alpha / 2) * n) - 1]
    return (lower, upper)
