import copy
import math
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


@dataclass
class Chunk:
    id: int
    doc_id: int
    start: int
    end: int
    text: str


class BM25Index:
    def __init__(self, chunks: List[Chunk]):
        self._chunks = chunks
        self._doc_freqs: Dict[str, int] = {}
        self._term_freqs: List[Dict[str, int]] = []
        self._doc_lens: List[int] = []
        self._avg_len = 1.0
        self._build()

    def _build(self) -> None:
        total_len = 0
        for chunk in self._chunks:
            terms = _tokenize(chunk.text)
            freqs: Dict[str, int] = {}
            for term in terms:
                freqs[term] = freqs.get(term, 0) + 1
            self._term_freqs.append(freqs)
            length = max(1, len(terms))
            self._doc_lens.append(length)
            total_len += length
            for term in freqs.keys():
                self._doc_freqs[term] = self._doc_freqs.get(term, 0) + 1
        if self._chunks:
            self._avg_len = total_len / len(self._chunks)
        else:
            self._avg_len = 1.0

    def search(self, query: str, k: int = 5, k1: float = 1.5, b: float = 0.75) -> List[Dict[str, Any]]:
        query_terms = _tokenize(query)
        scores: Dict[int, float] = {}
        n_docs = max(1, len(self._chunks))
        for term in query_terms:
            df = self._doc_freqs.get(term, 0)
            if df == 0:
                continue
            idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
            for doc_id, freqs in enumerate(self._term_freqs):
                tf = freqs.get(term, 0)
                if tf == 0:
                    continue
                doc_len = self._doc_lens[doc_id]
                denom = tf + k1 * (1 - b + b * (doc_len / self._avg_len))
                score = idf * (tf * (k1 + 1)) / denom
                scores[doc_id] = scores.get(doc_id, 0.0) + score
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        hits = []
        for idx, score in ranked[: max(1, k)]:
            chunk = self._chunks[idx]
            hits.append(
                {
                    "chunk_id": chunk.id,
                    "doc_id": chunk.doc_id,
                    "start": chunk.start,
                    "end": chunk.end,
                    "text": chunk.text,
                    "score": score,
                }
            )
        return hits


class SelectionContext:
    def __init__(
        self,
        context: Any,
        trace_hook: Optional[Callable[[Dict[str, Any]], None]] = None,
        max_snippet_chars: int = 200,
        cache_enabled: bool = True,
    ) -> None:
        self._raw = context
        self._docs = _normalize_docs(context)
        self._trace_hook = trace_hook
        self._max_snippet_chars = max(20, int(max_snippet_chars))
        self._chunks: Optional[List[Chunk]] = None
        self._bm25: Optional[BM25Index] = None
        self._embed_cache: Optional[Dict[str, Any]] = None
        self._cache_enabled = bool(cache_enabled)
        self._cache: Dict[str, Any] = {}

    @property
    def docs(self) -> List[str]:
        return list(self._docs)

    def chunkify(self, chunk_chars: int = 1000, overlap: int = 120) -> List[Dict[str, Any]]:
        chunk_chars = max(50, int(chunk_chars))
        overlap = max(0, int(overlap))
        chunks: List[Chunk] = []
        chunk_id = 0
        for doc_id, text in enumerate(self._docs):
            if not text:
                continue
            start = 0
            while start < len(text):
                end = min(len(text), start + chunk_chars)
                chunk_text = text[start:end]
                chunks.append(
                    Chunk(
                        id=chunk_id,
                        doc_id=doc_id,
                        start=start,
                        end=end,
                        text=chunk_text,
                    )
                )
                chunk_id += 1
                if end == len(text):
                    break
                start = max(0, end - overlap)
        self._chunks = chunks
        self._bm25 = None
        self._embed_cache = None
        return [_chunk_to_dict(chunk) for chunk in chunks]

    def grep(self, pattern: str, *, window: int = 80, max_hits: int = 20) -> List[Dict[str, Any]]:
        window = max(10, int(window))
        max_hits = max(1, int(max_hits))
        cache_key = self._cache_key("grep", pattern, window, max_hits)
        cached = self._cache_get(cache_key)
        if cached is not None:
            self._trace(
                "grep",
                query=pattern,
                params={"window": window, "max_hits": max_hits, "cache_hit": True},
                hits=_hits_for_trace(cached, snippet_key="snippet", max_chars=self._max_snippet_chars),
            )
            return cached
        hits: List[Dict[str, Any]] = []
        regex = re.compile(pattern, re.IGNORECASE)
        for doc_id, text in enumerate(self._docs):
            for match_idx, match in enumerate(regex.finditer(text)):
                if len(hits) >= max_hits:
                    break
                start = max(0, match.start() - window)
                end = min(len(text), match.end() + window)
                snippet = text[start:end]
                hit = {
                    "ref_id": f"doc{doc_id}-m{match_idx}",
                    "doc_id": doc_id,
                    "start": match.start(),
                    "end": match.end(),
                    "snippet": snippet,
                    "match": match.group(0),
                }
                hits.append(hit)
            if len(hits) >= max_hits:
                break
        self._cache_set(cache_key, hits)
        self._trace(
            "grep",
            query=pattern,
            params={"window": window, "max_hits": max_hits, "cache_hit": False},
            hits=_hits_for_trace(hits, snippet_key="snippet", max_chars=self._max_snippet_chars),
        )
        return hits

    def bm25_search(self, query: str, *, k: int = 5, field: str = "text") -> List[Dict[str, Any]]:
        self._ensure_chunks()
        if not self._chunks:
            return []
        if self._bm25 is None:
            self._bm25 = BM25Index(self._chunks)
        cache_key = self._cache_key("bm25", query, k, field)
        cached = self._cache_get(cache_key)
        if cached is not None:
            self._trace(
                "bm25_search",
                query=query,
                params={"k": k, "field": field, "cache_hit": True},
                hits=_hits_for_trace(cached, snippet_key="text", max_chars=self._max_snippet_chars),
            )
            return cached
        hits = self._bm25.search(query, k=k)
        self._cache_set(cache_key, hits)
        self._trace(
            "bm25_search",
            query=query,
            params={"k": k, "field": field, "cache_hit": False},
            hits=_hits_for_trace(hits, snippet_key="text", max_chars=self._max_snippet_chars),
        )
        return hits

    def embed_search(self, query: str, *, k: int = 5, dims: int = 256) -> List[Dict[str, Any]]:
        self._ensure_chunks()
        if not self._chunks:
            return []
        dims = max(64, int(dims))
        cache_key = self._cache_key("embed", query, k, dims)
        cached = self._cache_get(cache_key)
        if cached is not None:
            self._trace(
                "embed_search",
                query=query,
                params={"k": k, "dims": dims, "cache_hit": True},
                hits=_hits_for_trace(cached, snippet_key="text", max_chars=self._max_snippet_chars),
            )
            return cached
        cache = self._embed_cache
        if cache is None or cache.get("dims") != dims:
            vectors = []
            norms = []
            for chunk in self._chunks:
                vec = _hash_vector(_tokenize(chunk.text), dims)
                vectors.append(vec)
                norms.append(_vector_norm(vec))
            cache = {"dims": dims, "vectors": vectors, "norms": norms}
            self._embed_cache = cache
        query_vec = _hash_vector(_tokenize(query), dims)
        query_norm = _vector_norm(query_vec)
        hits = []
        scores = []
        for idx, vec in enumerate(cache["vectors"]):
            score = _dot(query_vec, vec) / max(query_norm * cache["norms"][idx], 1e-6)
            scores.append((idx, score))
        scores.sort(key=lambda item: item[1], reverse=True)
        for idx, score in scores[: max(1, k)]:
            chunk = self._chunks[idx]
            hits.append(
                {
                    "chunk_id": chunk.id,
                    "doc_id": chunk.doc_id,
                    "start": chunk.start,
                    "end": chunk.end,
                    "text": chunk.text,
                    "score": score,
                }
            )
        self._cache_set(cache_key, hits)
        self._trace(
            "embed_search",
            query=query,
            params={"k": k, "dims": dims, "cache_hit": False},
            hits=_hits_for_trace(hits, snippet_key="text", max_chars=self._max_snippet_chars),
        )
        return hits

    def expand(self, hit: Any, radius: int = 200) -> Dict[str, Any]:
        radius = max(0, int(radius))
        cache_key = self._cache_key("expand", str(hit), radius)
        cached = self._cache_get(cache_key)
        if cached is not None:
            self._trace(
                "expand",
                query=None,
                params={"radius": radius, "cache_hit": True},
                hits=_hits_for_trace([cached], snippet_key="snippet", max_chars=self._max_snippet_chars),
            )
            return cached
        doc_id, start, end = _resolve_hit(hit)
        if doc_id is None:
            self._ensure_chunks()
            if isinstance(hit, int) and self._chunks and 0 <= hit < len(self._chunks):
                chunk = self._chunks[hit]
                doc_id, start, end = chunk.doc_id, chunk.start, chunk.end
            elif isinstance(hit, dict) and "chunk_id" in hit and self._chunks:
                chunk_id = hit.get("chunk_id")
                if isinstance(chunk_id, int) and 0 <= chunk_id < len(self._chunks):
                    chunk = self._chunks[chunk_id]
                    doc_id, start, end = chunk.doc_id, chunk.start, chunk.end
        if doc_id is None or doc_id >= len(self._docs):
            return {"text": "", "doc_id": doc_id, "start": start, "end": end}
        text = self._docs[doc_id]
        start = max(0, (start or 0) - radius)
        end = min(len(text), (end or start) + radius)
        snippet = text[start:end]
        event_hit = {
            "ref_id": f"doc{doc_id}-expand",
            "doc_id": doc_id,
            "start": start,
            "end": end,
            "snippet": snippet,
        }
        self._cache_set(cache_key, event_hit)
        self._trace(
            "expand",
            query=None,
            params={"radius": radius, "cache_hit": False},
            hits=_hits_for_trace([event_hit], snippet_key="snippet", max_chars=self._max_snippet_chars),
        )
        return {"text": snippet, "doc_id": doc_id, "start": start, "end": end}

    def heading_index(self) -> List[Dict[str, Any]]:
        cache_key = self._cache_key("heading_index")
        cached = self._cache_get(cache_key)
        if cached is not None:
            self._trace(
                "heading_index",
                query=None,
                params={"cache_hit": True},
                hits=_hits_for_trace(cached, snippet_key="text", max_chars=self._max_snippet_chars),
            )
            return cached
        headings = []
        for doc_id, text in enumerate(self._docs):
            for idx, line in enumerate(text.splitlines()):
                stripped = line.strip()
                if not stripped:
                    continue
                lowered = stripped.lower()
                if stripped.startswith("#") or lowered.startswith("chapter ") or lowered.startswith("section "):
                    headings.append({"doc_id": doc_id, "line": idx + 1, "text": stripped})
                elif stripped.isupper() and len(stripped) >= 4:
                    headings.append({"doc_id": doc_id, "line": idx + 1, "text": stripped})
        self._cache_set(cache_key, headings)
        self._trace(
            "heading_index",
            query=None,
            params={"cache_hit": False},
            hits=_hits_for_trace(headings, snippet_key="text", max_chars=self._max_snippet_chars),
        )
        return headings

    def toc_index(self) -> List[Dict[str, Any]]:
        cache_key = self._cache_key("toc_index")
        cached = self._cache_get(cache_key)
        if cached is not None:
            self._trace(
                "toc_index",
                query=None,
                params={"cache_hit": True},
                hits=_hits_for_trace(cached, snippet_key="text", max_chars=self._max_snippet_chars),
            )
            return cached
        entries = []
        toc_pattern = re.compile(r"^(chapter|section)?\s*\d+[\.\-]?\s+", re.IGNORECASE)
        for doc_id, text in enumerate(self._docs):
            for idx, line in enumerate(text.splitlines()):
                stripped = line.strip()
                if toc_pattern.match(stripped):
                    entries.append({"doc_id": doc_id, "line": idx + 1, "text": stripped})
        self._cache_set(cache_key, entries)
        self._trace(
            "toc_index",
            query=None,
            params={"cache_hit": False},
            hits=_hits_for_trace(entries, snippet_key="text", max_chars=self._max_snippet_chars),
        )
        return entries

    def search_grep(self, pattern: str, *, window: int = 80, max_hits: int = 20) -> List[Dict[str, Any]]:
        return self.grep(pattern, window=window, max_hits=max_hits)

    def search_bm25(self, query: str, *, k: int = 5) -> List[Dict[str, Any]]:
        return self.bm25_search(query, k=k)

    def search_embed(self, query: str, *, k: int = 5, dims: int = 256) -> List[Dict[str, Any]]:
        return self.embed_search(query, k=k, dims=dims)

    def _ensure_chunks(self) -> None:
        if self._chunks is None:
            self.chunkify()

    def _trace(
        self,
        op: str,
        query: Optional[str],
        params: Dict[str, Any],
        hits: List[Dict[str, Any]],
    ) -> None:
        if not self._trace_hook:
            return
        event = {"op": op, "query": query, "params": params, "hits": hits}
        self._trace_hook(event)

    def _cache_key(self, *parts: Any) -> str:
        return "|".join(str(part) for part in parts)

    def _cache_get(self, key: str) -> Optional[Any]:
        if not self._cache_enabled:
            return None
        if key not in self._cache:
            return None
        return copy.deepcopy(self._cache[key])

    def _cache_set(self, key: str, value: Any) -> None:
        if not self._cache_enabled:
            return
        self._cache[key] = copy.deepcopy(value)


def _normalize_docs(context: Any) -> List[str]:
    if isinstance(context, (list, tuple)):
        return [str(item) for item in context]
    return [str(context)]


def _chunk_to_dict(chunk: Chunk) -> Dict[str, Any]:
    return {
        "chunk_id": chunk.id,
        "doc_id": chunk.doc_id,
        "start": chunk.start,
        "end": chunk.end,
        "text": chunk.text,
    }


def _hits_for_trace(
    hits: Iterable[Dict[str, Any]],
    *,
    snippet_key: str,
    max_chars: int,
) -> List[Dict[str, Any]]:
    trimmed = []
    for hit in hits:
        snippet = str(hit.get(snippet_key, ""))
        if len(snippet) > max_chars:
            snippet = snippet[: max_chars] + "..."
        ref_id = str(hit.get("ref_id") or hit.get("chunk_id") or hit.get("doc_id") or "")
        trimmed.append(
            {
                "ref_id": ref_id,
                "score": hit.get("score"),
                "snippet": snippet,
            }
        )
    return trimmed


def _resolve_hit(hit: Any) -> (Optional[int], Optional[int], Optional[int]):
    if isinstance(hit, dict):
        if "doc_id" in hit:
            return hit.get("doc_id"), hit.get("start"), hit.get("end")
        if "chunk_id" in hit:
            return hit.get("doc_id"), hit.get("start"), hit.get("end")
    if isinstance(hit, int):
        return None, None, None
    return None, None, None


def _hash_vector(tokens: List[str], dims: int) -> List[float]:
    import hashlib

    vec = [0.0] * dims
    for token in tokens:
        h = hashlib.md5(token.encode("utf-8")).hexdigest()
        idx = int(h, 16) % dims
        vec[idx] += 1.0
    return vec


def _dot(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _vector_norm(vec: List[float]) -> float:
    return math.sqrt(sum(v * v for v in vec)) or 1.0
