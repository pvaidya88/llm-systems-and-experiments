from typing import Iterable, List, Dict, Any

import numpy as np

from ..schema import Chunk
from ..llm_clients import EmbeddingClient


class DenseIndex:
    def __init__(self, vectors: np.ndarray, meta: List[Dict[str, Any]]):
        self._vectors = vectors
        self._meta = meta
        self._norms = np.linalg.norm(vectors, axis=1) if vectors.size else np.array([])

    @classmethod
    def build(
        cls,
        chunks: Iterable[Chunk],
        embed_client: EmbeddingClient,
        batch_size: int = 32,
    ) -> "DenseIndex":
        texts = []
        meta = []
        for chunk in chunks:
            texts.append(chunk.text)
            meta.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "start": chunk.start,
                    "end": chunk.end,
                    "text": chunk.text,
                }
            )
        vectors = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            vecs = embed_client.embed_texts(batch)
            vectors.append(vecs)
        if vectors:
            matrix = np.vstack(vectors)
        else:
            matrix = np.zeros((0, max(1, int(embed_client.dim or 1))))
        return cls(matrix, meta)

    def search(self, query: str, k: int, embed_client: EmbeddingClient) -> List[Dict[str, Any]]:
        if not self._vectors.size:
            return []
        query_vec = np.array(embed_client.embed_query(query), dtype=float).reshape(-1)
        if query_vec.size == 0:
            return []
        q_norm = np.linalg.norm(query_vec) or 1e-6
        norms = self._norms + 1e-6
        sims = (self._vectors @ query_vec) / (norms * q_norm)
        top_k = max(1, int(k))
        top_idx = np.argsort(-sims)[:top_k]
        hits = []
        for idx in top_idx:
            if idx < 0 or idx >= len(self._meta):
                continue
            hit = dict(self._meta[idx])
            hit["score"] = float(sims[idx])
            hits.append(hit)
        return hits
