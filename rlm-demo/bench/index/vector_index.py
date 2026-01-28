import json
import os
from typing import Iterable, List, Dict, Any

from ..schema import Chunk
from ..llm_clients import EmbeddingClient


class VectorIndex:
    def __init__(self, dim: int, path: str):
        self._dim = dim
        self._path = path
        self._index = None
        self._meta = []

    @classmethod
    def build(
        cls,
        chunks: Iterable[Chunk],
        embed_client: EmbeddingClient,
        batch_size: int,
        path: str,
    ) -> "VectorIndex":
        try:
            import hnswlib
        except Exception as exc:
            raise RuntimeError("hnswlib is required for VectorIndex") from exc
        try:
            import numpy as np
        except Exception as exc:
            raise RuntimeError("numpy is required for VectorIndex") from exc

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
        vectors = np.vstack(vectors) if vectors else np.zeros((0, embed_client.dim))
        dim = vectors.shape[1] if vectors.size else embed_client.dim

        index = hnswlib.Index(space="cosine", dim=dim)
        index.init_index(max_elements=len(meta), ef_construction=200, M=16)
        if len(meta):
            index.add_items(vectors, list(range(len(meta))))
        index.set_ef(64)

        inst = cls(dim=dim, path=path)
        inst._index = index
        inst._meta = meta
        inst.save()
        return inst

    @classmethod
    def load(cls, path: str) -> "VectorIndex":
        try:
            import hnswlib
        except Exception as exc:
            raise RuntimeError("hnswlib is required for VectorIndex") from exc
        with open(os.path.join(path, "meta.json"), "r", encoding="utf-8") as handle:
            meta = json.load(handle)
        dim = int(meta["dim"])
        index = hnswlib.Index(space="cosine", dim=dim)
        index.load_index(os.path.join(path, "index.bin"))
        inst = cls(dim=dim, path=path)
        inst._index = index
        inst._meta = meta["chunks"]
        return inst

    def save(self) -> None:
        os.makedirs(self._path, exist_ok=True)
        if self._index is None:
            return
        self._index.save_index(os.path.join(self._path, "index.bin"))
        with open(os.path.join(self._path, "meta.json"), "w", encoding="utf-8") as handle:
            json.dump({"dim": self._dim, "chunks": self._meta}, handle, ensure_ascii=False)

    def add(self, chunks: Iterable[Chunk], embed_client: EmbeddingClient, batch_size: int) -> None:
        if self._index is None:
            raise RuntimeError("Index not initialized")
        try:
            import numpy as np
        except Exception as exc:
            raise RuntimeError("numpy is required for VectorIndex") from exc
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
            vecs = embed_client.embed_texts(texts[i : i + batch_size])
            vectors.append(vecs)
        vectors = np.vstack(vectors) if vectors else np.zeros((0, self._dim))
        start = len(self._meta)
        ids = list(range(start, start + len(meta)))
        if len(ids):
            self._index.add_items(vectors, ids)
            self._meta.extend(meta)

    def search(self, query: str, k: int, embed_client: EmbeddingClient) -> List[Dict[str, Any]]:
        if self._index is None:
            return []
        query_vec = embed_client.embed_query(query)
        labels, distances = self._index.knn_query(query_vec, k=k)
        hits = []
        if labels is None:
            return hits
        for idx, dist in zip(labels[0], distances[0]):
            if idx < 0 or idx >= len(self._meta):
                continue
            meta = self._meta[idx]
            hits.append(
                {
                    **meta,
                    "score": float(1.0 - dist),
                }
            )
        return hits
