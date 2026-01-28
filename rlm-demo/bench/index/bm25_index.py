import os
import sqlite3
from typing import Iterable, List, Dict, Any

from ..schema import Chunk


class BM25Index:
    def __init__(self, path: str):
        self._path = path
        self._conn = sqlite3.connect(self._path)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._init_tables()

    def _init_tables(self) -> None:
        self._conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING fts5(chunk_id, doc_id, start, end, text)"
        )
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS chunks_meta(chunk_id TEXT PRIMARY KEY, doc_id TEXT, start INT, end INT, text TEXT)"
        )
        self._conn.commit()

    @classmethod
    def build(cls, chunks: Iterable[Chunk], path: str) -> "BM25Index":
        if os.path.exists(path):
            os.remove(path)
        index = cls(path)
        index._bulk_insert(chunks)
        return index

    def _bulk_insert(self, chunks: Iterable[Chunk]) -> None:
        cur = self._conn.cursor()
        for chunk in chunks:
            cur.execute(
                "INSERT INTO chunks (chunk_id, doc_id, start, end, text) VALUES (?, ?, ?, ?, ?)",
                (chunk.chunk_id, chunk.doc_id, chunk.start, chunk.end, chunk.text),
            )
            cur.execute(
                "INSERT INTO chunks_meta (chunk_id, doc_id, start, end, text) VALUES (?, ?, ?, ?, ?)",
                (chunk.chunk_id, chunk.doc_id, chunk.start, chunk.end, chunk.text),
            )
        self._conn.commit()

    @classmethod
    def load(cls, path: str) -> "BM25Index":
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        return cls(path)

    def search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        cur = self._conn.cursor()
        cur.execute(
            "SELECT chunk_id, doc_id, start, end, text, bm25(chunks) as score FROM chunks WHERE chunks MATCH ? ORDER BY score LIMIT ?",
            (query, int(k)),
        )
        hits = []
        for row in cur.fetchall():
            hits.append(
                {
                    "chunk_id": row[0],
                    "doc_id": row[1],
                    "start": int(row[2]),
                    "end": int(row[3]),
                    "text": row[4],
                    "score": float(-row[5]),
                }
            )
        return hits

    def fetch_chunk(self, chunk_id: str) -> Dict[str, Any]:
        cur = self._conn.cursor()
        cur.execute(
            "SELECT chunk_id, doc_id, start, end, text FROM chunks_meta WHERE chunk_id = ?",
            (chunk_id,),
        )
        row = cur.fetchone()
        if not row:
            return {}
        return {
            "chunk_id": row[0],
            "doc_id": row[1],
            "start": int(row[2]),
            "end": int(row[3]),
            "text": row[4],
        }

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
