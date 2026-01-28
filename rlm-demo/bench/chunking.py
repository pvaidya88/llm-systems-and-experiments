import json
import os
from typing import Iterable, Iterator, List, Dict, Any

from .schema import Document, Chunk


def chunk_documents(
    docs: Iterable[Document],
    chunk_chars: int = 1000,
    overlap: int = 120,
) -> Iterator[Chunk]:
    chunk_chars = max(50, int(chunk_chars))
    overlap = max(0, int(overlap))
    chunk_id = 0
    for doc in docs:
        text = doc.text or ""
        start = 0
        while start < len(text):
            end = min(len(text), start + chunk_chars)
            chunk_text = text[start:end]
            yield Chunk(
                chunk_id=str(chunk_id),
                doc_id=str(doc.doc_id),
                start=start,
                end=end,
                text=chunk_text,
            )
            chunk_id += 1
            if end == len(text):
                break
            start = max(0, end - overlap)


def persist_chunks(chunks: Iterable[Chunk], out_dir: str, dataset_id: str, meta: Dict[str, Any]) -> str:
    dataset_dir = os.path.join(out_dir, dataset_id)
    os.makedirs(dataset_dir, exist_ok=True)
    chunks_path = os.path.join(dataset_dir, "chunks.jsonl")
    meta_path = os.path.join(dataset_dir, "chunks_meta.json")
    with open(chunks_path, "w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, ensure_ascii=False, indent=2)
    return dataset_dir


def load_chunks(path: str) -> List[Chunk]:
    chunks: List[Chunk] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            chunks.append(
                Chunk(
                    chunk_id=str(data["chunk_id"]),
                    doc_id=str(data["doc_id"]),
                    start=int(data["start"]),
                    end=int(data["end"]),
                    text=str(data["text"]),
                )
            )
    return chunks
