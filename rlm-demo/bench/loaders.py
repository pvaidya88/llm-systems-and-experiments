import json
import os
from typing import Iterable, Iterator, List, Dict, Any, Optional

from .schema import Document, QueryExample, GoldSpan


def iter_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_corpus_jsonl(path: str) -> Iterator[Document]:
    for item in iter_jsonl(path):
        doc_id = str(item.get("doc_id") or item.get("id") or item.get("docid"))
        text = str(item.get("text") or "")
        yield Document(
            doc_id=doc_id,
            text=text,
            title=item.get("title"),
            metadata=item.get("metadata"),
            updated_at=item.get("updated_at"),
            source=item.get("source"),
        )


def load_corpus_dir(path: str, suffixes: Optional[List[str]] = None) -> Iterator[Document]:
    if suffixes is None:
        suffixes = [".txt", ".md"]
    for root, _, files in os.walk(path):
        for fname in files:
            if suffixes and not any(fname.lower().endswith(s) for s in suffixes):
                continue
            full_path = os.path.join(root, fname)
            with open(full_path, "r", encoding="utf-8", errors="ignore") as handle:
                text = handle.read()
            doc_id = os.path.relpath(full_path, path)
            yield Document(doc_id=doc_id, text=text, source=full_path)


def load_queries_jsonl(path: str) -> List[QueryExample]:
    examples: List[QueryExample] = []
    for item in iter_jsonl(path):
        spans = None
        if item.get("gold_spans"):
            spans = [GoldSpan(**span) for span in item["gold_spans"]]
        examples.append(
            QueryExample(
                qid=str(item.get("qid") or item.get("id")),
                question=str(item.get("question") or item.get("query")),
                answer=item.get("answer"),
                bucket=item.get("bucket"),
                gold_doc_ids=[str(x) for x in item.get("gold_doc_ids", [])],
                gold_spans=spans,
                gold_snippets=item.get("gold_snippets"),
                updated_at=item.get("updated_at"),
            )
        )
    return examples


def subsample_corpus(
    items: Iterable[Any],
    target_sizes: List[int],
    seed: int = 42,
) -> Dict[int, List[Any]]:
    """
    Deterministic subsampling using a stable hash and top-N selection.

    This keeps memory proportional to the largest target size.
    """
    target_sizes = sorted(set(int(x) for x in target_sizes))
    if not target_sizes:
        return {}

    def stable_hash(val: str) -> int:
        import hashlib

        data = f"{seed}:{val}".encode("utf-8")
        return int(hashlib.md5(data).hexdigest(), 16)

    # Maintain buffers for each target size.
    buffers: Dict[int, List[tuple]] = {size: [] for size in target_sizes}
    max_size = max(target_sizes)
    for item in items:
        item_id = getattr(item, "doc_id", None) or getattr(item, "chunk_id", None) or str(item)
        h = stable_hash(str(item_id))
        for size in target_sizes:
            buf = buffers[size]
            if len(buf) < size:
                buf.append((h, item))
                if len(buf) == size:
                    buf.sort(key=lambda x: x[0])
            else:
                if h < buf[-1][0]:
                    buf[-1] = (h, item)
                    buf.sort(key=lambda x: x[0])
    return {size: [item for _, item in buffers[size]] for size in target_sizes}
