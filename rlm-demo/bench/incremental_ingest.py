import argparse
import json
import os
from typing import List

from .loaders import load_corpus_jsonl, load_queries_jsonl
from .chunking import chunk_documents
from .index.bm25_index import BM25Index
from .index.vector_index import VectorIndex
from .llm_clients import OpenAIEmbeddingClient


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--delta", required=True)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--out", default="incremental_report.json")
    args = parser.parse_args()

    base_docs = list(load_corpus_jsonl(args.base))
    base_chunks = list(chunk_documents(base_docs))

    bm25_path = os.path.join("artifacts", "bm25_incremental.sqlite")
    bm25 = BM25Index.build(base_chunks, bm25_path)

    embed_model = os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-3-small")
    embed_client = OpenAIEmbeddingClient(embed_model, os.environ.get("OPENAI_API_KEY"), os.environ.get("OPENAI_BASE_URL"))
    vector_path = os.path.join("artifacts", "vector_incremental")
    vector = VectorIndex.build(base_chunks, embed_client, batch_size=32, path=vector_path)

    delta_docs = list(load_corpus_jsonl(args.delta))
    delta_chunks = list(chunk_documents(delta_docs))
    vector.add(delta_chunks, embed_client, batch_size=32)
    vector.save()

    queries = load_queries_jsonl(args.queries)
    outdated = 0
    for ex in queries:
        if ex.updated_at:
            outdated += 1

    report = {
        "delta_docs": len(delta_docs),
        "delta_chunks": len(delta_chunks),
        "outdated_queries": outdated,
        "note": "BM25 rebuild required for delta; vector index updated incrementally.",
    }
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
