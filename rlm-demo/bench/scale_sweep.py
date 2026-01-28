import argparse
import csv
import json
import os
import time
from typing import Dict, Any

from .loaders import load_corpus_jsonl, load_queries_jsonl, subsample_corpus
from .chunking import chunk_documents
from .index.bm25_index import BM25Index
from .index.vector_index import VectorIndex
from .llm_clients import OpenAIEmbeddingClient, LLMWrapper
from .rerank.llm_reranker import LLMReranker
from rlm_demo.llm import OpenAIResponsesClient
from .pipelines import hybrid_rag, vector_rag, rlm_vectorless
from rlm_demo.tools import ToolRegistry
from .metrics import exact_match


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        cfg = json.load(handle)

    sizes = cfg.get("sizes", [100000, 500000, 1000000])
    output_path = cfg.get("output", "scaling_report.csv")

    docs = list(load_corpus_jsonl(cfg["corpus_path"]))
    chunks = list(chunk_documents(docs, cfg.get("chunk_chars", 1000), cfg.get("overlap", 120)))
    subsamples = subsample_corpus(chunks, sizes, seed=cfg.get("seed", 42))

    queries = load_queries_jsonl(cfg["queries_path"])

    embed_model = os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-3-small")
    embed_client = OpenAIEmbeddingClient(embed_model, os.environ.get("OPENAI_API_KEY"), os.environ.get("OPENAI_BASE_URL"))
    reranker = LLMReranker(model=cfg.get("rerank_model"))
    gen_client = OpenAIResponsesClient(
        model=cfg.get("gen_model", "gpt-5.2"),
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_BASE_URL"),
        reasoning_effort=os.environ.get("OPENAI_REASONING_EFFORT"),
        text_verbosity="low",
    )
    llm = LLMWrapper(gen_client)

    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "size", "accuracy", "p50_latency_s"
        ])
        for size, subset in subsamples.items():
            bm25 = BM25Index.build(subset, os.path.join("artifacts", f"bm25_{size}.sqlite"))
            vector = VectorIndex.build(subset, embed_client, batch_size=cfg.get("embed_batch_size", 32), path=os.path.join("artifacts", f"vector_{size}"))
            tool_registry = ToolRegistry()
            tool_registry.register("bm25_search", lambda query, k=10: bm25.search(query, k))
            tool_registry.register("vector_search", lambda query, k=10: vector.search(query, k, embed_client))
            tool_registry.register("fetch_chunk", lambda chunk_id: bm25.fetch_chunk(str(chunk_id)))
            latencies = []
            correct = 0
            for ex in queries:
                result = hybrid_rag(ex.question, bm25, vector, reranker, llm, cfg.get("k_retrieval", 10), cfg.get("k_rerank", 5), embed_client)
                latencies.append(result["latency"])
                if ex.answer:
                    if exact_match(result.get("answer", ""), ex.answer) == 1.0:
                        correct += 1
            acc = correct / max(1, len(queries))
            latencies.sort()
            p50 = latencies[len(latencies)//2] if latencies else 0
            writer.writerow([size, acc, p50])
            print("size", size, "acc", acc, "p50", p50)


if __name__ == "__main__":
    main()
