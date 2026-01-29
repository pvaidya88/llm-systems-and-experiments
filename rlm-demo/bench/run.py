import argparse
import json
import os
import time
from typing import Dict, Any, List, Optional

from .schema import Document, QueryExample, PerQueryMetrics
from .loaders import load_corpus_jsonl, load_queries_jsonl
from .chunking import chunk_documents, persist_chunks
from .index.bm25_index import BM25Index
from .index.vector_index import VectorIndex
from .index.dense_index import DenseIndex
from .rerank.llm_reranker import LLMReranker
from .llm_clients import OpenAIEmbeddingClient, SentenceTransformerEmbeddingClient, LLMWrapper
from rlm_demo.llm import OpenAIResponsesClient
from .metrics import (
    exact_match,
    f1_score,
    recall_at_k,
    citation_precision,
    hallucination_proxy,
    bootstrap_ci,
)
from .bucketing import infer_buckets
from .pipelines import vector_rag, hybrid_rag, rlm_vectorless, bm25_only, vector_oracle
from rlm_demo.tools import ToolRegistry


def _load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        text = handle.read()
    try:
        import yaml

        return yaml.safe_load(text)
    except Exception:
        return json.loads(text)


def _build_embed_client(cfg: Dict[str, Any]):
    backend = (cfg.get("embed_backend") or os.environ.get("EMBED_BACKEND") or "openai").lower()
    if backend in ("sentence-transformers", "sentence_transformers", "st", "sbert"):
        model = cfg.get("embed_model") or os.environ.get("EMBED_MODEL") or "sentence-transformers/all-MiniLM-L6-v2"
        device = cfg.get("embed_device") or os.environ.get("EMBED_DEVICE")
        batch_size = cfg.get("embed_batch_size", 32)
        return SentenceTransformerEmbeddingClient(model=model, device=device, batch_size=batch_size)

    embed_model = cfg.get("embed_model") or os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-3-small")
    return OpenAIEmbeddingClient(
        embed_model,
        os.environ.get("OPENAI_API_KEY"),
        os.environ.get("OPENAI_BASE_URL"),
    )


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _snapshot_stats(stats) -> Dict[str, int]:
    return {
        "calls": int(getattr(stats, "calls", 0)),
        "prompt": int(getattr(stats, "total_prompt_chars", 0)),
        "output": int(getattr(stats, "total_output_chars", 0)),
    }


def _delta_stats(before: Dict[str, int], after: Dict[str, int]) -> Dict[str, int]:
    return {
        "calls": max(0, after["calls"] - before["calls"]),
        "prompt": max(0, after["prompt"] - before["prompt"]),
        "output": max(0, after["output"] - before["output"]),
    }


def _avg(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / len(values)


def _percentile(values: List[float], pct: float) -> Optional[float]:
    if not values:
        return None
    values_sorted = sorted(values)
    idx = int(pct * (len(values_sorted) - 1))
    return values_sorted[idx]


def _cost_proxy(metrics: PerQueryMetrics) -> float:
    return float(
        (metrics.model_input_chars or 0)
        + (metrics.llm_prompt_chars or 0)
        + (metrics.llm_output_chars or 0)
        + (metrics.tool_result_chars or 0)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = _load_config(args.config)
    dataset_id = cfg["dataset_id"]
    run_id = cfg.get("run_id") or time.strftime("%Y%m%d_%H%M%S")
    artifacts_dir = os.path.join("artifacts", run_id)
    _ensure_dir(artifacts_dir)

    corpus_path = cfg["corpus_path"]
    queries_path = cfg["queries_path"]
    chunk_chars = cfg.get("chunk_chars", 1000)
    overlap = cfg.get("overlap", 120)

    docs = list(load_corpus_jsonl(corpus_path))
    doc_lookup = {doc.doc_id: doc.text for doc in docs}
    chunks = list(chunk_documents(docs, chunk_chars=chunk_chars, overlap=overlap))
    persist_chunks(chunks, "artifacts", dataset_id, {"chunk_chars": chunk_chars, "overlap": overlap})

    bm25_path = os.path.join("artifacts", dataset_id, "bm25.sqlite")
    bm25_index = BM25Index.build(chunks, bm25_path)

    pipeline_name = cfg.get("pipeline", "hybrid_rag")
    embed_client = None
    vector_index = None
    dense_index = None
    if pipeline_name in ("vector_rag", "hybrid_rag"):
        embed_client = _build_embed_client(cfg)
        vector_path = os.path.join("artifacts", dataset_id, "vector_index")
        vector_index = VectorIndex.build(
            chunks,
            embed_client,
            batch_size=cfg.get("embed_batch_size", 32),
            path=vector_path,
        )
    elif pipeline_name == "vector_oracle":
        embed_client = _build_embed_client(cfg)
        dense_index = DenseIndex.build(
            chunks,
            embed_client,
            batch_size=cfg.get("embed_batch_size", 32),
        )

    reranker = LLMReranker(model=cfg.get("rerank_model"))
    gen_client = OpenAIResponsesClient(
        model=cfg.get("gen_model", "gpt-5.2"),
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_BASE_URL"),
        reasoning_effort=os.environ.get("OPENAI_REASONING_EFFORT"),
        text_verbosity="low",
    )
    llm = LLMWrapper(gen_client)

    queries = load_queries_jsonl(queries_path)

    # chunk lookup
    chunk_lookup = {chunk.chunk_id: chunk.to_dict() for chunk in chunks}

    per_query_records = []
    summary = {"run_id": run_id, "dataset_id": dataset_id}

    k_retrieval = cfg.get("k_retrieval", 10)
    k_rerank = cfg.get("k_rerank", 5)

    tool_registry = ToolRegistry()
    tool_registry.register("bm25_search", lambda query, k=10: bm25_index.search(query, k))
    tool_registry.register("fetch_chunk", lambda chunk_id: bm25_index.fetch_chunk(str(chunk_id)))

    def _expand_span(doc_id: str, start: int, end: int, radius: int = 200):
        text = doc_lookup.get(str(doc_id), "")
        start = max(0, start - radius)
        end = min(len(text), end + radius)
        return {"doc_id": doc_id, "start": start, "end": end, "text": text[start:end]}

    tool_registry.register("expand_span", _expand_span)

    for ex in queries:
        gen_before = _snapshot_stats(llm.stats)
        rerank_before = _snapshot_stats(reranker.stats)
        if pipeline_name == "vector_rag":
            if vector_index is None or embed_client is None:
                raise RuntimeError("Vector index not initialized")
            result = vector_rag(ex.question, vector_index, reranker, llm, k_retrieval, k_rerank, embed_client)
        elif pipeline_name == "hybrid_rag":
            if vector_index is None or embed_client is None:
                raise RuntimeError("Vector index not initialized")
            result = hybrid_rag(ex.question, bm25_index, vector_index, reranker, llm, k_retrieval, k_rerank, embed_client)
        elif pipeline_name == "vector_oracle":
            if dense_index is None or embed_client is None:
                raise RuntimeError("Vector index not initialized")
            result = vector_oracle(ex.question, dense_index, embed_client, k_retrieval, gold_answer=ex.answer)
        elif pipeline_name == "bm25_only":
            result = bm25_only(ex.question, bm25_index, k_retrieval)
        else:
            result = rlm_vectorless(ex.question, tool_registry)
        gen_after = _snapshot_stats(llm.stats)
        rerank_after = _snapshot_stats(reranker.stats)
        gen_delta = _delta_stats(gen_before, gen_after)
        rerank_delta = _delta_stats(rerank_before, rerank_after)
        answer = result.get("answer", "")
        hits = result.get("hits", [])
        metrics = PerQueryMetrics(qid=ex.qid)
        if ex.answer:
            metrics.exact_match = exact_match(answer, ex.answer)
            metrics.f1 = f1_score(answer, ex.answer)
            metrics.correct = metrics.exact_match == 1.0
        if hits:
            metrics.recall_at_5 = recall_at_k(hits, ex.gold_doc_ids, 5)
            metrics.recall_at_10 = recall_at_k(hits, ex.gold_doc_ids, 10)
            metrics.recall_at_20 = recall_at_k(hits, ex.gold_doc_ids, 20)
        metrics.citation_precision = citation_precision(answer, chunk_lookup)
        metrics.hallucination = hallucination_proxy(answer, chunk_lookup)
        metrics.latency_ms = (result.get("latency", 0.0) or 0.0) * 1000
        metrics.retrieval_ms = (result.get("latency_breakdown", {}).get("retrieval", 0.0)) * 1000
        metrics.rerank_ms = (result.get("latency_breakdown", {}).get("rerank", 0.0)) * 1000
        metrics.generate_ms = (result.get("latency_breakdown", {}).get("generate", 0.0)) * 1000
        metrics.llm_prompt_chars = gen_delta["prompt"] + rerank_delta["prompt"]
        metrics.llm_output_chars = gen_delta["output"] + rerank_delta["output"]
        metrics.llm_calls = gen_delta["calls"] + rerank_delta["calls"]
        tool_stats = result.get("tool_stats") or {}
        metrics.tool_calls = tool_stats.get("tool_calls")
        metrics.tool_result_chars = tool_stats.get("tool_result_chars")
        metrics.surfaced_chunk_chars = tool_stats.get("surfaced_chunk_chars")
        trace = result.get("trace")
        if trace is not None:
            metrics.model_input_chars = getattr(trace, "model_input_chars", None)
        metrics.tool_trace = result.get("tool_trace")
        metrics.cost_proxy = _cost_proxy(metrics)

        per_query_records.append(metrics)

    # Summary
    summary["pipeline"] = pipeline_name
    summary["total_queries"] = len(per_query_records)
    if per_query_records:
        summary["accuracy"] = sum(1 for m in per_query_records if m.correct) / len(per_query_records)
        summary["em"] = sum(m.exact_match or 0 for m in per_query_records) / len(per_query_records)
        summary["f1"] = sum(m.f1 or 0 for m in per_query_records) / len(per_query_records)
        summary["recall_at_10"] = sum(m.recall_at_10 or 0 for m in per_query_records) / len(per_query_records)
        summary["hallucination_rate"] = sum(1 for m in per_query_records if m.hallucination) / len(per_query_records)

        acc_values = [1.0 if m.correct else 0.0 for m in per_query_records if m.correct is not None]
        summary["accuracy_ci"] = bootstrap_ci(acc_values)

        latencies = [m.latency_ms for m in per_query_records if m.latency_ms is not None]
        if latencies:
            latencies_sorted = sorted(latencies)
            summary["p50_latency"] = latencies_sorted[len(latencies_sorted) // 2]
            summary["p95_latency"] = latencies_sorted[int(0.95 * (len(latencies_sorted) - 1))]

        cost_values = [m.cost_proxy for m in per_query_records if m.cost_proxy is not None]
        summary["cost_per_query"] = _avg(cost_values)
        summary["p50_cost_proxy"] = _percentile(cost_values, 0.5)
        summary["p95_cost_proxy"] = _percentile(cost_values, 0.95)

        summary["avg_tool_calls"] = _avg([float(m.tool_calls or 0) for m in per_query_records])
        summary["avg_tool_result_chars"] = _avg([float(m.tool_result_chars or 0) for m in per_query_records])
        summary["avg_surfaced_chunk_chars"] = _avg([float(m.surfaced_chunk_chars or 0) for m in per_query_records])
        summary["avg_llm_prompt_chars"] = _avg([float(m.llm_prompt_chars or 0) for m in per_query_records])
        summary["avg_llm_output_chars"] = _avg([float(m.llm_output_chars or 0) for m in per_query_records])
        summary["avg_model_input_chars"] = _avg([float(m.model_input_chars or 0) for m in per_query_records])

        bucket_map = {}
        for ex, metrics in zip(queries, per_query_records):
            if ex.bucket:
                buckets = [ex.bucket]
            else:
                gold_text = ""
                if ex.gold_doc_ids:
                    gold_text = doc_lookup.get(str(ex.gold_doc_ids[0]), "")
                elif ex.gold_snippets:
                    gold_text = " ".join(ex.gold_snippets)
                buckets = infer_buckets(ex.question, gold_text, ex.gold_doc_ids)
            for bucket in buckets:
                bucket_map.setdefault(bucket, []).append(metrics)
        summary["buckets"] = {}
        for bucket, items in bucket_map.items():
            summary["buckets"][bucket] = {
                "accuracy": sum(1 for m in items if m.correct) / len(items),
                "recall_at_10": sum(m.recall_at_10 or 0 for m in items) / len(items),
            }

    # Write outputs
    per_query_path = os.path.join(artifacts_dir, "per_query.jsonl")
    with open(per_query_path, "w", encoding="utf-8") as handle:
        for m in per_query_records:
            handle.write(json.dumps(m.to_dict(), ensure_ascii=False) + "\n")

    summary_path = os.path.join(artifacts_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    summary_csv = os.path.join(artifacts_dir, "summary.csv")
    with open(summary_csv, "w", encoding="utf-8") as handle:
        headers = list(summary.keys())
        handle.write(",".join(headers) + "\n")
        handle.write(",".join(str(summary[h]) for h in headers))

    print(f"Wrote {per_query_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
