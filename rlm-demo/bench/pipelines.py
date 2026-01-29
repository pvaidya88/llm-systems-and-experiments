import json
import os
import re
import time
from typing import Any, Dict, List, Optional

from rlm_demo import RLM, RLMOptions
from rlm_demo.tools import ToolRegistry
from rlm_demo.llm import OpenAIResponsesClient

from .llm_clients import LLMWrapper
from .rerank.llm_reranker import LLMReranker


def _format_passages(passages: List[Dict[str, Any]]) -> str:
    lines = []
    for hit in passages:
        snippet = str(hit.get("text", ""))[:500]
        lines.append(
            f"[doc_id={hit.get('doc_id')} chunk_id={hit.get('chunk_id')} start={hit.get('start')} end={hit.get('end')}] {snippet}"
        )
    return "\n".join(lines)


def _estimate_result_chars(result: Any) -> int:
    if result is None:
        return 0
    if isinstance(result, str):
        return len(result)
    try:
        return len(json.dumps(result, ensure_ascii=False))
    except Exception:
        return len(str(result))


def _surfaced_chars_from_result(name: str, result: Any) -> int:
    if name not in ("bm25_search", "fetch_chunk", "expand_span"):
        return 0
    total = 0
    if isinstance(result, list):
        for item in result:
            if isinstance(item, dict):
                total += len(str(item.get("text", "")))
    elif isinstance(result, dict):
        total += len(str(result.get("text", "")))
    return total


def _add_hit(hit: Dict[str, Any], hits_by_key: Dict[str, Dict[str, Any]], ordered_hits: List[Dict[str, Any]]) -> None:
    if not hit:
        return
    chunk_id = hit.get("chunk_id")
    doc_id = hit.get("doc_id")
    start = hit.get("start")
    end = hit.get("end")
    key = str(chunk_id) if chunk_id is not None else f"{doc_id}:{start}:{end}"
    if key in hits_by_key:
        return
    hits_by_key[key] = hit
    ordered_hits.append(hit)


def _collect_hits(name: str, result: Any, hits_by_key: Dict[str, Dict[str, Any]], ordered_hits: List[Dict[str, Any]]) -> None:
    if name == "bm25_search" and isinstance(result, list):
        for item in result:
            if isinstance(item, dict):
                _add_hit(item, hits_by_key, ordered_hits)
    elif name == "fetch_chunk" and isinstance(result, dict):
        _add_hit(result, hits_by_key, ordered_hits)


def _generate_answer(question: str, passages: List[Dict[str, Any]], llm: LLMWrapper) -> str:
    prompt = (
        "Answer the question using ONLY the passages below. "
        "Cite sources with CITE(doc_id=<id>, chunk_id=<id>, start=<start>, end=<end>).\n"
        f"Question: {question}\n\nPassages:\n{_format_passages(passages)}\n\nAnswer:"
    )
    messages = [
        {"role": "system", "content": "You are a retrieval QA assistant."},
        {"role": "user", "content": prompt},
    ]
    return llm.complete(messages)


def bm25_only(question: str, bm25_index, k_retrieval: int = 5) -> Dict[str, Any]:
    start = time.perf_counter()
    hits = bm25_index.search(question, k_retrieval)
    retrieval_time = time.perf_counter() - start
    answer = hits[0]["text"] if hits else ""
    return {
        "answer": answer,
        "hits": hits,
        "latency": retrieval_time,
        "latency_breakdown": {
            "retrieval": retrieval_time,
            "rerank": 0.0,
            "generate": 0.0,
        },
    }


def _split_sentences(text: str) -> List[str]:
    parts = [seg.strip() for seg in re.split(r"(?<=[.!?])\\s+", text or "") if seg.strip()]
    return parts or ([text.strip()] if text.strip() else [])


def dense_extractive(
    question: str,
    dense_index,
    embed_client,
    k_retrieval: int = 5,
) -> Dict[str, Any]:
    start = time.perf_counter()
    hits = dense_index.search(question, k_retrieval, embed_client)
    retrieval_time = time.perf_counter() - start

    q_tokens = set(re.findall(r"[a-z0-9]+", (question or "").lower()))
    best_sentence = ""
    best_score = -1.0
    for hit in hits:
        for sentence in _split_sentences(str(hit.get("text", ""))):
            s_tokens = set(re.findall(r"[a-z0-9]+", sentence.lower()))
            if not q_tokens:
                score = 0.0
            else:
                score = len(q_tokens.intersection(s_tokens)) / max(1, len(q_tokens))
            if score > best_score:
                best_score = score
                best_sentence = sentence

    answer = best_sentence
    return {
        "answer": answer,
        "hits": hits,
        "latency": retrieval_time,
        "latency_breakdown": {
            "retrieval": retrieval_time,
            "rerank": 0.0,
            "generate": 0.0,
        },
    }


def vector_oracle(
    question: str,
    dense_index,
    embed_client,
    k_retrieval: int = 5,
    gold_answer: Optional[str] = None,
) -> Dict[str, Any]:
    start = time.perf_counter()
    hits = dense_index.search(question, k_retrieval, embed_client)
    retrieval_time = time.perf_counter() - start
    answer = ""
    if gold_answer:
        gold_lower = gold_answer.strip().lower()
        for hit in hits:
            if gold_lower and gold_lower in str(hit.get("text", "")).lower():
                answer = gold_answer
                break
    if not answer:
        answer = hits[0]["text"] if hits else ""
    return {
        "answer": answer,
        "hits": hits,
        "latency": retrieval_time,
        "latency_breakdown": {
            "retrieval": retrieval_time,
            "rerank": 0.0,
            "generate": 0.0,
        },
    }


def vector_rag(question: str, vector_index, reranker: LLMReranker, llm: LLMWrapper, k_retrieval: int, k_rerank: int, embed_client) -> Dict[str, Any]:
    start = time.perf_counter()
    hits = vector_index.search(question, k_retrieval, embed_client)
    retrieval_time = time.perf_counter() - start

    start = time.perf_counter()
    reranked = reranker.rerank(question, hits, top_k=k_rerank)
    rerank_time = time.perf_counter() - start

    start = time.perf_counter()
    answer = _generate_answer(question, reranked, llm)
    generate_time = time.perf_counter() - start

    return {
        "answer": answer,
        "hits": hits,
        "reranked": reranked,
        "latency": retrieval_time + rerank_time + generate_time,
        "latency_breakdown": {
            "retrieval": retrieval_time,
            "rerank": rerank_time,
            "generate": generate_time,
        },
    }


def hybrid_rag(question: str, bm25_index, vector_index, reranker: LLMReranker, llm: LLMWrapper, k_retrieval: int, k_rerank: int, embed_client, weights=(0.5, 0.5)) -> Dict[str, Any]:
    from .index.hybrid import hybrid_search

    start = time.perf_counter()
    bm25_hits = bm25_index.search(question, k_retrieval)
    vector_hits = vector_index.search(question, k_retrieval, embed_client)
    combined = hybrid_search(bm25_hits, vector_hits, k_retrieval, weights=weights)
    retrieval_time = time.perf_counter() - start

    start = time.perf_counter()
    reranked = reranker.rerank(question, combined, top_k=k_rerank)
    rerank_time = time.perf_counter() - start

    start = time.perf_counter()
    answer = _generate_answer(question, reranked, llm)
    generate_time = time.perf_counter() - start

    return {
        "answer": answer,
        "hits": combined,
        "reranked": reranked,
        "latency": retrieval_time + rerank_time + generate_time,
        "latency_breakdown": {
            "retrieval": retrieval_time,
            "rerank": rerank_time,
            "generate": generate_time,
        },
    }


def rlm_vectorless(question: str, tool_registry: ToolRegistry, options: Optional[RLMOptions] = None) -> Dict[str, Any]:
    tool_trace: List[Dict[str, Any]] = []
    hits_by_key: Dict[str, Dict[str, Any]] = {}
    ordered_hits: List[Dict[str, Any]] = []
    tool_stats = {"tool_calls": 0, "tool_result_chars": 0, "surfaced_chunk_chars": 0}

    def _logger(name: str, args: list, kwargs: dict, result: Any, elapsed_s: float) -> None:
        result_chars = _estimate_result_chars(result)
        tool_stats["tool_calls"] += 1
        tool_stats["tool_result_chars"] += result_chars
        tool_stats["surfaced_chunk_chars"] += _surfaced_chars_from_result(name, result)
        tool_trace.append(
            {
                "name": name,
                "args": args,
                "kwargs": kwargs,
                "result_chars": result_chars,
                "elapsed_ms": elapsed_s * 1000.0,
            }
        )
        _collect_hits(name, result, hits_by_key, ordered_hits)

    previous_logger = tool_registry.get_logger()
    tool_registry.set_logger(_logger)
    if options is None:
        options = RLMOptions(
            require_repl=True,
            retry_on_invalid=True,
            protocol="json",
            enable_ctx=False,
            remote_tools_enabled=True,
            tool_registry=tool_registry,
        )
    root = OpenAIResponsesClient(
        model=os.environ.get("OPENAI_MODEL", "gpt-5.2"),
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_BASE_URL"),
        reasoning_effort=os.environ.get("OPENAI_REASONING_EFFORT"),
        text_verbosity="low",
    )
    rlm = RLM(root_llm=root, sub_llm=root, options=options)
    prompt = (
        "You MUST use the REPL tools (tools.*).\n"
        "Use tools.bm25_search to retrieve, then tools.fetch_chunk.\n"
        "Cite sources using CITE(doc_id=<id>, chunk_id=<id>, start=<start>, end=<end>).\n"
        f"Question: {question}"
    )
    start = time.perf_counter()
    try:
        answer = rlm.answer(prompt, context={"note": "remote tools"})
    finally:
        tool_registry.set_logger(previous_logger)
    latency = time.perf_counter() - start
    return {
        "answer": answer,
        "hits": ordered_hits,
        "latency": latency,
        "latency_breakdown": {},
        "trace": rlm.last_trace,
        "tool_trace": tool_trace,
        "tool_stats": tool_stats,
    }
