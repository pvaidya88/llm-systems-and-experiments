import os
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
    answer = rlm.answer(prompt, context={"note": "remote tools"})
    latency = time.perf_counter() - start
    return {
        "answer": answer,
        "latency": latency,
        "latency_breakdown": {},
        "trace": rlm.last_trace,
    }
