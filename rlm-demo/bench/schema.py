from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class Document:
    doc_id: str
    text: str
    title: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    updated_at: Optional[str] = None
    source: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    start: int
    end: int
    text: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GoldSpan:
    doc_id: str
    start: int
    end: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class QueryExample:
    qid: str
    question: str
    answer: Optional[str] = None
    bucket: Optional[str] = None
    gold_doc_ids: List[str] = field(default_factory=list)
    gold_spans: Optional[List[GoldSpan]] = None
    gold_snippets: Optional[List[str]] = None
    updated_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        if self.gold_spans is not None:
            data["gold_spans"] = [span.to_dict() for span in self.gold_spans]
        return data


@dataclass
class RunConfig:
    run_id: str
    dataset_id: str
    corpus_path: str
    queries_path: str
    pipeline: str
    seed: int = 42
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PerQueryMetrics:
    qid: str
    correct: Optional[bool] = None
    exact_match: Optional[float] = None
    f1: Optional[float] = None
    recall_at_5: Optional[float] = None
    recall_at_10: Optional[float] = None
    recall_at_20: Optional[float] = None
    hallucination: Optional[bool] = None
    citation_precision: Optional[float] = None
    latency_ms: Optional[float] = None
    retrieval_ms: Optional[float] = None
    rerank_ms: Optional[float] = None
    generate_ms: Optional[float] = None
    cost_proxy: Optional[float] = None
    tool_calls: Optional[int] = None
    tool_result_chars: Optional[int] = None
    surfaced_chunk_chars: Optional[int] = None
    model_input_chars: Optional[int] = None
    llm_prompt_chars: Optional[int] = None
    llm_output_chars: Optional[int] = None
    llm_calls: Optional[int] = None
    tool_trace: Optional[List[Dict[str, Any]]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RunResult:
    run_id: str
    pipeline: str
    total_queries: int
    metrics: Dict[str, Any]
    per_query: List[PerQueryMetrics]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "pipeline": self.pipeline,
            "total_queries": self.total_queries,
            "metrics": self.metrics,
            "per_query": [item.to_dict() for item in self.per_query],
        }
