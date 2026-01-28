from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RetrievalHit:
    ref_id: str
    score: Optional[float]
    snippet: str


@dataclass
class RetrievalEvent:
    op: str
    query: Optional[str]
    params: Dict[str, Any]
    hits: List[RetrievalHit] = field(default_factory=list)


@dataclass
class ReplBlockTrace:
    index: int
    code_chars: int
    output_chars: int
    truncated: bool
    error: Optional[str] = None


@dataclass
class SubcallTrace:
    prompt_chars: int
    system_chars: int
    response_chars: int


@dataclass
class RLMTrace:
    steps: int = 0
    repl_blocks: List[ReplBlockTrace] = field(default_factory=list)
    subcalls: List[SubcallTrace] = field(default_factory=list)
    retrieval: List[RetrievalEvent] = field(default_factory=list)
    assistant_messages: List[Dict[str, Any]] = field(default_factory=list)
    surfaced_chars: int = 0
    repl_output_chars: int = 0
    subcall_input_chars: int = 0
    subcall_output_chars: int = 0
    model_input_chars: int = 0
    rlm_queries: int = 0
    rlm_cache_hits: int = 0
    rlm_cache_misses: int = 0
    retrieval_cache_hits: int = 0
    retrieval_cache_misses: int = 0
    termination_reason: Optional[str] = None
