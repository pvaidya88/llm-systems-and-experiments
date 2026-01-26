from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(frozen=True)
class Config:
    openai_api_key: str
    model: str
    data_dir: Path
    db_path: Path
    extraction_version: str
    prompt_cache_prefix: str
    openai_base_url: str | None
    openai_org: str | None
    openai_project: str | None
    max_retries: int
    request_timeout_s: int
    max_output_tokens_inventory: int
    max_output_tokens_table: int
    max_output_tokens_figure: int
    max_output_tokens_query: int
    max_output_tokens_rerank: int
    max_output_tokens_answer: int
    fts_top_n: int
    rerank_top_k: int
    attach_page_pdf: bool


def load_config() -> Config:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required")

    data_dir = Path(os.getenv("DOCQA_DATA_DIR", "data")).resolve()
    db_path = Path(os.getenv("DOCQA_DB_PATH", str(data_dir / "docqa.db"))).resolve()
    model = os.getenv("DOCQA_OPENAI_MODEL", "gpt-5.2")
    if model != "gpt-5.2":
        raise ValueError("Only model gpt-5.2 is supported for this project")

    return Config(
        openai_api_key=api_key,
        model="gpt-5.2",
        data_dir=data_dir,
        db_path=db_path,
        extraction_version=os.getenv("DOCQA_EXTRACTION_VERSION", "v1"),
        prompt_cache_prefix=os.getenv("DOCQA_PROMPT_CACHE_PREFIX", "docqa"),
        openai_base_url=os.getenv("DOCQA_OPENAI_BASE_URL"),
        openai_org=os.getenv("DOCQA_OPENAI_ORG"),
        openai_project=os.getenv("DOCQA_OPENAI_PROJECT"),
        max_retries=int(os.getenv("DOCQA_MAX_RETRIES", "3")),
        request_timeout_s=int(os.getenv("DOCQA_REQUEST_TIMEOUT_S", "120")),
        max_output_tokens_inventory=int(os.getenv("DOCQA_MAX_TOKENS_INVENTORY", "800")),
        max_output_tokens_table=int(os.getenv("DOCQA_MAX_TOKENS_TABLE", "2000")),
        max_output_tokens_figure=int(os.getenv("DOCQA_MAX_TOKENS_FIGURE", "1500")),
        max_output_tokens_query=int(os.getenv("DOCQA_MAX_TOKENS_QUERY", "500")),
        max_output_tokens_rerank=int(os.getenv("DOCQA_MAX_TOKENS_RERANK", "500")),
        max_output_tokens_answer=int(os.getenv("DOCQA_MAX_TOKENS_ANSWER", "1200")),
        fts_top_n=int(os.getenv("DOCQA_FTS_TOP_N", "25")),
        rerank_top_k=int(os.getenv("DOCQA_RERANK_TOP_K", "5")),
        attach_page_pdf=os.getenv("DOCQA_ATTACH_PAGE_PDF", "false").lower() in {"1", "true", "yes"},
    )


def ensure_dirs(cfg: Config) -> None:
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    cfg.db_path.parent.mkdir(parents=True, exist_ok=True)
