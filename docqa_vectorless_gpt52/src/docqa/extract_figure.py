from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .config import Config
from .openai_client import OpenAIClient
from .schemas import FIGURE_SCHEMA

logger = logging.getLogger(__name__)

FIGURE_INSTRUCTIONS = (
    "Extract ONLY the specified figure/chart object from the page. Ignore all other tables or figures. "
    "Return JSON that exactly matches the schema and always include quality. If you cannot read values, "
    "set quality.is_partial=true and add warnings; use nulls and chart.extraction_warnings for chart "
    "details. Do not guess. If the figure is not a chart, set chart=null and still provide key_takeaways."
)


def extract_figure(
    *,
    client: OpenAIClient,
    cfg: Config,
    doc_id: str,
    page_number: int,
    page_pdf_path: Path,
    openai_file_id: str | None,
    object_id: str,
    label: str | None,
    caption: str | None,
    description: str | None,
) -> dict[str, Any]:
    file_id = openai_file_id or client.upload_file(page_pdf_path)
    prompt_cache_key = f"{cfg.prompt_cache_prefix}:{doc_id}:{cfg.extraction_version}:figure"

    descriptor = (
        f"Object ID: {object_id}\n"
        f"Page: {page_number}\n"
        f"Label: {label}\n"
        f"Caption: {caption}\n"
        f"Description: {description}\n"
    )

    input_items = [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": descriptor},
                {"type": "input_file", "file_id": file_id},
            ],
        }
    ]

    figure_json = client.call_json(
        instructions=FIGURE_INSTRUCTIONS,
        input_items=input_items,
        schema=FIGURE_SCHEMA,
        schema_name="figure_extraction",
        prompt_cache_key=prompt_cache_key,
        max_output_tokens=cfg.max_output_tokens_figure,
    )

    figure_json["object_id"] = object_id
    figure_json["page_number"] = page_number
    return figure_json
