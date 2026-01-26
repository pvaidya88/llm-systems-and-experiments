from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .config import Config
from .openai_client import OpenAIClient
from .schemas import TABLE_SCHEMA
from .utils import parse_number

logger = logging.getLogger(__name__)

TABLE_INSTRUCTIONS = (
    "Extract ONLY the specified table object from the page. Ignore all other tables or figures. "
    "Return JSON that exactly matches the schema and always include quality. If you cannot read cells, "
    "set quality.is_partial=true and add warnings. Do not guess values."
)


def extract_table(
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
    prompt_cache_key = f"{cfg.prompt_cache_prefix}:{doc_id}:{cfg.extraction_version}:table"

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

    table_json = client.call_json(
        instructions=TABLE_INSTRUCTIONS,
        input_items=input_items,
        schema=TABLE_SCHEMA,
        schema_name="table_extraction",
        prompt_cache_key=prompt_cache_key,
        max_output_tokens=cfg.max_output_tokens_table,
    )

    table_json["object_id"] = object_id
    table_json["page_number"] = page_number
    return table_json


def normalize_numeric_table(table_json: dict[str, Any]) -> dict[str, Any]:
    rows = table_json.get("rows", [])
    normalized_rows = []
    for row in rows:
        normalized_row = []
        for cell in row:
            normalized_row.append({"raw": cell, "numeric": parse_number(cell)})
        normalized_rows.append(normalized_row)
    return {
        "object_id": table_json.get("object_id"),
        "page_number": table_json.get("page_number"),
        "columns": table_json.get("columns"),
        "rows": normalized_rows,
    }
