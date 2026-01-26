from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .config import Config
from .openai_client import OpenAIClient
from .schemas import INVENTORY_SCHEMA
from .utils import slugify

logger = logging.getLogger(__name__)

INVENTORY_INSTRUCTIONS = (
    "You are extracting a page-level inventory of tables and figures/charts from a PDF page. "
    "Return JSON that exactly matches the schema. Include quality and set quality.is_partial=true "
    "with warnings if you cannot read the page clearly; do not guess. For individual items you cannot "
    "read, use low confidence and keep descriptions minimal."
)


def make_object_id(doc_id: str, page_number: int, obj_type: str, label: str | None, index: int) -> str:
    if label:
        key = slugify(label)
    else:
        key = str(index)
    return f"{doc_id}:p{page_number}:{obj_type}:{key}"


def extract_page_inventory(
    *,
    client: OpenAIClient,
    cfg: Config,
    doc_id: str,
    doc_name: str,
    page_number: int,
    page_pdf_path: Path,
    openai_file_id: str | None,
) -> dict[str, Any]:
    file_id = openai_file_id or client.upload_file(page_pdf_path)
    prompt_cache_key = f"{cfg.prompt_cache_prefix}:{doc_id}:{cfg.extraction_version}:inventory"

    input_items = [
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": (
                        f"Document: {doc_name}\n"
                        f"Page: {page_number}\n"
                        "List every distinct table and figure/chart on this page."
                    ),
                },
                {"type": "input_file", "file_id": file_id},
            ],
        }
    ]

    inventory = client.call_json(
        instructions=INVENTORY_INSTRUCTIONS,
        input_items=input_items,
        schema=INVENTORY_SCHEMA,
        schema_name="page_inventory",
        prompt_cache_key=prompt_cache_key,
        max_output_tokens=cfg.max_output_tokens_inventory,
    )

    inventory["page_number"] = page_number
    return inventory


def inventory_to_objects(
    *,
    inventory: dict[str, Any],
    doc_id: str,
) -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    page_number = inventory.get("page_number")

    tables = inventory.get("tables", [])
    for idx, tbl in enumerate(tables, start=1):
        object_id = make_object_id(doc_id, page_number, "table", tbl.get("label"), idx)
        if object_id in used_ids:
            object_id = f"{object_id}-{idx}"
        used_ids.add(object_id)
        objects.append(
            {
                "object_id": object_id,
                "object_type": "table",
                "page_number": page_number,
                "label": tbl.get("label"),
                "caption": tbl.get("caption"),
                "description": tbl.get("description"),
                "chart_type": None,
                "confidence": tbl.get("confidence"),
            }
        )

    figures = inventory.get("figures", [])
    for idx, fig in enumerate(figures, start=1):
        object_id = make_object_id(doc_id, page_number, "figure", fig.get("label"), idx)
        if object_id in used_ids:
            object_id = f"{object_id}-{idx}"
        used_ids.add(object_id)
        objects.append(
            {
                "object_id": object_id,
                "object_type": "figure",
                "page_number": page_number,
                "label": fig.get("label"),
                "caption": fig.get("caption"),
                "description": fig.get("description"),
                "chart_type": fig.get("chart_type"),
                "confidence": fig.get("confidence"),
            }
        )
    return objects
