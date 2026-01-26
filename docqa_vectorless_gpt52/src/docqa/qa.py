from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

from .config import Config
from .db import get_document, get_figure_json, get_object, get_objects_metadata, get_page, get_table_json
from .openai_client import OpenAIClient
from .retrieval import build_query, search_candidates
from .schemas import FINAL_ANSWER_SCHEMA, QUERY_EXPANSION_SCHEMA, RERANK_SCHEMA
from .utils import make_prompt_cache_key, parse_number, string_similarity

logger = logging.getLogger(__name__)

QUERY_INSTRUCTIONS = (
    "Expand the question into search hints for document QA. Return JSON that matches the schema. "
    "Use concise keywords and entities. likely_objects should contain table/figure hints."
)

RERANK_INSTRUCTIONS = (
    "You are reranking candidate objects to answer the question. Use only the provided metadata. "
    "Return the most relevant object_ids in ranked order. If nothing is relevant, return an empty list."
)

ANSWER_INSTRUCTIONS = (
    "You are answering a document question using provided evidence JSON and analysis payload. "
    "Do not invent numbers. Use citations like [Doc: <name>, Page: <n>, Object: Table 3] or [Figure 2]. "
    "If evidence is insufficient, set status=NOT_FOUND and explain briefly."
)


def ask(
    *,
    conn: sqlite3.Connection,
    client: OpenAIClient,
    cfg: Config,
    doc_id: str,
    question: str,
) -> dict[str, Any]:
    doc = get_document(conn, doc_id)
    if not doc:
        return {
            "answer": "Document not found.",
            "citations": [],
            "used_objects": [],
            "status": "NOT_FOUND",
        }

    expansion = client.call_json(
        instructions=QUERY_INSTRUCTIONS,
        input_items=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": f"Question: {question}"},
                ],
            }
        ],
        schema=QUERY_EXPANSION_SCHEMA,
        schema_name="query_expansion",
        prompt_cache_key=make_prompt_cache_key(cfg.prompt_cache_prefix, doc_id, cfg.extraction_version, "query"),
        max_output_tokens=cfg.max_output_tokens_query,
    )

    fts_query = build_query(question, expansion)
    candidate_ids = search_candidates(conn, doc_id=doc_id, query=fts_query, top_n=cfg.fts_top_n)
    if not candidate_ids:
        return {
            "answer": "No relevant pages or objects were found for this question.",
            "citations": [],
            "used_objects": [],
            "status": "NOT_FOUND",
        }

    metadata = get_objects_metadata(conn, candidate_ids)
    rerank_payload = {
        "question": question,
        "candidates": [
            {
                "object_id": obj["object_id"],
                "object_type": obj["object_type"],
                "page_number": obj["page_number"],
                "label": obj["label"],
                "caption": obj["caption"],
                "description": obj["description"],
            }
            for obj in metadata
        ],
    }

    rerank = client.call_json(
        instructions=RERANK_INSTRUCTIONS,
        input_items=[
            {
                "role": "user",
                "content": [{"type": "input_text", "text": json.dumps(rerank_payload)}],
            }
        ],
        schema=RERANK_SCHEMA,
        schema_name="rerank",
        prompt_cache_key=make_prompt_cache_key(cfg.prompt_cache_prefix, doc_id, cfg.extraction_version, "rerank"),
        max_output_tokens=cfg.max_output_tokens_rerank,
    )

    ranked = rerank.get("ranked_object_ids", [])
    if not ranked:
        return {
            "answer": "No relevant objects were identified after reranking.",
            "citations": [],
            "used_objects": [],
            "status": "NOT_FOUND",
        }

    selected = ranked[: cfg.rerank_top_k]

    evidence = []
    for object_id in selected:
        obj = get_object(conn, object_id)
        if not obj:
            continue
        if obj["object_type"] == "table":
            table_json = get_table_json(conn, object_id)
            if table_json:
                evidence.append(
                    {
                        "object_id": object_id,
                        "object_type": "table",
                        "page_number": obj["page_number"],
                        "label": obj["label"],
                        "caption": obj["caption"],
                        "table": table_json,
                    }
                )
        if obj["object_type"] == "figure":
            figure_json = get_figure_json(conn, object_id)
            if figure_json:
                evidence.append(
                    {
                        "object_id": object_id,
                        "object_type": "figure",
                        "page_number": obj["page_number"],
                        "label": obj["label"],
                        "caption": obj["caption"],
                        "figure": figure_json,
                    }
                )

    if not evidence:
        return {
            "answer": "No extracted evidence was available for the selected objects.",
            "citations": [],
            "used_objects": [],
            "status": "NOT_FOUND",
        }

    analysis_payload = deterministic_analysis(question, evidence)

    answer_payload = {
        "question": question,
        "document": {"doc_id": doc["doc_id"], "name": doc["name"]},
        "evidence": evidence,
        "analysis": analysis_payload,
        "citation_format": "[Doc: <name>, Page: <n>, Object: <label>] or [Figure <n>]",
    }

    content_items = [{"type": "input_text", "text": json.dumps(answer_payload)}]
    if cfg.attach_page_pdf:
        seen_pages: set[int] = set()
        for ev in evidence:
            page_number = int(ev["page_number"])
            if page_number in seen_pages:
                continue
            seen_pages.add(page_number)
            page = get_page(conn, doc_id, page_number)
            if not page:
                continue
            file_id = page.get("openai_file_id")
            if not file_id:
                page_path = Path(page["page_pdf_path"])
                file_id = client.upload_file(page_path)
            content_items.append({"type": "input_text", "text": f"Page {page_number} PDF for verification"})
            content_items.append({"type": "input_file", "file_id": file_id})

    final = client.call_json(
        instructions=ANSWER_INSTRUCTIONS,
        input_items=[
            {
                "role": "user",
                "content": content_items,
            }
        ],
        schema=FINAL_ANSWER_SCHEMA,
        schema_name="final_answer",
        prompt_cache_key=make_prompt_cache_key(cfg.prompt_cache_prefix, doc_id, cfg.extraction_version, "answer"),
        max_output_tokens=cfg.max_output_tokens_answer,
    )

    if final.get("status") == "NOT_FOUND" and final.get("answer"):
        return final

    final["used_objects"] = list({ev["object_id"] for ev in evidence})
    return final


def deterministic_analysis(question: str, evidence: list[dict[str, Any]]) -> dict[str, Any]:
    tables = [ev for ev in evidence if ev.get("object_type") == "table"]
    if not tables:
        return {"type": "none", "warnings": ["no tables in evidence"]}

    question_lower = question.lower()
    wants_compare = any(term in question_lower for term in ["compare", "difference", "versus", "vs", "change"])

    table_stats = {tbl["object_id"]: _compute_table_stats(tbl["table"]) for tbl in tables}
    analysis: dict[str, Any] = {
        "type": "table_stats",
        "tables": table_stats,
        "warnings": [],
    }

    if wants_compare and len(tables) >= 2:
        comparison = _compare_tables(tables[0]["table"], tables[1]["table"])
        analysis["type"] = "table_comparison"
        analysis["comparison"] = comparison
        analysis["warnings"].extend(comparison.get("warnings", []))

    return analysis


def _compute_table_stats(table_json: dict[str, Any]) -> dict[str, Any]:
    columns = table_json.get("columns", [])
    rows = table_json.get("rows", [])

    stats: dict[str, Any] = {}
    for idx, col in enumerate(columns):
        values = []
        for row in rows:
            if idx >= len(row):
                continue
            num = parse_number(row[idx])
            if num is not None:
                values.append(num)
        if values:
            stats[col.get("name") or f"col_{idx}"] = {
                "count": len(values),
                "sum": sum(values),
                "avg": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
            }
    return stats


def _compare_tables(table_a: dict[str, Any], table_b: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {"warnings": []}
    key_pair = _select_column_pair(table_a, table_b, prefer_numeric=False)
    num_pair = _select_column_pair(table_a, table_b, prefer_numeric=True)

    key_idx_a, key_idx_b = key_pair if key_pair else (_guess_text_column(table_a), _guess_text_column(table_b))
    num_idx_a, num_idx_b = num_pair if num_pair else (_guess_numeric_column(table_a), _guess_numeric_column(table_b))

    if not key_pair:
        result["warnings"].append("Key column match used heuristic fallback")
    if not num_pair:
        result["warnings"].append("Numeric column match used heuristic fallback")

    if key_idx_a is None or key_idx_b is None or num_idx_a is None or num_idx_b is None:
        result["warnings"].append("Could not determine comparable columns across tables")
        return result

    result["column_matching"] = {
        "table_a": {"key_idx": key_idx_a, "value_idx": num_idx_a},
        "table_b": {"key_idx": key_idx_b, "value_idx": num_idx_b},
    }

    map_a = _table_key_value(table_a, key_idx_a, num_idx_a)
    map_b = _table_key_value(table_b, key_idx_b, num_idx_b)

    if not map_a or not map_b:
        result["warnings"].append("Could not derive key-value mappings for comparison")
        return result

    shared_keys = set(map_a.keys()) & set(map_b.keys())
    comparisons = {}
    for key in shared_keys:
        comparisons[key] = {
            "a": map_a[key],
            "b": map_b[key],
            "difference": map_b[key] - map_a[key],
        }
    if not comparisons:
        result["warnings"].append("No overlapping keys between tables for comparison")
    result["comparisons"] = comparisons
    return result


def _table_key_value(table_json: dict[str, Any], key_idx: int, num_idx: int) -> dict[str, float]:
    columns = table_json.get("columns", [])
    rows = table_json.get("rows", [])
    if not columns or not rows:
        return {}
    if max(key_idx, num_idx) >= len(columns):
        return {}

    mapping: dict[str, float] = {}
    for row in rows:
        if max(key_idx, num_idx) >= len(row):
            continue
        key = str(row[key_idx]).strip()
        val = parse_number(row[num_idx])
        if key and val is not None:
            mapping[key] = val
    return mapping


def _column_numeric_ratios(table_json: dict[str, Any]) -> list[float]:
    columns = table_json.get("columns", [])
    rows = table_json.get("rows", [])
    ratios: list[float] = []
    for idx in range(len(columns)):
        values = [parse_number(row[idx]) for row in rows if idx < len(row)]
        values = [v for v in values if v is not None]
        ratio = len(values) / max(1, len(rows))
        ratios.append(ratio)
    return ratios


def _select_column_pair(
    table_a: dict[str, Any], table_b: dict[str, Any], *, prefer_numeric: bool
) -> tuple[int, int] | None:
    cols_a = table_a.get("columns", [])
    cols_b = table_b.get("columns", [])
    if not cols_a or not cols_b:
        return None

    ratios_a = _column_numeric_ratios(table_a)
    ratios_b = _column_numeric_ratios(table_b)

    candidates: list[tuple[float, int, int]] = []
    for i, col_a in enumerate(cols_a):
        if prefer_numeric and ratios_a[i] < 0.5:
            continue
        if not prefer_numeric and ratios_a[i] >= 0.5:
            continue
        for j, col_b in enumerate(cols_b):
            if prefer_numeric and ratios_b[j] < 0.5:
                continue
            if not prefer_numeric and ratios_b[j] >= 0.5:
                continue
            sim = string_similarity(col_a.get("name", ""), col_b.get("name", ""))
            if sim > 0:
                candidates.append((sim, i, j))

    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    best = candidates[0]
    if best[0] < 0.4:
        return None
    return (best[1], best[2])


def _guess_numeric_column(table_json: dict[str, Any]) -> int | None:
    ratios = _column_numeric_ratios(table_json)
    if not ratios:
        return None
    best_idx = max(range(len(ratios)), key=lambda i: ratios[i])
    return best_idx if ratios[best_idx] > 0 else None


def _guess_text_column(table_json: dict[str, Any]) -> int | None:
    ratios = _column_numeric_ratios(table_json)
    if not ratios:
        return None
    best_idx = min(range(len(ratios)), key=lambda i: ratios[i])
    return best_idx
