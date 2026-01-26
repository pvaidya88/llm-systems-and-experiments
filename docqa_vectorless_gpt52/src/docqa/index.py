from __future__ import annotations

import json
import logging
import sqlite3
from typing import Any

from .db import add_fts_entry, clear_fts_for_doc
from .utils import normalize_text

logger = logging.getLogger(__name__)


def build_index_for_doc(conn: sqlite3.Connection, doc_id: str) -> None:
    clear_fts_for_doc(conn, doc_id)

    pages = conn.execute("SELECT * FROM pages WHERE doc_id=?", (doc_id,)).fetchall()
    for row in pages:
        page_summary = row["page_summary"] or ""
        keywords = ""
        if row["keywords"]:
            try:
                keywords = " ".join(json.loads(row["keywords"]))
            except json.JSONDecodeError:
                keywords = row["keywords"]
        text = " ".join([page_summary, keywords]).strip()
        if text:
            add_fts_entry(
                conn,
                text=normalize_text(text),
                doc_id=doc_id,
                page_number=row["page_number"],
                object_id="",
                object_type="page",
            )

    objects = conn.execute("SELECT * FROM objects WHERE doc_id=?", (doc_id,)).fetchall()
    for row in objects:
        obj_type = row["object_type"]
        obj_text = _build_object_text(conn, row)
        if obj_text:
            add_fts_entry(
                conn,
                text=normalize_text(obj_text),
                doc_id=doc_id,
                page_number=row["page_number"],
                object_id=row["object_id"],
                object_type=obj_type,
            )

    conn.commit()


def _build_object_text(conn: sqlite3.Connection, obj_row: sqlite3.Row) -> str:
    parts: list[str] = []
    if obj_row["caption"]:
        parts.append(obj_row["caption"])
    if obj_row["label"]:
        parts.append(obj_row["label"])
    if obj_row["description"]:
        parts.append(obj_row["description"])

    if obj_row["object_type"] == "table":
        table_row = conn.execute(
            "SELECT json FROM extracted_table_json WHERE object_id=?", (obj_row["object_id"],)
        ).fetchone()
        if table_row:
            try:
                table_json = json.loads(table_row[0])
                parts.extend(_table_text_parts(table_json))
            except json.JSONDecodeError:
                logger.debug("Failed to parse table JSON for %s", obj_row["object_id"])
    if obj_row["object_type"] == "figure":
        fig_row = conn.execute(
            "SELECT json FROM extracted_figure_json WHERE object_id=?", (obj_row["object_id"],)
        ).fetchone()
        if fig_row:
            try:
                fig_json = json.loads(fig_row[0])
                parts.extend(_figure_text_parts(fig_json))
            except json.JSONDecodeError:
                logger.debug("Failed to parse figure JSON for %s", obj_row["object_id"])
    return " ".join([p for p in parts if p])


def _table_text_parts(table_json: dict[str, Any]) -> list[str]:
    parts: list[str] = []
    columns = table_json.get("columns", [])
    if columns:
        parts.append(" ".join([col.get("name", "") for col in columns]))

    rows = table_json.get("rows", [])
    row_texts: list[str] = []
    for row in rows[:200]:
        cell_texts = [str(cell) for cell in row[:50] if cell is not None]
        if cell_texts:
            row_texts.append(" ".join(cell_texts))
    if row_texts:
        parts.append(" ".join(row_texts))
    return parts


def _figure_text_parts(fig_json: dict[str, Any]) -> list[str]:
    parts: list[str] = []
    if fig_json.get("key_takeaways"):
        parts.extend(fig_json.get("key_takeaways"))

    chart = fig_json.get("chart")
    if chart:
        parts.append(chart.get("chart_type") or "")
        x_axis = chart.get("x_axis") or {}
        y_axis = chart.get("y_axis") or {}
        parts.append(x_axis.get("label") or "")
        parts.append(y_axis.get("label") or "")
        for series in chart.get("series", []):
            parts.append(series.get("name") or "")
    return [p for p in parts if p]
