from __future__ import annotations

import logging
import re
import sqlite3
from typing import Any

from .db import get_objects_for_page, query_fts

logger = logging.getLogger(__name__)


def build_query(question: str, expansion: dict[str, Any]) -> str:
    terms: list[str] = []
    question_clean = _sanitize(question)
    if question_clean:
        terms.append(f"\"{question_clean}\"")

    for key in ("keywords", "entities", "units", "time_ranges"):
        for item in expansion.get(key, []):
            cleaned = _sanitize(item)
            if cleaned:
                if " " in cleaned:
                    terms.append(f"\"{cleaned}\"")
                else:
                    terms.append(cleaned)

    if not terms:
        return _sanitize(question) or "*"

    return " OR ".join(dict.fromkeys(terms))


def search_candidates(
    conn: sqlite3.Connection,
    *,
    doc_id: str,
    query: str,
    top_n: int,
) -> list[str]:
    rows = query_fts(conn, doc_id=doc_id, query=query, limit=top_n)
    candidates: list[str] = []
    seen = set()

    for row in rows:
        object_id = row.get("object_id")
        if object_id:
            if object_id not in seen:
                candidates.append(object_id)
                seen.add(object_id)
            continue

        page_number = row.get("page_number")
        if page_number is None:
            continue
        page_objects = get_objects_for_page(conn, doc_id, page_number)
        for obj in page_objects:
            oid = obj.get("object_id")
            if oid and oid not in seen:
                candidates.append(oid)
                seen.add(oid)
    return candidates


def _sanitize(text: str) -> str:
    text = text or ""
    text = text.replace('"', " ")
    text = re.sub(r"[^A-Za-z0-9 ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
