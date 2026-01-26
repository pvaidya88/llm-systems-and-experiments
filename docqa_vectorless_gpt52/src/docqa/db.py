from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any

from .utils import now_iso, safe_json_dumps

logger = logging.getLogger(__name__)


def get_conn(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS documents (
            doc_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            original_path TEXT NOT NULL,
            sha256 TEXT NOT NULL,
            num_pages INTEGER NOT NULL,
            status TEXT NOT NULL,
            extraction_version TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS pages (
            doc_id TEXT NOT NULL,
            page_number INTEGER NOT NULL,
            page_pdf_path TEXT NOT NULL,
            openai_file_id TEXT,
            page_summary TEXT,
            keywords TEXT,
            inventory_json TEXT,
            extraction_version TEXT NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (doc_id, page_number),
            FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS objects (
            object_id TEXT PRIMARY KEY,
            doc_id TEXT NOT NULL,
            page_number INTEGER NOT NULL,
            object_type TEXT NOT NULL,
            label TEXT,
            caption TEXT,
            description TEXT,
            chart_type TEXT,
            confidence REAL,
            extraction_version TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS extracted_table_json (
            object_id TEXT PRIMARY KEY,
            json TEXT NOT NULL,
            normalized_numeric_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (object_id) REFERENCES objects(object_id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS extracted_figure_json (
            object_id TEXT PRIMARY KEY,
            json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (object_id) REFERENCES objects(object_id) ON DELETE CASCADE
        );
        """
    )

    conn.executescript(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS fts_index USING fts5(
            text,
            doc_id UNINDEXED,
            page_number UNINDEXED,
            object_id UNINDEXED,
            object_type UNINDEXED
        );
        """
    )


def upsert_document(
    conn: sqlite3.Connection,
    *,
    doc_id: str,
    name: str,
    original_path: str,
    sha256: str,
    num_pages: int,
    extraction_version: str,
    status: str,
) -> None:
    conn.execute(
        """
        INSERT INTO documents (doc_id, name, original_path, sha256, num_pages, status, extraction_version, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(doc_id) DO UPDATE SET
            name=excluded.name,
            original_path=excluded.original_path,
            sha256=excluded.sha256,
            num_pages=excluded.num_pages,
            status=excluded.status,
            extraction_version=excluded.extraction_version
        """,
        (
            doc_id,
            name,
            original_path,
            sha256,
            num_pages,
            status,
            extraction_version,
            now_iso(),
        ),
    )
    conn.commit()


def set_document_status(conn: sqlite3.Connection, doc_id: str, status: str) -> None:
    conn.execute("UPDATE documents SET status=? WHERE doc_id=?", (status, doc_id))
    conn.commit()


def get_document(conn: sqlite3.Connection, doc_id: str) -> dict[str, Any] | None:
    row = conn.execute("SELECT * FROM documents WHERE doc_id=?", (doc_id,)).fetchone()
    return dict(row) if row else None


def list_documents(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute("SELECT * FROM documents ORDER BY created_at DESC").fetchall()
    return [dict(row) for row in rows]


def upsert_page(
    conn: sqlite3.Connection,
    *,
    doc_id: str,
    page_number: int,
    page_pdf_path: str,
    openai_file_id: str | None,
    page_summary: str | None,
    keywords: list[str] | None,
    inventory_json: dict[str, Any] | None,
    extraction_version: str,
) -> None:
    conn.execute(
        """
        INSERT INTO pages (
            doc_id, page_number, page_pdf_path, openai_file_id, page_summary, keywords, inventory_json,
            extraction_version, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(doc_id, page_number) DO UPDATE SET
            page_pdf_path=excluded.page_pdf_path,
            openai_file_id=excluded.openai_file_id,
            page_summary=excluded.page_summary,
            keywords=excluded.keywords,
            inventory_json=excluded.inventory_json,
            extraction_version=excluded.extraction_version
        """,
        (
            doc_id,
            page_number,
            page_pdf_path,
            openai_file_id,
            page_summary,
            safe_json_dumps(keywords) if keywords is not None else None,
            safe_json_dumps(inventory_json) if inventory_json is not None else None,
            extraction_version,
            now_iso(),
        ),
    )
    conn.commit()


def get_page(conn: sqlite3.Connection, doc_id: str, page_number: int) -> dict[str, Any] | None:
    row = conn.execute(
        "SELECT * FROM pages WHERE doc_id=? AND page_number=?", (doc_id, page_number)
    ).fetchone()
    return dict(row) if row else None


def upsert_object(
    conn: sqlite3.Connection,
    *,
    object_id: str,
    doc_id: str,
    page_number: int,
    object_type: str,
    label: str | None,
    caption: str | None,
    description: str | None,
    chart_type: str | None,
    confidence: float | None,
    extraction_version: str,
) -> None:
    conn.execute(
        """
        INSERT INTO objects (
            object_id, doc_id, page_number, object_type, label, caption, description, chart_type, confidence,
            extraction_version, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(object_id) DO UPDATE SET
            doc_id=excluded.doc_id,
            page_number=excluded.page_number,
            object_type=excluded.object_type,
            label=excluded.label,
            caption=excluded.caption,
            description=excluded.description,
            chart_type=excluded.chart_type,
            confidence=excluded.confidence,
            extraction_version=excluded.extraction_version
        """,
        (
            object_id,
            doc_id,
            page_number,
            object_type,
            label,
            caption,
            description,
            chart_type,
            confidence,
            extraction_version,
            now_iso(),
        ),
    )
    conn.commit()


def get_object(conn: sqlite3.Connection, object_id: str) -> dict[str, Any] | None:
    row = conn.execute("SELECT * FROM objects WHERE object_id=?", (object_id,)).fetchone()
    return dict(row) if row else None


def get_objects_for_page(conn: sqlite3.Connection, doc_id: str, page_number: int) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT * FROM objects WHERE doc_id=? AND page_number=?", (doc_id, page_number)
    ).fetchall()
    return [dict(row) for row in rows]


def get_objects_metadata(conn: sqlite3.Connection, object_ids: list[str]) -> list[dict[str, Any]]:
    if not object_ids:
        return []
    placeholders = ",".join(["?"] * len(object_ids))
    rows = conn.execute(
        f"SELECT * FROM objects WHERE object_id IN ({placeholders})", tuple(object_ids)
    ).fetchall()
    return [dict(row) for row in rows]


def upsert_table_json(
    conn: sqlite3.Connection,
    *,
    object_id: str,
    table_json: dict[str, Any],
    normalized_numeric_json: dict[str, Any],
) -> None:
    conn.execute(
        """
        INSERT INTO extracted_table_json (object_id, json, normalized_numeric_json, created_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(object_id) DO UPDATE SET
            json=excluded.json,
            normalized_numeric_json=excluded.normalized_numeric_json
        """,
        (
            object_id,
            safe_json_dumps(table_json),
            safe_json_dumps(normalized_numeric_json),
            now_iso(),
        ),
    )
    conn.commit()


def upsert_figure_json(
    conn: sqlite3.Connection,
    *,
    object_id: str,
    figure_json: dict[str, Any],
) -> None:
    conn.execute(
        """
        INSERT INTO extracted_figure_json (object_id, json, created_at)
        VALUES (?, ?, ?)
        ON CONFLICT(object_id) DO UPDATE SET
            json=excluded.json
        """,
        (
            object_id,
            safe_json_dumps(figure_json),
            now_iso(),
        ),
    )
    conn.commit()


def get_table_json(conn: sqlite3.Connection, object_id: str) -> dict[str, Any] | None:
    row = conn.execute("SELECT json FROM extracted_table_json WHERE object_id=?", (object_id,)).fetchone()
    return json.loads(row[0]) if row else None


def get_figure_json(conn: sqlite3.Connection, object_id: str) -> dict[str, Any] | None:
    row = conn.execute("SELECT json FROM extracted_figure_json WHERE object_id=?", (object_id,)).fetchone()
    return json.loads(row[0]) if row else None


def clear_fts_for_doc(conn: sqlite3.Connection, doc_id: str) -> None:
    conn.execute("DELETE FROM fts_index WHERE doc_id=?", (doc_id,))
    conn.commit()


def add_fts_entry(
    conn: sqlite3.Connection,
    *,
    text: str,
    doc_id: str,
    page_number: int,
    object_id: str,
    object_type: str,
) -> None:
    conn.execute(
        "INSERT INTO fts_index (text, doc_id, page_number, object_id, object_type) VALUES (?, ?, ?, ?, ?)",
        (text, doc_id, page_number, object_id, object_type),
    )


def query_fts(
    conn: sqlite3.Connection,
    *,
    doc_id: str,
    query: str,
    limit: int,
) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT object_id, page_number, object_type, bm25(fts_index) AS score
        FROM fts_index
        WHERE doc_id = ? AND fts_index MATCH ?
        ORDER BY score
        LIMIT ?
        """,
        (doc_id, query, limit),
    ).fetchall()
    return [dict(row) for row in rows]
