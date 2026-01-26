from __future__ import annotations

import logging
from pathlib import Path

from .config import Config, ensure_dirs
from .db import (
    clear_fts_for_doc,
    get_conn,
    get_document,
    init_db,
    set_document_status,
    upsert_document,
    upsert_object,
    upsert_page,
    upsert_table_json,
    upsert_figure_json,
)
from .extract_figure import extract_figure
from .extract_inventory import extract_page_inventory, inventory_to_objects
from .extract_table import extract_table, normalize_numeric_table
from .index import build_index_for_doc
from .openai_client import OpenAIClient
from .pdf_utils import ensure_pdf, pdf_page_count, split_pdf
from .utils import sha256_file

logger = logging.getLogger(__name__)


def ingest_document(
    *,
    path: Path,
    cfg: Config,
    client: OpenAIClient,
    force: bool = False,
) -> str:
    ensure_dirs(cfg)
    conn = get_conn(cfg.db_path)
    init_db(conn)

    input_path = path.resolve()
    pdf_path = ensure_pdf(input_path, cfg.data_dir / "converted")
    doc_id = sha256_file(pdf_path)
    existing = get_document(conn, doc_id)

    if existing and existing.get("extraction_version") == cfg.extraction_version and existing.get("status") == "READY":
        if not force:
            logger.info("Document already ingested: %s", doc_id)
            return doc_id

    if existing and (force or existing.get("extraction_version") != cfg.extraction_version):
        logger.info("Re-ingesting document %s", doc_id)
        conn.execute(
            "DELETE FROM extracted_table_json WHERE object_id IN (SELECT object_id FROM objects WHERE doc_id=?)",
            (doc_id,),
        )
        conn.execute(
            "DELETE FROM extracted_figure_json WHERE object_id IN (SELECT object_id FROM objects WHERE doc_id=?)",
            (doc_id,),
        )
        conn.execute("DELETE FROM objects WHERE doc_id=?", (doc_id,))
        conn.execute("DELETE FROM pages WHERE doc_id=?", (doc_id,))
        clear_fts_for_doc(conn, doc_id)
        conn.commit()

    num_pages = pdf_page_count(pdf_path)
    upsert_document(
        conn,
        doc_id=doc_id,
        name=input_path.name,
        original_path=str(input_path),
        sha256=doc_id,
        num_pages=num_pages,
        extraction_version=cfg.extraction_version,
        status="INGESTING",
    )

    page_dir = cfg.data_dir / "pages" / doc_id
    page_paths = split_pdf(pdf_path, page_dir)

    for page_number, page_path in enumerate(page_paths, start=1):
        file_id = client.upload_file(page_path)
        logger.info("Inventory extraction: page %s", page_number)
        inventory = extract_page_inventory(
            client=client,
            cfg=cfg,
            doc_id=doc_id,
            doc_name=input_path.name,
            page_number=page_number,
            page_pdf_path=page_path,
            openai_file_id=file_id,
        )

        objects = inventory_to_objects(inventory=inventory, doc_id=doc_id)
        upsert_page(
            conn,
            doc_id=doc_id,
            page_number=page_number,
            page_pdf_path=str(page_path),
            openai_file_id=file_id,
            page_summary=inventory.get("page_summary"),
            keywords=inventory.get("keywords"),
            inventory_json=inventory,
            extraction_version=cfg.extraction_version,
        )

        for obj in objects:
            upsert_object(
                conn,
                object_id=obj["object_id"],
                doc_id=doc_id,
                page_number=obj["page_number"],
                object_type=obj["object_type"],
                label=obj.get("label"),
                caption=obj.get("caption"),
                description=obj.get("description"),
                chart_type=obj.get("chart_type"),
                confidence=obj.get("confidence"),
                extraction_version=cfg.extraction_version,
            )

            if obj["object_type"] == "table":
                table_json = extract_table(
                    client=client,
                    cfg=cfg,
                    doc_id=doc_id,
                    page_number=page_number,
                    page_pdf_path=page_path,
                    openai_file_id=file_id,
                    object_id=obj["object_id"],
                    label=obj.get("label"),
                    caption=obj.get("caption"),
                    description=obj.get("description"),
                )
                normalized = normalize_numeric_table(table_json)
                upsert_table_json(
                    conn,
                    object_id=obj["object_id"],
                    table_json=table_json,
                    normalized_numeric_json=normalized,
                )

            if obj["object_type"] == "figure":
                figure_json = extract_figure(
                    client=client,
                    cfg=cfg,
                    doc_id=doc_id,
                    page_number=page_number,
                    page_pdf_path=page_path,
                    openai_file_id=file_id,
                    object_id=obj["object_id"],
                    label=obj.get("label"),
                    caption=obj.get("caption"),
                    description=obj.get("description"),
                )
                upsert_figure_json(
                    conn,
                    object_id=obj["object_id"],
                    figure_json=figure_json,
                )

    build_index_for_doc(conn, doc_id)
    set_document_status(conn, doc_id, "READY")
    return doc_id
