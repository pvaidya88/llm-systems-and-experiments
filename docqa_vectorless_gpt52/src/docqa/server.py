from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .config import ensure_dirs, load_config
from .db import get_conn, init_db, list_documents
from .ingest import ingest_document
from .openai_client import OpenAIClient
from .qa import ask as ask_question

logger = logging.getLogger(__name__)

app = FastAPI(title="docqa-vectorless-gpt52")


class IngestRequest(BaseModel):
    path: str
    force: bool = False
    max_pages: int | None = None


class AskRequest(BaseModel):
    doc_id: str
    question: str


@app.on_event("startup")
def _startup() -> None:
    cfg = load_config()
    ensure_dirs(cfg)
    app.state.cfg = cfg
    app.state.client = OpenAIClient(cfg)
    conn = get_conn(cfg.db_path)
    init_db(conn)
    conn.close()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ingest")
def ingest(req: IngestRequest) -> dict[str, str]:
    cfg = app.state.cfg
    client = app.state.client
    try:
        doc_id = ingest_document(
            path=Path(req.path),
            cfg=cfg,
            client=client,
            force=req.force,
            max_pages=req.max_pages,
        )
        return {"doc_id": doc_id, "status": "READY"}
    except Exception as exc:  # noqa: BLE001
        logger.exception("Ingest failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/ask")
def ask(req: AskRequest) -> dict[str, object]:
    cfg = app.state.cfg
    client = app.state.client
    conn = get_conn(cfg.db_path)
    init_db(conn)
    try:
        return ask_question(conn=conn, client=client, cfg=cfg, doc_id=req.doc_id, question=req.question)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Ask failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        conn.close()


@app.get("/documents")
def documents() -> dict[str, object]:
    cfg = app.state.cfg
    conn = get_conn(cfg.db_path)
    init_db(conn)
    try:
        return {"documents": list_documents(conn)}
    finally:
        conn.close()
