from __future__ import annotations

import json
import logging
from pathlib import Path

import typer

from .config import ensure_dirs, load_config
from .db import get_conn, init_db, list_documents
from .ingest import ingest_document
from .openai_client import OpenAIClient
from .qa import ask as ask_question
from .utils import parse_page_list

app = typer.Typer(help="Vectorless document QA with GPT-5.2")


@app.command()
def ingest(
    path: str,
    force: bool = False,
    max_pages: int | None = None,
    page_list: str | None = None,
) -> None:
    """Ingest a document (PDF required; DOCX/PPTX optional if conversion available)."""
    cfg = load_config()
    ensure_dirs(cfg)
    client = OpenAIClient(cfg)
    pages = parse_page_list(page_list) if page_list else None
    if pages and max_pages:
        raise typer.BadParameter("Use either --max-pages or --page-list, not both.")
    doc_id = ingest_document(
        path=Path(path),
        cfg=cfg,
        client=client,
        force=force,
        max_pages=max_pages,
        page_numbers=pages,
    )
    typer.echo(doc_id)


@app.command()
def ask(doc_id: str, question: str) -> None:
    """Ask a question against an ingested document."""
    cfg = load_config()
    ensure_dirs(cfg)
    client = OpenAIClient(cfg)
    conn = get_conn(cfg.db_path)
    init_db(conn)
    result = ask_question(conn=conn, client=client, cfg=cfg, doc_id=doc_id, question=question)
    typer.echo(json.dumps(result, indent=2))


@app.command("list")
def list_docs() -> None:
    """List ingested documents."""
    cfg = load_config()
    conn = get_conn(cfg.db_path)
    init_db(conn)
    docs = list_documents(conn)
    typer.echo(json.dumps(docs, indent=2))


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Run the FastAPI server."""
    import uvicorn

    uvicorn.run("docqa.server:app", host=host, port=port, reload=False)


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


if __name__ == "__main__":
    _configure_logging()
    app()
