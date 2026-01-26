# docqa-vectorless-gpt52

Vectorless document QA with table + chart extraction using **only** OpenAI GPT-5.2 via the Responses API. No tools, no embeddings, no vector DB.

## Features
- PDF ingestion (DOCX/PPTX optional if conversion deps are installed)
- Splits PDFs into single-page PDFs
- Per-page object inventory (tables + figures) with stable object IDs
- Per-table and per-figure JSON extraction
- SQLite storage + FTS5 BM25 search index
- Query expansion -> sparse retrieval -> LLM rerank -> evidence assembly
- Deterministic numeric analysis for comparisons/aggregations
- Final answers with citations
- CLI + FastAPI server

## Requirements
- Python 3.10+
- `OPENAI_API_KEY` set in your environment

## Install
```bash
pip install -r requirements.txt
pip install -e .
```

## Usage
### Ingest
```bash
docqa ingest path/to/report.pdf
```
To limit processing to the first N pages:
```bash
docqa ingest path/to/report.pdf --max-pages 5
```

### Ask
```bash
docqa ask <doc_id> "Compare Table 1 vs Table 2 commitments and explain the main difference."
```

### Serve
```bash
docqa serve --host 0.0.0.0 --port 8000
```

## Environment variables
- `OPENAI_API_KEY` (required)
- `DOCQA_DATA_DIR` (default: `data`)
- `DOCQA_DB_PATH` (default: `data/docqa.db`)
- `DOCQA_EXTRACTION_VERSION` (default: `v1`)
- `DOCQA_PROMPT_CACHE_PREFIX` (default: `docqa`)
- `DOCQA_OPENAI_BASE_URL` (optional)
- `DOCQA_OPENAI_ORG` (optional)
- `DOCQA_OPENAI_PROJECT` (optional)
- `DOCQA_ATTACH_PAGE_PDF` (default: `false`, set to `true` to attach page PDFs during final answer)
> Note: The model is fixed to `gpt-5.2` to comply with project constraints.

## Optional DOCX/PPTX conversion
If you want DOCX or PPTX ingestion, install one of:
- `docx2pdf` (Windows, Word required)
- `pypandoc` + a local `pandoc` installation

## Notes
- Uses GPT-5.2 with structured outputs via JSON Schema.
- All calls disable tool use (`tools=[]`, `max_tool_calls=0`).
- For large PDFs, each page is processed independently to avoid oversized requests.

## Project layout
```
_docqa_vectorless_gpt52/
  README.md
  requirements.txt
  pyproject.toml
  src/docqa/
```
