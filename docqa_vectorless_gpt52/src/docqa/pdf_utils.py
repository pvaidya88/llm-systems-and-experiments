from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

from pypdf import PdfReader, PdfWriter

logger = logging.getLogger(__name__)


def ensure_pdf(input_path: Path, output_dir: Path) -> Path:
    input_path = input_path.resolve()
    suffix = input_path.suffix.lower()
    if suffix == ".pdf":
        return input_path

    output_dir.mkdir(parents=True, exist_ok=True)
    output_pdf = output_dir / f"{input_path.stem}.pdf"

    if suffix == ".docx":
        if _try_docx2pdf(input_path, output_pdf):
            return output_pdf
    if suffix in {".docx", ".pptx"}:
        if _try_pandoc(input_path, output_pdf):
            return output_pdf

    raise ValueError(f"Unsupported input format or missing converter: {input_path}")


def split_pdf(pdf_path: Path, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    reader = PdfReader(str(pdf_path))
    page_paths: list[Path] = []
    for idx, page in enumerate(reader.pages, start=1):
        writer = PdfWriter()
        writer.add_page(page)
        out_path = output_dir / f"page_{idx:04d}.pdf"
        with out_path.open("wb") as f:
            writer.write(f)
        page_paths.append(out_path)
    return page_paths


def pdf_page_count(pdf_path: Path) -> int:
    reader = PdfReader(str(pdf_path))
    return len(reader.pages)


def iter_pdf_pages(pdf_path: Path) -> Iterable[int]:
    reader = PdfReader(str(pdf_path))
    return range(1, len(reader.pages) + 1)


def _try_docx2pdf(input_path: Path, output_pdf: Path) -> bool:
    try:
        import docx2pdf  # type: ignore

        docx2pdf.convert(str(input_path), str(output_pdf))
        return output_pdf.exists()
    except Exception as exc:  # noqa: BLE001
        logger.debug("docx2pdf conversion failed: %s", exc)
        return False


def _try_pandoc(input_path: Path, output_pdf: Path) -> bool:
    try:
        import pypandoc  # type: ignore

        tmp_pdf = pypandoc.convert_file(str(input_path), "pdf", outputfile=str(output_pdf))
        if tmp_pdf and Path(tmp_pdf).exists():
            return True
        return output_pdf.exists()
    except Exception as exc:  # noqa: BLE001
        logger.debug("pandoc conversion failed: %s", exc)
        return False
