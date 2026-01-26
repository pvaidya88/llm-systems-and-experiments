from __future__ import annotations

import datetime as _dt
import hashlib
import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


def now_iso() -> str:
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def make_prompt_cache_key(prefix: str, *parts: str, max_len: int = 64) -> str:
    base = ":".join([prefix, *parts])
    if len(base) <= max_len:
        return base
    digest = hashlib.sha256(base.encode("utf-8")).hexdigest()
    trimmed_prefix = prefix[: max(1, max_len - 1 - len(digest[:12]))]
    return f"{trimmed_prefix}:{digest[:12]}"


def parse_page_list(value: str) -> list[int]:
    if not value:
        return []
    pages: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_str, end_str = part.split("-", 1)
            if not start_str or not end_str:
                raise ValueError(f"Invalid page range: {part}")
            start = int(start_str)
            end = int(end_str)
            if start <= 0 or end <= 0 or end < start:
                raise ValueError(f"Invalid page range: {part}")
            pages.extend(range(start, end + 1))
        else:
            page = int(part)
            if page <= 0:
                raise ValueError(f"Invalid page number: {part}")
            pages.append(page)

    seen: set[int] = set()
    ordered: list[int] = []
    for page in pages:
        if page not in seen:
            ordered.append(page)
            seen.add(page)
    return ordered


def slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = text.strip("-")
    return text or "unnamed"


def safe_json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=True, separators=(",", ":"))


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return text.strip()


def string_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio()


def parse_number(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return None
    s = value.strip()
    if not s or s in {"-", "?", "?"}:
        return None

    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg = True
        s = s[1:-1].strip()

    s = s.replace(",", "")

    is_percent = False
    if s.endswith("%"):
        is_percent = True
        s = s[:-1].strip()

    s = re.sub(r"^[\$???]", "", s)
    s = re.sub(r"\s+[A-Za-z]+$", "", s)

    try:
        num = float(s)
    except ValueError:
        return None

    if neg:
        num = -num
    if is_percent:
        num = num / 100.0
    return num
