"""Utility helpers."""
from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import Optional


def safe_filename(filename: str, max_len: int = 120) -> str:
    """Return a filesystem-safe display filename (never used for storage paths)."""
    name = Path(filename or "image").name
    name = re.sub(r"[^A-Za-z0-9._\- ]+", "_", name)
    name = name.strip("._ ") or "image"
    if len(name) > max_len:
        stem, dot, ext = name.rpartition(".")
        if dot:
            keep = max_len - len(ext) - 1
            name = stem[:keep] + "." + ext
        else:
            name = name[:max_len]
    return name


def temp_upload_name(original: str, ext: Optional[str] = None) -> str:
    safe = safe_filename(original)
    suffix = ext or Path(safe).suffix or ".img"
    return f"{uuid.uuid4().hex}{suffix}"
