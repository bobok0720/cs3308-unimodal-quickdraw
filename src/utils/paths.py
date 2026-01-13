"""Path utilities for locating the repo root."""

from __future__ import annotations

from pathlib import Path
from typing import Optional


def find_repo_root(start: Optional[Path] = None) -> Path:
    """Find repo root by searching for a directory containing ``src``."""

    start = start or Path.cwd()
    start = start.resolve()
    for path in [start, *start.parents]:
        if (path / "src").is_dir():
            return path
    raise FileNotFoundError("Could not locate repo root containing src/")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
