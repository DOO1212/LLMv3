"""Filesystem paths used by the SQLite inventory workflow."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

PROJECT_ROOT: Final[Path] = Path(
    os.environ.get("LLM_PROJECT_ROOT", Path(__file__).resolve().parent.parent)
).resolve()

DATA_DIR: Final[str] = str(
    Path(os.environ.get("LLM_DATA_DIR", PROJECT_ROOT / "data")).resolve()
)
OUTPUT_DIR: Final[str] = str(
    Path(os.environ.get("LLM_OUTPUT_DIR", PROJECT_ROOT / "output")).resolve()
)

SQLITE_FILENAME: Final[str] = "inventory.sqlite3"

SQLITE_PATH: Final[str] = str(Path(OUTPUT_DIR) / SQLITE_FILENAME)

os.makedirs(OUTPUT_DIR, exist_ok=True)
