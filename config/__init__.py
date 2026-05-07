"""Project configuration package."""

from .paths import (
    DATA_DIR,
    OUTPUT_DIR,
    PROJECT_ROOT,
    SQLITE_FILENAME,
    SQLITE_PATH,
)
from .quality import QUALITY_DEFECT_VALUES, QUALITY_OK_VALUES
from .schema import COLUMN_SYNONYMS, TARGET_FIELDS
from .search import KEYWORD_STRIP_WORDS, TOP_K

__all__ = [
    "COLUMN_SYNONYMS",
    "DATA_DIR",
    "KEYWORD_STRIP_WORDS",
    "OUTPUT_DIR",
    "PROJECT_ROOT",
    "QUALITY_DEFECT_VALUES",
    "QUALITY_OK_VALUES",
    "SQLITE_FILENAME",
    "SQLITE_PATH",
    "TARGET_FIELDS",
    "TOP_K",
]
