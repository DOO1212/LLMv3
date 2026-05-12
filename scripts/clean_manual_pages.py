"""Clean extracted PDF page text and write normalized JSONL."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import OUTPUT_DIR

DEFAULT_IN_PATH = Path(OUTPUT_DIR) / "manual_pages.jsonl"
DEFAULT_OUT_PATH = Path(OUTPUT_DIR) / "manual_pages_cleaned.jsonl"

CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")
HANGUL_RE = re.compile(r"[가-힣]")
LATIN_RE = re.compile(r"[A-Za-z]")


def _normalize_line(line: str) -> str:
    line = CONTROL_CHARS_RE.sub("", line)
    line = re.sub(r"[ \t]+", " ", line).strip()
    return line


def _clean_text(text: str) -> str:
    text = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    lines = [_normalize_line(line) for line in text.split("\n")]
    kept = [line for line in lines if line]
    merged = "\n".join(kept)
    merged = re.sub(r"\n{3,}", "\n\n", merged).strip()
    return merged


def _quality_stats(text: str) -> dict[str, float]:
    total_chars = len(text)
    if total_chars == 0:
        return {
            "char_count": 0,
            "hangul_ratio": 0.0,
            "latin_ratio": 0.0,
            "needs_ocr_review": 1.0,
        }
    hangul_count = len(HANGUL_RE.findall(text))
    latin_count = len(LATIN_RE.findall(text))
    hangul_ratio = hangul_count / total_chars
    latin_ratio = latin_count / total_chars
    # Hangul manual pages with very low Hangul ratio are likely garbled extraction.
    needs_ocr_review = 1.0 if hangul_ratio < 0.15 else 0.0
    return {
        "char_count": float(total_chars),
        "hangul_ratio": round(hangul_ratio, 4),
        "latin_ratio": round(latin_ratio, 4),
        "needs_ocr_review": needs_ocr_review,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean extracted manual page JSONL")
    parser.add_argument("--in-file", default=str(DEFAULT_IN_PATH), help="input JSONL file")
    parser.add_argument(
        "--out-file", default=str(DEFAULT_OUT_PATH), help="output JSONL file"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    in_path = Path(args.in_file).expanduser().resolve()
    out_path = Path(args.out_file).expanduser().resolve()

    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    flagged_rows = 0
    with in_path.open("r", encoding="utf-8") as src, out_path.open(
        "w", encoding="utf-8"
    ) as dst:
        for raw_line in src:
            line = raw_line.strip()
            if not line:
                continue
            row = json.loads(line)
            cleaned_text = _clean_text(row.get("raw_text", ""))
            quality = _quality_stats(cleaned_text)
            if quality["needs_ocr_review"] >= 1.0:
                flagged_rows += 1
            out_row = {
                "source_file": row.get("source_file"),
                "page_no": row.get("page_no"),
                "cleaned_text": cleaned_text,
                "quality": quality,
            }
            dst.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            total_rows += 1

    print(
        f"[done] input_rows={total_rows} flagged_for_ocr={flagged_rows} out={out_path}"
    )


if __name__ == "__main__":
    main()
