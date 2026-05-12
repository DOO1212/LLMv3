"""Extract page-level text from a PDF into JSONL."""

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

DEFAULT_PDF_PATH = Path(OUTPUT_DIR) / "LG_CodeZero_A9_Air_manual_20240111.pdf"
DEFAULT_OUT_PATH = Path(OUTPUT_DIR) / "manual_pages.jsonl"


def _normalize_page_text(text: str) -> str:
    text = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _get_pdf_reader():
    try:
        from pypdf import PdfReader

        return PdfReader
    except ModuleNotFoundError:
        from PyPDF2 import PdfReader

        return PdfReader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract PDF pages into JSONL")
    parser.add_argument(
        "--pdf",
        default=str(DEFAULT_PDF_PATH),
        help="input PDF path",
    )
    parser.add_argument(
        "--out",
        default=str(DEFAULT_OUT_PATH),
        help="output JSONL path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pdf_path = Path(args.pdf).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)

    PdfReader = _get_pdf_reader()
    reader = PdfReader(str(pdf_path))

    written = 0
    with out_path.open("w", encoding="utf-8") as fp:
        for page_no, page in enumerate(reader.pages, start=1):
            text = _normalize_page_text(page.extract_text() or "")
            if not text:
                continue
            row = {
                "source_file": pdf_path.name,
                "page_no": page_no,
                "raw_text": text,
            }
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1

    print(
        f"[done] source={pdf_path.name} extracted_pages={written} total_pages={len(reader.pages)} out={out_path}"
    )


if __name__ == "__main__":
    main()
