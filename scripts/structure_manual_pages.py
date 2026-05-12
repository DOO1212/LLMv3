"""Detect section structure from cleaned manual pages."""

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

DEFAULT_IN_PATH = Path(OUTPUT_DIR) / "manual_pages_cleaned.jsonl"
DEFAULT_OUT_PATH = Path(OUTPUT_DIR) / "manual_pages_structured.jsonl"
DEFAULT_TOC_PATH = Path(OUTPUT_DIR) / "manual_toc.json"

SECTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("안전을 위해 주의하기", re.compile(r"안전을\s*위해\s*주의하기")),
    ("알아보기", re.compile(r"알아보기")),
    ("사용하기", re.compile(r"사용하기")),
    ("관리하기", re.compile(r"관리하기")),
    ("고장 신고 전 확인하기", re.compile(r"고장\s*신고\s*전\s*확인하기")),
    ("제품 보증서 보기", re.compile(r"제품\s*보증서\s*보기|제\s*품\s*보증서\s*보기")),
    ("부록", re.compile(r"부록")),
    ("제품 규격", re.compile(r"제품\s*규격")),
    ("생활 속 전기안전 캠페인", re.compile(r"생활\s*속\s*전기안전\s*캠페인")),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect manual section structure")
    parser.add_argument("--in-file", default=str(DEFAULT_IN_PATH), help="input JSONL file")
    parser.add_argument(
        "--out-file", default=str(DEFAULT_OUT_PATH), help="output structured JSONL file"
    )
    parser.add_argument(
        "--toc-file", default=str(DEFAULT_TOC_PATH), help="output TOC JSON file"
    )
    return parser.parse_args()


def _pick_section(text: str, fallback: str) -> tuple[str, str]:
    for section_name, pattern in SECTION_PATTERNS:
        if pattern.search(text):
            return section_name, "keyword_match"
    return fallback, "carry_over"


def _first_nonempty_line(text: str) -> str:
    for line in text.split("\n"):
        line = line.strip()
        if line:
            return line
    return ""


def main() -> None:
    args = parse_args()
    in_path = Path(args.in_file).expanduser().resolve()
    out_path = Path(args.out_file).expanduser().resolve()
    toc_path = Path(args.toc_file).expanduser().resolve()

    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    toc_path.parent.mkdir(parents=True, exist_ok=True)

    current_section = "기타"
    toc: dict[str, int] = {}
    total_rows = 0
    detected_rows = 0

    with in_path.open("r", encoding="utf-8") as src, out_path.open(
        "w", encoding="utf-8"
    ) as dst:
        for raw_line in src:
            line = raw_line.strip()
            if not line:
                continue
            row = json.loads(line)
            cleaned_text = str(row.get("cleaned_text", ""))
            section_name, detection_mode = _pick_section(cleaned_text, current_section)
            if detection_mode == "keyword_match":
                current_section = section_name
                if section_name not in toc:
                    toc[section_name] = int(row.get("page_no", 0))
                detected_rows += 1

            out_row = {
                "source_file": row.get("source_file"),
                "page_no": row.get("page_no"),
                "section_title": current_section,
                "section_detection": detection_mode,
                "page_title_guess": _first_nonempty_line(cleaned_text),
                "cleaned_text": cleaned_text,
                "quality": row.get("quality", {}),
            }
            dst.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            total_rows += 1

    toc_rows = [{"section_title": k, "start_page": v} for k, v in toc.items()]
    toc_payload = {
        "source_file": "LG_CodeZero_A9_Air_manual_20240111.pdf",
        "sections": toc_rows,
        "total_sections": len(toc_rows),
    }
    with toc_path.open("w", encoding="utf-8") as fp:
        fp.write(json.dumps(toc_payload, ensure_ascii=False, indent=2) + "\n")

    print(
        "[done] "
        f"rows={total_rows} keyword_detected={detected_rows} "
        f"sections={len(toc_rows)} out={out_path} toc={toc_path}"
    )


if __name__ == "__main__":
    main()
