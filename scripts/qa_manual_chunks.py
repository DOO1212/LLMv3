"""Run quality checks for manual chunk JSONL and save report."""

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

DEFAULT_IN_PATH = Path(OUTPUT_DIR) / "manual_chunks.jsonl"
DEFAULT_OUT_PATH = Path(OUTPUT_DIR) / "manual_chunks_qa_report.json"

HANGUL_RE = re.compile(r"[가-힣]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QA manual chunks")
    parser.add_argument("--in-file", default=str(DEFAULT_IN_PATH), help="input chunks JSONL")
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH), help="output QA report JSON")
    parser.add_argument("--short-threshold", type=int, default=180, help="short chunk threshold")
    parser.add_argument("--long-threshold", type=int, default=950, help="long chunk threshold")
    parser.add_argument(
        "--low-hangul-threshold",
        type=float,
        default=0.2,
        help="chunk low-hangul ratio threshold",
    )
    return parser.parse_args()


def _hangul_ratio(text: str) -> float:
    if not text:
        return 0.0
    return len(HANGUL_RE.findall(text)) / len(text)


def main() -> None:
    args = parse_args()
    in_path = Path(args.in_file).expanduser().resolve()
    out_path = Path(args.out_file).expanduser().resolve()

    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    short_threshold = max(1, int(args.short_threshold))
    long_threshold = max(short_threshold + 1, int(args.long_threshold))
    low_hangul_threshold = max(0.0, min(1.0, float(args.low_hangul_threshold)))

    chunks: list[dict] = []
    with in_path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            chunks.append(json.loads(line))

    issues = {
        "too_short": [],
        "too_long": [],
        "garbled_or_low_hangul": [],
    }
    section_counts: dict[str, int] = {}

    for chunk in chunks:
        chunk_id = int(chunk.get("chunk_id", -1))
        section_title = str(chunk.get("section_title", ""))
        page_start = chunk.get("page_start")
        page_end = chunk.get("page_end")
        text = str(chunk.get("chunk_text", ""))
        char_count = int(chunk.get("chunk_char_count", len(text)))
        ratio = round(_hangul_ratio(text), 4)
        section_counts[section_title] = section_counts.get(section_title, 0) + 1
        meta = {
            "chunk_id": chunk_id,
            "section_title": section_title,
            "page_start": page_start,
            "page_end": page_end,
            "chunk_char_count": char_count,
            "hangul_ratio": ratio,
        }
        if char_count < short_threshold:
            issues["too_short"].append(meta)
        if char_count > long_threshold:
            issues["too_long"].append(meta)
        if ratio < low_hangul_threshold:
            issues["garbled_or_low_hangul"].append(meta)

    report = {
        "input_file": str(in_path),
        "total_chunks": len(chunks),
        "thresholds": {
            "short_threshold": short_threshold,
            "long_threshold": long_threshold,
            "low_hangul_threshold": low_hangul_threshold,
        },
        "section_counts": section_counts,
        "issue_counts": {k: len(v) for k, v in issues.items()},
        "issues": issues,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fp:
        fp.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    print(
        "[done] "
        f"total_chunks={report['total_chunks']} "
        f"too_short={report['issue_counts']['too_short']} "
        f"too_long={report['issue_counts']['too_long']} "
        f"low_hangul={report['issue_counts']['garbled_or_low_hangul']} "
        f"out={out_path}"
    )


if __name__ == "__main__":
    main()
