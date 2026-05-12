"""Create chunk-level records from structured manual pages."""

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

DEFAULT_IN_PATH = Path(OUTPUT_DIR) / "manual_pages_structured.jsonl"
DEFAULT_OUT_PATH = Path(OUTPUT_DIR) / "manual_chunks.jsonl"
SECTION_OVERRIDE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("생활 속 전기안전 캠페인", re.compile(r"생활\s*속\s*전기안전\s*캠페인")),
    ("고장 신고 전 확인하기", re.compile(r"고장\s*신고\s*전\s*확인하기")),
    ("제품 보증서 보기", re.compile(r"제품\s*보증서\s*보기|제\s*품\s*보증서\s*보기")),
    ("안전을 위해 주의하기", re.compile(r"안전을\s*위해\s*주의하기")),
    ("제품 규격", re.compile(r"제품\s*규격")),
    ("알아보기", re.compile(r"(^|\n)\s*\d*\s*알아보기(\s|$)")),
    ("사용하기", re.compile(r"(^|\n)\s*\d*\s*사용하기(\s|$)")),
    ("관리하기", re.compile(r"(^|\n)\s*\d*\s*관리하기(\s|$)")),
    ("부록", re.compile(r"(^|\n)\s*\d*\s*부록(\s|$)")),
]


MINOR_TITLE_BLOCKLIST = {
    "경고",
    "주의",
    "준수 사항",
    "금지 사항",
    "알아두기",
    "알아두면 좋은 정보",
}


def _extract_minor_title(cleaned_text: str, major_title: str) -> str:
    """Pick only real subsection headings from content lines."""
    lines = [line.strip() for line in str(cleaned_text or "").split("\n") if line.strip()]
    if not lines:
        return major_title

    for raw in lines[:40]:
        line = re.sub(r"\s+", " ", raw)
        line = re.sub(r"^\d+\s*", "", line)
        line = re.sub(r"\s*\d+$", "", line)
        line = line.strip(" -:|")
        if not line:
            continue
        if line.startswith(("•", "-", "*")):
            continue
        if line in MINOR_TITLE_BLOCKLIST:
            continue
        if line == major_title:
            continue
        if line.endswith(("다.", "요.", "니다.", "습니다.", "합니다.", "됩니다.")):
            continue
        if any(p in line for p in (".", "?", "!", ":", "(", ")")):
            continue
        # Prefer noun+verb guide-like headings used in manuals.
        if re.search(r"(하기|보기|살펴보기|설치하기|분리하기|청소하기|사용하기|보관하기|조작하기|해결하기)$", line):
            return line
        # Also allow short clean noun headings (e.g. "제품 구성", "문제 해결하기").
        if 3 <= len(line) <= 14 and len(re.findall(r"[가-힣]", line)) >= 2:
            return line
    return major_title


def _normalize_for_chunking(text: str) -> str:
    text = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _split_to_units(text: str) -> list[str]:
    units: list[str] = []

    def _split_overlong(part: str, max_len: int = 240) -> list[str]:
        if len(part) <= max_len:
            return [part]
        sentence_parts = [p.strip() for p in re.split(r"(?<=\.)\s+", part) if p.strip()]
        if len(sentence_parts) <= 1:
            return [part]
        merged: list[str] = []
        current = ""
        for sent in sentence_parts:
            candidate = f"{current} {sent}".strip() if current else sent
            if len(candidate) <= max_len:
                current = candidate
                continue
            if current:
                merged.append(current)
            current = sent
        if current:
            merged.append(current)
        return merged if merged else [part]

    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        # Keep list-like manuals granular by splitting bullet-prefixed fragments.
        bullet_parts = [p.strip() for p in re.split(r"\s*(?=•)", line) if p.strip()]
        if not bullet_parts:
            bullet_parts = [line]
        for part in bullet_parts:
            units.extend(_split_overlong(part))
    return units


def _build_chunks(text: str, chunk_size: int, overlap: int) -> list[str]:
    units = _split_to_units(_normalize_for_chunking(text))
    if not units:
        return []

    chunks: list[str] = []
    current = ""

    def _tail_by_units(block: str, tail_chars: int) -> str:
        if tail_chars <= 0 or not block:
            return ""
        units = _split_to_units(block)
        selected: list[str] = []
        total = 0
        for unit in reversed(units):
            add_len = len(unit) + (1 if selected else 0)
            if selected and total + add_len > tail_chars:
                break
            selected.append(unit)
            total += add_len
            if total >= tail_chars:
                break
        selected.reverse()
        return "\n".join(selected).strip()

    for unit in units:
        candidate = f"{current}\n{unit}".strip() if current else unit
        if len(candidate) <= chunk_size:
            current = candidate
            continue
        if current:
            chunks.append(current)
            if overlap > 0:
                tail = _tail_by_units(current, overlap)
                current = f"{tail}\n{unit}".strip()
            else:
                current = unit
        else:
            # Unit itself exceeds chunk_size; hard split by character window.
            start = 0
            step = max(1, chunk_size - overlap)
            while start < len(unit):
                end = min(len(unit), start + chunk_size)
                chunks.append(unit[start:end].strip())
                if end >= len(unit):
                    current = ""
                    break
                start += step
    if current:
        chunks.append(current)
    return [c for c in chunks if c.strip()]


def _resolve_section_title(text: str, current: str | None) -> str:
    head = "\n".join(_split_to_units(text)[:5])
    for section_name, pattern in SECTION_OVERRIDE_PATTERNS:
        if pattern.search(head):
            return section_name
    return str(current or "기타")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chunk structured manual pages")
    parser.add_argument("--in-file", default=str(DEFAULT_IN_PATH), help="input JSONL file")
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH), help="output JSONL file")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=450,
        help="max chars per chunk",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=80,
        help="overlap chars between consecutive chunks",
    )
    parser.add_argument(
        "--skip-short-pages",
        type=int,
        default=3,
        help="skip pages with cleaned_text shorter than this length",
    )
    parser.add_argument(
        "--group-by",
        choices=["page", "section", "section_subsection"],
        default="section_subsection",
        help="chunking scope: per page or per section",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    in_path = Path(args.in_file).expanduser().resolve()
    out_path = Path(args.out_file).expanduser().resolve()

    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    chunk_size = max(200, int(args.chunk_size))
    overlap = max(0, min(int(args.overlap), chunk_size // 2))
    skip_short_pages = max(0, int(args.skip_short_pages))
    group_by = str(args.group_by)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    chunk_id = 1
    input_rows = 0
    skipped_rows = 0
    written_chunks = 0

    rows: list[dict] = []
    with in_path.open("r", encoding="utf-8") as src:
        for raw_line in src:
            line = raw_line.strip()
            if not line:
                continue
            input_rows += 1
            row = json.loads(line)
            text = str(row.get("cleaned_text", "")).strip()
            if len(text) < skip_short_pages:
                skipped_rows += 1
                continue
            row["cleaned_text"] = text
            rows.append(row)

    groups: list[dict] = []
    if group_by == "page":
        for row in rows:
            section_title = _resolve_section_title(
                text=row.get("cleaned_text", ""),
                current=row.get("section_title"),
            )
            groups.append(
                {
                    "source_file": row.get("source_file"),
                    "section_title": section_title,
                    "page_start": row.get("page_no"),
                    "page_end": row.get("page_no"),
                    "page_nos": [row.get("page_no")],
                    "page_title_guess": row.get("page_title_guess"),
                    "quality_rows": [row.get("quality", {})],
                    "text": row.get("cleaned_text", ""),
                }
            )
    elif group_by == "section":
        current_group: dict | None = None
        for row in rows:
            section_title = _resolve_section_title(
                text=row.get("cleaned_text", ""),
                current=row.get("section_title"),
            )
            source_file = row.get("source_file")
            page_no = int(row.get("page_no"))

            if (
                current_group is None
                or current_group["section_title"] != section_title
                or current_group["source_file"] != source_file
            ):
                if current_group is not None:
                    groups.append(current_group)
                current_group = {
                    "source_file": source_file,
                    "section_title": section_title,
                    "page_start": page_no,
                    "page_end": page_no,
                    "page_nos": [page_no],
                    "page_title_guess": row.get("page_title_guess"),
                    "quality_rows": [row.get("quality", {})],
                    "text_parts": [row.get("cleaned_text", "")],
                }
            else:
                current_group["page_end"] = page_no
                current_group["page_nos"].append(page_no)
                current_group["quality_rows"].append(row.get("quality", {}))
                current_group["text_parts"].append(row.get("cleaned_text", ""))
        if current_group is not None:
            groups.append(current_group)
        for group in groups:
            group["text"] = "\n".join(group.pop("text_parts"))
    else:
        # Hierarchical grouping: major(section) -> minor(page_title_guess)
        current_group: dict | None = None
        for row in rows:
            section_title = _resolve_section_title(
                text=row.get("cleaned_text", ""),
                current=row.get("section_title"),
            )
            subsection_title = _extract_minor_title(
                cleaned_text=row.get("cleaned_text", ""),
                major_title=section_title,
            )
            source_file = row.get("source_file")
            page_no = int(row.get("page_no"))

            if (
                current_group is None
                or current_group["section_title"] != section_title
                or current_group["subsection_title"] != subsection_title
                or current_group["source_file"] != source_file
            ):
                if current_group is not None:
                    groups.append(current_group)
                current_group = {
                    "source_file": source_file,
                    "section_title": section_title,
                    "subsection_title": subsection_title,
                    "page_start": page_no,
                    "page_end": page_no,
                    "page_nos": [page_no],
                    "page_title_guess": row.get("page_title_guess"),
                    "quality_rows": [row.get("quality", {})],
                    "text_parts": [row.get("cleaned_text", "")],
                }
            else:
                current_group["page_end"] = page_no
                current_group["page_nos"].append(page_no)
                current_group["quality_rows"].append(row.get("quality", {}))
                current_group["text_parts"].append(row.get("cleaned_text", ""))
        if current_group is not None:
            groups.append(current_group)
        for group in groups:
            group["text"] = "\n".join(group.pop("text_parts"))

    with out_path.open("w", encoding="utf-8") as dst:
        for group in groups:
            chunks = _build_chunks(
                text=group.get("text", ""),
                chunk_size=chunk_size,
                overlap=overlap,
            )
            if not chunks:
                continue

            avg_hangul_ratio = 0.0
            quality_rows = group.get("quality_rows", [])
            if quality_rows:
                ratios = [
                    float(item.get("hangul_ratio", 0.0))
                    for item in quality_rows
                    if isinstance(item, dict)
                ]
                if ratios:
                    avg_hangul_ratio = round(sum(ratios) / len(ratios), 4)

            for idx, chunk_text in enumerate(chunks, start=1):
                chunk_row = {
                    "chunk_id": chunk_id,
                    "source_file": group.get("source_file"),
                    "section_title": group.get("section_title"),
                    "subsection_title": group.get(
                        "subsection_title", group.get("section_title")
                    ),
                    "page_start": group.get("page_start"),
                    "page_end": group.get("page_end"),
                    "page_nos": group.get("page_nos"),
                    "page_title_guess": group.get("page_title_guess"),
                    "chunk_index_in_group": idx,
                    "group_mode": group_by,
                    "chunk_text": chunk_text,
                    "chunk_char_count": len(chunk_text),
                    "quality": {"avg_hangul_ratio": avg_hangul_ratio},
                }
                dst.write(json.dumps(chunk_row, ensure_ascii=False) + "\n")
                chunk_id += 1
                written_chunks += 1

    print(
        "[done] "
        f"group_by={group_by} input_rows={input_rows} skipped_rows={skipped_rows} "
        f"written_chunks={written_chunks} out={out_path}"
    )


if __name__ == "__main__":
    main()
