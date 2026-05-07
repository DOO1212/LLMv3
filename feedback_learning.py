"""피드백 로그 기반 발주 파싱 규칙 자동 학습."""

from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


FEEDBACK_LOG_PATH = Path("output/feedback_log.jsonl")
LEARNED_RULES_PATH = Path("output/feedback_learned_rules.json")

COMMAND_TOKENS = {
    "발주서",
    "발주",
    "작성",
    "써줘",
    "써",
    "만들어줘",
    "만들어",
    "정리해줘",
    "정리",
    "부족",
    "없는",
    "것들",
    "것",
    "중에",
}

QUESTION_NOISE_TOKENS = {
    "중에",
    "것",
    "것들",
    "없는",
    "부족한",
    "부족",
}

QUESTION_LOW_STOCK_SIGNALS = {
    "없는",
    "부족",
    "부족한",
    "품절",
    "미달",
}


def _tokenize(text: str) -> list[str]:
    tokens = []
    for raw in re.split(r"\s+", text.strip()):
        tok = re.sub(r"[^\w가-힣]", "", raw).strip()
        if tok:
            tokens.append(tok)
    return tokens


def _read_feedback_rows() -> list[dict]:
    if not FEEDBACK_LOG_PATH.exists():
        return []
    rows: list[dict] = []
    for line in FEEDBACK_LOG_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def _extract_keyword_from_row(row: dict) -> str | None:
    parsed = row.get("parsed_query") or {}
    if isinstance(parsed, dict):
        filters = parsed.get("filters")
        if isinstance(filters, dict):
            kw = filters.get("keyword")
            if isinstance(kw, str) and kw.strip():
                return kw.strip()
        kw = parsed.get("keyword")
        if isinstance(kw, str) and kw.strip():
            return kw.strip()
    return None


def build_learned_rules(rows: list[dict]) -> dict:
    dislikes = [r for r in rows if r.get("feedback") == "dislike"]
    po_dislikes = [r for r in dislikes if str(r.get("route") or "") == "purchase_order"]

    token_counts: Counter[str] = Counter()
    question_noise_counts: Counter[str] = Counter()
    low_signal_counts: Counter[str] = Counter()
    no_stock_hits = 0
    for row in po_dislikes:
        q = str(row.get("question") or "")
        compact = re.sub(r"\s+", "", q)
        if "재고없" in compact or "재고없는" in compact:
            no_stock_hits += 1
        for tok in _tokenize(q):
            if tok in QUESTION_NOISE_TOKENS:
                question_noise_counts[tok] += 1
            if tok in QUESTION_LOW_STOCK_SIGNALS:
                low_signal_counts[tok] += 1

        kw = _extract_keyword_from_row(row)
        if not kw:
            continue
        for tok in _tokenize(kw):
            if tok in COMMAND_TOKENS:
                token_counts[tok] += 1

    # 데이터가 적으면 1회 등장도 반영, 커지면 2회 이상만 반영
    min_freq = 1 if len(po_dislikes) < 5 else 2
    learned_tail_words = sorted(
        {
            tok
            for tok, cnt in token_counts.items()
            if cnt >= min_freq
        }
        | {
            tok
            for tok, cnt in question_noise_counts.items()
            if cnt >= min_freq
        }
    )

    learned_noise_patterns: list[str] = []
    learned_low_stock_tokens: list[str] = [
        tok for tok, cnt in low_signal_counts.items() if cnt >= min_freq
    ]
    if no_stock_hits >= 1:
        learned_noise_patterns.extend(
            [
                r"재고(?:가)?\s*없는\s*것(?:들)?",
                r"재고(?:가)?\s*없(?:는|음)",
                r"품절",
            ]
        )
        learned_low_stock_tokens.extend(["재고없", "품절"])

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_feedback_rows": len(rows),
        "source_dislike_rows": len(dislikes),
        "source_purchase_order_dislikes": len(po_dislikes),
        "learned_tail_words": learned_tail_words,
        "learned_noise_patterns": sorted(set(learned_noise_patterns)),
        "learned_low_stock_tokens": sorted(set(learned_low_stock_tokens)),
    }


def retrain_feedback_rules_if_needed(force: bool = False) -> bool:
    """feedback_log 변경 시 자동 재학습하고 규칙 파일을 갱신한다."""
    if not FEEDBACK_LOG_PATH.exists():
        return False

    log_mtime = FEEDBACK_LOG_PATH.stat().st_mtime
    if LEARNED_RULES_PATH.exists() and not force:
        try:
            existing = json.loads(LEARNED_RULES_PATH.read_text(encoding="utf-8"))
            if float(existing.get("source_feedback_mtime", -1)) == float(log_mtime):
                return False
        except Exception:
            pass

    rows = _read_feedback_rows()
    learned = build_learned_rules(rows)
    learned["source_feedback_mtime"] = log_mtime

    LEARNED_RULES_PATH.parent.mkdir(parents=True, exist_ok=True)
    LEARNED_RULES_PATH.write_text(
        json.dumps(learned, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return True

