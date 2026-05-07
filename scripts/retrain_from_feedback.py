#!/usr/bin/env python3
"""feedback_log 기반 파싱 보정 규칙 리포트/학습 파일 생성기."""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from feedback_learning import (
    FEEDBACK_LOG_PATH,
    LEARNED_RULES_PATH,
    _read_feedback_rows,
    build_learned_rules,
)
REPORT_PATH = PROJECT_ROOT / "output" / "feedback_retrain_report.json"

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


def _tokenize(text: str) -> list[str]:
    return [t for t in re.split(r"\s+", text.strip()) if t]


def build_report(rows: list[dict]) -> dict:
    dislikes = [r for r in rows if r.get("feedback") == "dislike"]
    by_route = Counter(str(r.get("route") or "unknown") for r in dislikes)

    bad_keywords: Counter[str] = Counter()
    bad_tokens: Counter[str] = Counter()
    no_stock_questions = 0
    no_stock_dislikes = 0

    for row in dislikes:
        q = str(row.get("question") or "")
        compact_q = re.sub(r"\s+", "", q)
        if "재고없" in compact_q or "재고없는" in compact_q:
            no_stock_dislikes += 1

        parsed = row.get("parsed_query") or {}
        kw = parsed.get("keyword")
        if isinstance(kw, str) and kw.strip():
            keyword = kw.strip()
            bad_keywords[keyword] += 1
            for tok in _tokenize(keyword):
                if tok in COMMAND_TOKENS:
                    bad_tokens[tok] += 1

    for row in rows:
        q = str(row.get("question") or "")
        compact_q = re.sub(r"\s+", "", q)
        if "재고없" in compact_q or "재고없는" in compact_q:
            no_stock_questions += 1

    return {
        "total_feedback": len(rows),
        "dislike_count": len(dislikes),
        "dislike_by_route": dict(by_route),
        "top_bad_keywords": bad_keywords.most_common(10),
        "suggested_noise_tokens": bad_tokens.most_common(10),
        "no_stock_question_count": no_stock_questions,
        "no_stock_dislike_count": no_stock_dislikes,
        "recommendations": [
            "재고 없는/품절 표현을 발주 의도(low stock) 감지 패턴에 포함",
            "keyword 후처리에서 명령어/조사/불용어 토큰 제거 강화",
            "feedback 로그를 주기적으로 집계해 상위 실패 패턴을 룰에 반영",
        ],
    }


def main() -> None:
    rows = _read_feedback_rows()
    report = build_report(rows)
    learned = build_learned_rules(rows)
    if FEEDBACK_LOG_PATH.exists():
        learned["source_feedback_mtime"] = FEEDBACK_LOG_PATH.stat().st_mtime
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    LEARNED_RULES_PATH.write_text(
        json.dumps(learned, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"saved: {REPORT_PATH}")
    print(f"saved: {LEARNED_RULES_PATH}")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

