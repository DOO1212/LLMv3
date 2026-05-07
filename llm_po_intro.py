"""발주서 화면 상단 안내 문구 — Ollama 짧은 응답 (실패 시 None)."""

from __future__ import annotations

import os

from llm_fallback import generate_text
from llm_fallback import LOCALE_RESPONSE_LANG


def _po_timeout():
    try:
        return int(os.environ.get("PO_INTRO_TIMEOUT", "60"))
    except (TypeError, ValueError):
        return 60


def fetch_po_intro_llm(user_query: str, item_count: int, locale: str = "ko") -> str | None:
    lang = LOCALE_RESPONSE_LANG.get(locale, LOCALE_RESPONSE_LANG["ko"])

    prompt = f"""
너는 재고·발주 업무 도우미다.
사용자가 부족 재고 품목에 대한 발주서 작성을 요청했다.
아래 조건으로 **짧게 2~4문장**만 답하라. 목록·표·JSON은 쓰지 마라.
- 반드시 {lang}로만 작성.
- 발주 수량 열은 사용자가 직접 수정한다고 안내.
- 품목 수: {item_count}건.

사용자 질문:
{user_query}
""".strip()

    text = generate_text(
        prompt,
        max_new_tokens=min(_po_timeout() * 8, 256),
        temperature=0.6,
    )
    return text or None
