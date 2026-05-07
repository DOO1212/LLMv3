"""질문 재해석 후보 생성기 (모호/무응답 시 사용자 확인용)."""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request

from llm_fallback import DEFAULT_LLAMA_MODEL, DEFAULT_OLLAMA_URL, get_ollama_timeout

DEFAULT_REINTERPRET_MODEL = os.environ.get(
    "REINTERPRET_LLAMA_MODEL",
    DEFAULT_LLAMA_MODEL,
)


def _extract_json_array(text: str) -> list[str] | None:
    text = str(text or "").strip()
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(data, list):
        return None
    items = [str(item).strip() for item in data if str(item).strip()]
    return items[:3] if items else None


def _fallback_candidates(keyword: str | None, query: str) -> list[str]:
    kw = _sanitize_candidate(keyword or "")
    if kw:
        return [
            f"{kw} 부족 품목 발주서 작성해줘",
            f"{kw} 안전재고 미달 품목 발주서 작성해줘",
            f"{kw} 발주 필요 수량 정리해줘",
        ]
    return [
        "부족 재고 품목 발주서 작성해줘",
        _sanitize_candidate(query) or "재고 부족 품목 발주서 작성해줘",
    ]


def _sanitize_candidate(text: str) -> str:
    cleaned = " ".join(str(text).split()).strip()
    if not cleaned:
        return ""
    tokens = cleaned.split()
    compact: list[str] = []
    prev = None
    repeat = 0
    for tok in tokens:
        if tok == prev:
            repeat += 1
            if repeat >= 2:
                continue
        else:
            repeat = 0
        compact.append(tok)
        prev = tok
    cleaned = " ".join(compact)
    cleaned = re.sub(r"(?:\s*관련){2,}", " 관련", cleaned).strip()
    if len(cleaned) > 45:
        return ""
    return cleaned


def _focus_keyword(keyword: str | None) -> str | None:
    if not keyword:
        return None
    cleaned = " ".join(str(keyword).split()).strip()
    if not cleaned:
        return None
    cleaned = re.sub(r"\b(?:부족|수량|재고|발주|품목|제품|상품)\b", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned or None


def _postprocess_candidates(candidates: list[str], keyword: str | None, query: str) -> list[str]:
    seen: set[str] = set()
    cleaned: list[str] = []
    for item in candidates:
        candidate = _sanitize_candidate(item)
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        cleaned.append(candidate)
        if len(cleaned) >= 3:
            break
    if cleaned:
        return cleaned
    return _fallback_candidates(keyword, query)


def propose_reinterpret_queries(query: str, keyword: str | None = None) -> list[str]:
    focus_kw = _focus_keyword(keyword)
    prompt = f"""
너는 재고/발주 질문 재해석 도우미다.
아래 사용자 질문이 의도 파악 실패 가능성이 있을 때, 다시 시도할 질문 문장 2~3개를 제안하라.
출력은 반드시 JSON 배열(string[])만 반환하라. 다른 설명 문장 금지.

제약:
- 한국어로 작성
- 짧고 명확한 명령형 문장
- 재고 부족 발주 의도에 맞게
- 과도한 조건/수식어 제거
- 키워드 재해석 중심: 아래 "집중 키워드"를 최우선으로 해석
- "집중 키워드"가 있으면 원문 전체보다 키워드 정제에 집중
- "관련" 같은 군더더기 단어 사용 금지
- 첫 번째 후보는 가장 보수적이고 안전한 질문으로 작성

원문 질문: {query}
실패한 키워드(있으면 참고): {keyword or "없음"}
집중 키워드: {focus_kw or "없음"}
""".strip()
    payload = {
        "model": DEFAULT_REINTERPRET_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.45,
            "top_p": 0.85,
            "repeat_penalty": 1.15,
        },
    }
    request = urllib.request.Request(
        os.environ.get("OLLAMA_URL", DEFAULT_OLLAMA_URL),
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=get_ollama_timeout()) as response:
            body = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError):
        return _postprocess_candidates(_fallback_candidates(focus_kw or keyword, query), keyword, query)

    candidates = _extract_json_array(body.get("response", ""))
    if candidates:
        return _postprocess_candidates(candidates, keyword, query)
    return _postprocess_candidates(_fallback_candidates(focus_kw or keyword, query), keyword, query)
