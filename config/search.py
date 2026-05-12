"""SQLite search settings."""

from __future__ import annotations

from typing import Final

TOP_K: Final[int] = 5


def compress_keyword_tokens(tokens: list[str]) -> list[str]:
    """공백 분리 토큰이 많을 때 짧은 보조어를 줄여 AND 과매칭을 완화한다.

    고정 단어 목록 없이, **토큰 글자 수 분산**만 본다.
    가장 짧은 길이의 토큰이 있고 그 길이가 최장 토큰보다 짧으면,
    문장 끝에서 가까운 쪽의 짧은 토큰부터 하나씩 제거한다.
    (예: 청소기·설치·방법 → 설치하기 없는 문서도 걸리도록 마지막 짧은 토큰 제거)
    """
    out = [t for t in tokens if t]
    if len(out) <= 2:
        return out
    while len(out) >= 3:
        lengths = [len(t) for t in out]
        max_len, min_len = max(lengths), min(lengths)
        if max_len <= min_len:
            break
        drop_idx: int | None = None
        for i in range(len(out) - 1, -1, -1):
            if len(out[i]) == min_len:
                drop_idx = i
                break
        if drop_idx is None:
            break
        out = out[:drop_idx] + out[drop_idx + 1 :]
    return out


KEYWORD_STRIP_WORDS: Final[tuple[str, ...]] = (
    "만원",
    "천원",
    "억원",
    "단가",
    "가격",
    "원",
    "이하",
    "이상",
    "미만",
    "초과",
    "보다싼",
    "보다비싼",
    "밑",
    "알려줘",
    "알려주세요",
    "찾아줘",
    "찾아주세요",
    "목록",
    "리스트",
    "list",
    "보여줘",
    "보여주세요",
    "추천해줘",
    "추천해주세요",
    "검색해줘",
    "검색해주세요",
    "종류",
    "뭐 있어",
    "뭐있어",
    "뭐가 있어",
    "뭐가있어",
    "있어?",
    "있어",
    "있나요",
    "있나",
    "어떤",
    "공정",
    "라인",
    "현재",
    "데이터",
    "있는",
    "인한",
)
