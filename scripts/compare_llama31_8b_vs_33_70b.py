#!/usr/bin/env python3
"""동일 프롬프트로 두 Ollama 모델 응답·지연을 비교한다 (Ollama /api/generate).

기본값은 둘 다 llama3.1:8b. 다른 모델과 비교할 때는 --small / --large 로 지정한다.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434/api/generate")
TIMEOUT = int(os.environ.get("OLLAMA_TIMEOUT", "600"))

# 매뉴얼 검색 2단계와 동일한 스타일 (manual_search._llm_intent_query_llm_only)
QUERIES: list[str] = [
    "청소기 보관 방법",
    "배터리 충전이 안돼",
    "필터 청소하는 법",
    "흡입력이 약해졌어",
    "먼지통 비우는 방법",
    "회전솔이 회전하지 않아",
    "Wi-Fi 연결",
    "제품 본체 물세척해도 돼?",
    "소음이 심할 때 해결법",
    "배터리 보관 방법",
]


def _intent_prompt(user_query: str) -> str:
    lang_rule = (
        "- intent_query 언어는 사용자 질문과 같게: 한국어 질문이면 **한국어 검색어**만 사용한다. "
        "영어 번역·영문 요약을 넣지 마라.\n"
    )
    return (
        "사용자 매뉴얼 질문을 검색용 짧은 문장으로 정리해 JSON으로 출력하라. 설명 금지.\n"
        f"{lang_rule}"
        '출력 형식: {"intent_query":"..."}\n'
        f"질문: {user_query}"
    )


def _ollama_generate(url: str, model: str, prompt: str, num_predict: int) -> tuple[str | None, float]:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": num_predict},
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
        return None, time.perf_counter() - t0
    elapsed = time.perf_counter() - t0
    text = (data.get("response") or "").strip()
    return text or None, elapsed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--small", default="llama3.1:8b")
    parser.add_argument("--large", default="llama3.1:8b")
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--num-predict", type=int, default=96)
    args = parser.parse_args()

    print(f"Ollama: {args.url}")
    print(f"모델 A(작음): {args.small}  |  모델 B(큼): {args.large}")
    print(f"질문 수: {len(QUERIES)}, num_predict={args.num_predict}")
    print("=" * 100)

    for i, q in enumerate(QUERIES, 1):
        prompt = _intent_prompt(q)
        print(f"\n[{i:02d}] 질문: {q}")

        r_small, t_small = _ollama_generate(args.url, args.small, prompt, args.num_predict)
        r_large, t_large = _ollama_generate(args.url, args.large, prompt, args.num_predict)

        print(f"  [{args.small}] {t_small:.2f}s")
        print(f"    {r_small or '(실패/빈 응답)'}")
        print(f"  [{args.large}] {t_large:.2f}s")
        print(f"    {r_large or '(실패/빈 응답)'}")
        print("-" * 100)


if __name__ == "__main__":
    main()
