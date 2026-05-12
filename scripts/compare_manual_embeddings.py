#!/usr/bin/env python3
"""매뉴얼 청크에 대해 임베딩 모델별 상위 검색 결과를 비교한다.

사용:
  python3 scripts/compare_manual_embeddings.py
  MANUAL_EMBED_URL=http://127.0.0.1:11434 python3 scripts/compare_manual_embeddings.py --models bge-m3,mxbai-embed-large

DB의 manual_chunk_embeddings 테이블을 수정하지 않고, 메모리에만 벡터를 캐시한다.
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from pathlib import Path

# 프로젝트 루트를 path에 추가
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import SQLITE_PATH  # noqa: E402
from manual_search import (  # noqa: E402
    DEFAULT_EMBED_URL,
    _clean_snippet,
    _cosine_vectors,
    _embed_text_ollama,
    _normalize,
)

# 매뉴얼/청소기 사용 시나리오 위주 질문 20개 (원하면 파일로 교체)
DEFAULT_QUERIES: list[str] = [
    "청소기 설치 방법",
    "청소기 설치하기",
    "필터 청소하는 법",
    "먼지통 비우기",
    "충전 방법",
    "배터리 표시등",
    "전원이 안 켜져요",
    "흡입력이 약해요",
    "소음이 커요",
    "브러시 교체",
    "헤드 회전이 안 돼요",
    "물걸레 사용법",
    "Wi-Fi 연결",
    "앱 페어링",
    "에러 코드",
    "보증 기간",
    "고객센터 전화",
    "부속품 구성",
    "보관 방법",
    "안전 주의사항",
]


def _load_chunks(conn: sqlite3.Connection, limit: int) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT id AS chunk_id, source_file, page_no, chunk_text
        FROM manual_chunks
        LIMIT ?
        """,
        (limit,),
    ).fetchall()


def _rank_for_model(
    query: str,
    rows: list[sqlite3.Row],
    model: str,
    embed_url: str,
    vec_cache: dict[tuple[int, str], list[float]],
    top_k: int,
) -> list[tuple[float, int, int, str]]:
    q_text = _normalize(query)
    q_vec = _embed_text_ollama(q_text, model, embed_url)
    if not q_vec:
        return []

    scored: list[tuple[float, int, int, str]] = []
    for row in rows:
        cid = int(row["chunk_id"])
        key = (cid, model)
        if key not in vec_cache:
            v = _embed_text_ollama(str(row["chunk_text"])[:8000], model, embed_url)
            if not v:
                continue
            vec_cache[key] = v
        c_vec = vec_cache[key]
        if len(c_vec) != len(q_vec):
            continue
        sim = _cosine_vectors(q_vec, c_vec)
        scored.append((sim, cid, int(row["page_no"]), _clean_snippet(row["chunk_text"], max_len=72)))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]


def _top1_overlap(a: list[tuple[float, int, int, str]], b: list[tuple[float, int, int, str]]) -> bool:
    if not a or not b:
        return False
    return a[0][1] == b[0][1]


def _jaccard_topk(ids_a: set[int], ids_b: set[int]) -> float:
    if not ids_a and not ids_b:
        return 1.0
    inter = len(ids_a & ids_b)
    union = len(ids_a | ids_b)
    return inter / union if union else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="임베딩 모델별 매뉴얼 검색 비교")
    parser.add_argument(
        "--models",
        default="bge-m3",
        help="쉼표 구분 모델 이름 (Ollama에 pull 필요). 기본은 bge-m3; Top-1/Jaccard 요약은 모델 2개일 때만 계산.",
    )
    parser.add_argument("--top-k", type=int, default=5, help="모델당 출력 상위 k")
    parser.add_argument("--max-chunks", type=int, default=2000, help="스캔할 최대 청크 수")
    parser.add_argument("--sqlite", default=SQLITE_PATH, help="SQLite 경로")
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    if not models:
        print("--models 에 최소 1개 모델을 지정하세요.", file=sys.stderr)
        sys.exit(1)

    embed_url = os.environ.get("MANUAL_EMBED_URL", DEFAULT_EMBED_URL).strip()

    conn = sqlite3.connect(args.sqlite, timeout=60)
    conn.row_factory = sqlite3.Row
    try:
        rows = _load_chunks(conn, args.max_chunks)
    finally:
        conn.close()

    if not rows:
        print(f"manual_chunks 가 비어 있습니다: {args.sqlite}", file=sys.stderr)
        sys.exit(1)

    print(f"청크 수: {len(rows)} (상한 {args.max_chunks})")
    print(f"Ollama: {embed_url}")
    print(f"모델: {models}")
    print("=" * 80)

    vec_cache: dict[tuple[int, str], list[float]] = {}
    overlap_top1 = 0
    jaccard_sum = 0.0

    for qi, query in enumerate(DEFAULT_QUERIES, start=1):
        print(f"\n[{qi:02d}] 질문: {query}")
        per_model: dict[str, list[tuple[float, int, int, str]]] = {}
        for model in models:
            ranked = _rank_for_model(
                query, rows, model, embed_url, vec_cache, args.top_k
            )
            per_model[model] = ranked
            if not ranked:
                print(f"  [{model}] 임베딩 실패 또는 차원 불일치 (Ollama에 모델 있나 확인)")
                continue
            for rank, (sim, cid, page, snip) in enumerate(ranked, start=1):
                print(f"  [{model}] #{rank} sim={sim:.4f} chunk={cid} p.{page} | {snip}")

        if len(models) == 2 and per_model.get(models[0]) and per_model.get(models[1]):
            a, b = per_model[models[0]], per_model[models[1]]
            if _top1_overlap(a, b):
                overlap_top1 += 1
            sa = {x[1] for x in a}
            sb = {x[1] for x in b}
            jaccard_sum += _jaccard_topk(sa, sb)

        print("-" * 80)

    n = len(DEFAULT_QUERIES)
    if len(models) == 2 and n:
        print("\n요약 (2모델 기준)")
        print(f"  Top-1 청크 동일 비율: {overlap_top1}/{n} ({100.0 * overlap_top1 / n:.1f}%)")
        print(f"  Top-{args.top_k} 청크 집합 평균 Jaccard: {100.0 * jaccard_sum / n:.1f}%")


if __name__ == "__main__":
    main()
