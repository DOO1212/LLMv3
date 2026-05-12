#!/usr/bin/env python3
"""manual_chunks 각 행에 대해 로케일별 번역을 SQLite `manual_chunk_translations`에 저장한다.

실시간 번역 대신 서빙할 때는 DB 조회만 하면 되므로, 이 스크립트는 시간이 오래 걸려도 한 번 돌려 두면 된다.

  TRANSLATION_MODEL=llama3.1:8b python scripts/pregenerate_manual_translations.py
  # PDF에서 추출한 청크만 (source_file 이 .pdf 로 끝남):
  TRANSLATION_MODEL=llama3.1:8b python scripts/pregenerate_manual_translations.py --pdf-only --force

미리 번역이 없을 때만 즉시 번역하려면 MANUAL_TRANSLATION_LIVE_FALLBACK=true (기본).
미리 번역만 쓰려면 MANUAL_TRANSLATION_LIVE_FALLBACK=false.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import SQLITE_PATH
from manual_search import (
    MANUAL_TRANSLATION_LOCALES,
    _ensure_manual_translation_table,
    _translate_manual_snippet_live,
    upsert_manual_translation,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="매뉴얼 청크 로케일별 미리 번역")
    parser.add_argument(
        "--locales",
        default=",".join(MANUAL_TRANSLATION_LOCALES),
        help="쉼표 구분 로케일",
    )
    parser.add_argument("--force", action="store_true", help="이미 있어도 덮어쓰기")
    parser.add_argument("--limit", type=int, default=0, help="테스트용 최대 청크 행 수")
    parser.add_argument("--start-id", type=int, default=0, help="manual_chunks.id 가 이 값 이상만")
    parser.add_argument(
        "--pdf-only",
        action="store_true",
        help="source_file 이 .pdf 로 끝나는 행만 (매뉴얼 PDF 청크 전용)",
    )
    args = parser.parse_args()

    locales = [x.strip().lower() for x in args.locales.split(",") if x.strip()]
    locales = [x for x in locales if x != "ko"]
    if not locales:
        print("로케일이 비었습니다.", file=sys.stderr)
        sys.exit(1)

    conn = sqlite3.connect(SQLITE_PATH, timeout=120)
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        _ensure_manual_translation_table(conn)
        conn.commit()
        where = "id >= ?"
        params: list = [max(0, args.start_id)]
        if args.pdf_only:
            where += " AND LOWER(COALESCE(source_file, '')) LIKE ?"
            params.append("%.pdf")
        rows = list(
            conn.execute(
                f"SELECT id, chunk_text FROM manual_chunks WHERE {where} ORDER BY id",
                params,
            )
        )
        if args.limit > 0:
            rows = rows[: args.limit]

        total_pairs = len(rows) * len(locales)
        done = 0
        t0 = time.perf_counter()
        print(
            f"시작: 청크 {len(rows)}개, 로케일 {locales}, 총 {total_pairs}건, force={args.force}",
            flush=True,
        )
        for chunk_id, chunk_text in rows:
            text = str(chunk_text or "").strip()
            if not text:
                done += len(locales)
                continue
            for loc in locales:
                if not args.force:
                    ex = conn.execute(
                        "SELECT 1 FROM manual_chunk_translations WHERE chunk_id = ? AND locale = ?",
                        (chunk_id, loc),
                    ).fetchone()
                    if ex:
                        done += 1
                        continue
                tr = _translate_manual_snippet_live(text, loc)
                if tr:
                    upsert_manual_translation(conn, chunk_id, loc, tr)
                    conn.commit()
                done += 1
                if done % 20 == 0 or done == total_pairs:
                    print(
                        f"[{done}/{total_pairs}] last chunk_id={chunk_id} locale={loc} "
                        f"elapsed={time.perf_counter() - t0:.1f}s",
                        flush=True,
                    )

        print(f"완료: {done}/{total_pairs} 작업, 총 {time.perf_counter() - t0:.1f}s", flush=True)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
