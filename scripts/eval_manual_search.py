"""Evaluate manual search quality with a fixed Korean query set."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from manual_search import parse_manual_query, search_manual_chunks  # noqa: E402


DEFAULT_OUT_PATH = PROJECT_ROOT / "output" / "manual_search_eval_report.json"

EVAL_QUERIES: list[str] = [
    "청소기 청소법좀 알려줘",
    "청소기 소리가 커졌어",
    "청소기에 솔의 용도가 뭐야?",
    "회전솔 청소 방법 알려줘",
    "배기 필터 청소 어떻게 해?",
    "프리 필터는 어떻게 관리해?",
    "흡입구 청소 순서 알려줘",
    "먼지통 비우는 방법",
    "큰 먼지 분리 장치 세척법",
    "필터 말리는 시간은 얼마나 돼?",
    "배터리 충전이 안돼",
    "배터리 완충이 안 되는 이유",
    "제품 사용 중 전원이 꺼져",
    "흡입력이 약해졌어",
    "제품에서 냄새가 나",
    "소음이 심할 때 해결법",
    "회전솔이 회전하지 않아",
    "파워드라이브 흡입구 사용법",
    "틈새 흡입구는 언제 써?",
    "2 in 1 흡입구 사용 방법",
    "솔형 흡입구 용도",
    "액자 표면 청소할 때 어떤 흡입구 써?",
    "가구 표면 청소는 어떻게 해?",
    "마루 흡입구 청소하기",
    "모터 부분 먼지 청소",
    "제품 본체 물세척해도 돼?",
    "배터리 보관 방법",
    "고장 신고 전 확인할 사항",
    "제품이 작동하지 않을 때",
    "필터 청소 주기는 어떻게 돼?",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate manual search quality")
    parser.add_argument("--out-file", default=str(DEFAULT_OUT_PATH), help="output report JSON")
    parser.add_argument("--top-k", type=int, default=20, help="manual search top-k")
    parser.add_argument(
        "--preview-len",
        type=int,
        default=120,
        help="snippet preview length in report",
    )
    return parser.parse_args()


def _clean_preview(text: str, limit: int) -> str:
    normalized = " ".join(str(text or "").split())
    return normalized[:limit]


def main() -> None:
    args = parse_args()
    out_path = Path(args.out_file).expanduser().resolve()
    top_k = max(1, int(args.top_k))
    preview_len = max(40, int(args.preview_len))

    per_query: list[dict] = []
    page_histogram: dict[str, int] = {}
    zero_result_queries: list[str] = []

    for query in EVAL_QUERIES:
        parsed = parse_manual_query(query)
        results = search_manual_chunks(query, top_k=top_k, manual_parsed=parsed)
        if not results:
            zero_result_queries.append(query)

        top_results = []
        for item in results[:5]:
            page_no = item.get("page_no")
            page_key = str(page_no)
            page_histogram[page_key] = page_histogram.get(page_key, 0) + 1
            top_results.append(
                {
                    "page_no": page_no,
                    "score": item.get("score"),
                    "snippet_preview": _clean_preview(item.get("snippet", ""), preview_len),
                }
            )

        per_query.append(
            {
                "query": query,
                "parsed": {
                    "raw_terms": parsed.get("raw_terms", []),
                    "intent_query": parsed.get("intent_query"),
                    "search_terms": parsed.get("search_terms", []),
                    "anchor_terms": parsed.get("anchor_terms", []),
                },
                "result_count": len(results),
                "top_results": top_results,
            }
        )

    report = {
        "query_count": len(EVAL_QUERIES),
        "top_k": top_k,
        "zero_result_count": len(zero_result_queries),
        "zero_result_queries": zero_result_queries,
        "top_result_page_histogram": dict(
            sorted(page_histogram.items(), key=lambda x: (-x[1], int(x[0]) if x[0].isdigit() else x[0]))
        ),
        "queries": per_query,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fp:
        fp.write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    print(
        "[done] "
        f"queries={report['query_count']} "
        f"zero_results={report['zero_result_count']} "
        f"out={out_path}"
    )


if __name__ == "__main__":
    main()

