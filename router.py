"""Route user queries between SQLite search and LLM fallback."""

from __future__ import annotations

import os

from config import SQLITE_PATH, TOP_K
from llm_query_parser import parse_query_with_llm
from manual_search import (
    build_manual_answer,
    parse_manual_query_for_ui,
    search_manual_chunks,
)
from query_parser import parse_query
from search import search_products_by_parsed_query


def _is_inventory_focused_query(original_query: str, excel_parsed: dict) -> bool:
    text = str(original_query or "").strip()
    compact = text.replace(" ", "")
    if not compact:
        return False

    manual_cues = (
        "매뉴얼",
        "설명서",
        "사용법",
        "청소기",
        "흡입",
        "필터",
        "고장",
        "에러",
        "증상",
    )
    if any(cue in compact for cue in manual_cues):
        return False

    inventory_cues = (
        "재고",
        "상품",
        "품목",
        "창고",
        "카테고리",
        "공급업체",
        "단가",
        "가격",
        "패딩",
        "목록",
        "리스트",
    )
    if any(cue in compact for cue in inventory_cues):
        return True

    for key in (
        "keyword",
        "product_code",
        "warehouse",
        "supplier",
        "status",
        "category",
        "price",
        "stock",
        "safety_stock",
        "quality_flag",
        "quality_status",
    ):
        value = excel_parsed.get(key)
        if value not in (None, "", [], {}):
            return True
    return False


def route_query(
    query,
    sqlite_path=SQLITE_PATH,
    top_k=TOP_K,
    response_locale="ko",
    search_query=None,
):
    original_query = str(query or "").strip()
    manual_query = str(search_query or original_query).strip() or original_query
    # 1) 규칙 파서를 우선 적용한다.
    excel_parsed = parse_query(original_query, limit=top_k)
    parsed_bundle = {
        "search_query": manual_query,
        "excel_parse": excel_parsed,
        "pdf_parse": None,
    }

    try:
        base_payload = search_products_by_parsed_query(
            excel_parsed,
            sqlite_path=sqlite_path,
        )
    except Exception as error:
        parsed_bundle["pdf_parse"] = parse_manual_query_for_ui(manual_query)
        return {
            "route": "sqlite",
            "parsed_query": parsed_bundle,
            "results": [],
            "sql": None,
            "sql_params": None,
            "answer": None,
            "error": str(error),
        }

    # 2) 규칙 파서 결과가 비어 있으면, 재고성 질의는 SQL(LLM 파서) 재시도를 먼저 한다.
    if not base_payload["results"]:
        inventory_focused = _is_inventory_focused_query(original_query, excel_parsed)

        def _try_llm_sql():
            llm_parsed = parse_query_with_llm(original_query, limit=top_k)
            if llm_parsed is None:
                return None
            try:
                llm_payload = search_products_by_parsed_query(
                    llm_parsed,
                    sqlite_path=sqlite_path,
                )
            except Exception:
                return None
            if llm_payload and llm_payload["results"]:
                return {
                    "route": "llm_sql",
                    "parsed_query": {
                        **parsed_bundle,
                        "excel_parse_llm": llm_parsed,
                    },
                    "results": llm_payload["results"],
                    "sql": llm_payload["sql"],
                    "sql_params": llm_payload["params"],
                    "answer": None,
                    "error": None,
                }
            return None

        if inventory_focused:
            llm_first = _try_llm_sql()
            if llm_first is not None:
                return llm_first

        # 3) manual 검색: 1) 토큰 SQL → 2) LLM 의도 SQL → 3) 벡터 (`search_manual_chunks` 내부).
        parsed_bundle["pdf_parse"] = parse_manual_query_for_ui(manual_query)
        manual_top_k = max(10, min(100, int(os.environ.get("MANUAL_TOP_K", "50"))))
        manual_hits = search_manual_chunks(manual_query, top_k=manual_top_k)
        if manual_hits:
            return {
                "route": "manual_pdf",
                "parsed_query": parsed_bundle,
                "results": [],
                "manual_results": manual_hits,
                "answer": build_manual_answer(
                    query=query,
                    manual_hits=manual_hits,
                    response_locale=response_locale,
                ),
                "sql": None,
                "sql_params": None,
                "error": None,
            }

        # 4) manual에서도 못 찾으면(또는 manual 우선 질의면) LLM 파서를 보조로 사용한다.
        if not inventory_focused:
            llm_after_manual = _try_llm_sql()
            if llm_after_manual is not None:
                return llm_after_manual

    if base_payload["results"]:
        parsed_bundle["pdf_parse"] = parse_manual_query_for_ui(manual_query)
    return {
        "route": "sqlite",
        "parsed_query": parsed_bundle,
        "results": base_payload["results"],
        "sql": base_payload["sql"],
        "sql_params": base_payload["params"],
        "answer": None,
        "error": None,
    }
