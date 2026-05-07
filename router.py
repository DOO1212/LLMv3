"""Route user queries between SQLite search and LLM fallback."""

from __future__ import annotations

from config import SQLITE_PATH, TOP_K
from llm_fallback import answer_with_llm
from llm_po_intro import fetch_po_intro_llm
from llm_query_parser import parse_query_with_llm
from purchase_order import (
    fetch_low_stock_order_lines,
    purchase_order_filters,
    wants_purchase_order_for_low_stock,
)
from query_parser import parse_query
from search import search_products_by_parsed_query


def route_query(query, sqlite_path=SQLITE_PATH, top_k=TOP_K, response_locale="ko"):
    if wants_purchase_order_for_low_stock(query):
        cap = min(int(top_k), 5000)
        filters = purchase_order_filters(query, cap)
        lines, sql, params = fetch_low_stock_order_lines(
            sqlite_path, limit=cap, filters=filters
        )
        intro = fetch_po_intro_llm(query, len(lines), locale="ko")
        parsed_po = {
            "intent": "purchase_order_low_stock",
            "row_count": len(lines),
            "filters": {k: v for k, v in filters.items() if v},
        }
        return {
            "route": "purchase_order",
            "parsed_query": parsed_po,
            "purchase_order_lines": lines,
            "po_intro": intro,
            "results": [],
            "sql": sql,
            "sql_params": params,
            "answer": None,
            "error": None,
        }

    llm_parsed_query = parse_query_with_llm(query, limit=top_k)
    used_llm_parser = llm_parsed_query is not None
    parsed_query = llm_parsed_query or parse_query(query, limit=top_k)

    try:
        search_payload = search_products_by_parsed_query(
            parsed_query,
            sqlite_path=sqlite_path,
        )
    except Exception as error:
        return {
            "route": "llm_fallback",
            "parsed_query": parsed_query,
            "results": [],
            "sql": None,
            "sql_params": None,
            "answer": answer_with_llm(
                query, parsed_query, search_error=error, locale=response_locale
            ),
            "error": str(error),
        }

    results = search_payload["results"]
    if results:
        return {
            "route": "llm_sql" if used_llm_parser else "sqlite",
            "parsed_query": parsed_query,
            "results": results,
            "sql": search_payload["sql"],
            "sql_params": search_payload["params"],
            "answer": None,
            "error": None,
        }

    return {
        "route": "llm_fallback",
        "parsed_query": parsed_query,
        "results": [],
        "sql": None,
        "sql_params": None,
        "answer": answer_with_llm(query, parsed_query, locale=response_locale),
        "error": None,
    }
