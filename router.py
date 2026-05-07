"""Route user queries between SQLite search and LLM fallback."""

from __future__ import annotations

from config import SQLITE_PATH, TOP_K
from llm_fallback import answer_with_llm
from llm_query_parser import parse_query_with_llm
from query_parser import parse_query
from search import search_products_by_parsed_query


def route_query(query, sqlite_path=SQLITE_PATH, top_k=TOP_K, response_locale="ko"):
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
