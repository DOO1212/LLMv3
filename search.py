"""Inventory search orchestration."""

from __future__ import annotations

import json

from config import SQLITE_PATH, TOP_K
from db import run_sql
from llm_query_parser import parse_query_with_llm
from query_parser import parse_query
from sql_builder import build_search_sql


def row_to_result(row, rank):
    raw_data = json.loads(row["raw_json"])
    record = {
        "source_file": row["source_file"],
        "sheet_name": row["sheet_name"],
        "row_index": row["row_index"],
        "text": row["raw_text"],
        "raw_data": raw_data,
        "structured_data": {
            "product_name": row["product_name"],
            "product_code": row["product_code"],
            "price": row["price"],
            "stock": row["stock"],
            "safety_stock": row["safety_stock"],
            "warehouse": row["warehouse"],
            "category": row["category"],
            "brand": row["brand"],
            "supplier": row["supplier"],
            "status": row["status"],
            "description": row["description"],
            "quality_status": row["quality_status"],
        },
    }

    return {
        "rank": rank,
        "score": 1.0,
        "metadata_index": row["id"],
        "record": record,
    }


def search_products_by_parsed_query(parsed_query, sqlite_path=SQLITE_PATH):
    sql, params = build_search_sql(parsed_query)
    rows = run_sql(sql, params, sqlite_path)

    return {
        "sql": sql,
        "params": params,
        "results": [row_to_result(row, rank) for rank, row in enumerate(rows, start=1)],
    }


def search_products(query, sqlite_path=SQLITE_PATH, top_k=TOP_K):
    parsed_query = parse_query_with_llm(query, limit=top_k)
    if parsed_query is None:
        parsed_query = parse_query(query, limit=top_k)
    return search_products_by_parsed_query(parsed_query, sqlite_path)


def exact_filter_search(query, sqlite_path=SQLITE_PATH, top_k=TOP_K):
    return search_products(query, sqlite_path, top_k)


def print_top_results(results):
    if not results:
        print("\n검색 결과가 없습니다.")
        return

    print("\n검색 결과")
    print("=" * 80)

    for item in results:
        structured = item["record"].get("structured_data", {})
        print(f"\n[{item['rank']}] {structured.get('product_name') or ''}")
        print(f"카테고리: {structured.get('category') or ''}")
        print(f"창고: {structured.get('warehouse') or ''}")
        print(f"가격: {structured.get('price') or ''}")
        print(f"재고: {structured.get('stock') or ''}")

    print("-" * 80)


def main():
    print("SQLite 검색을 시작합니다. 종료하려면 q를 입력하세요.")

    while True:
        query = input("\n질문 입력 >>> ").strip()

        if query.lower() == "q":
            print("검색을 종료합니다.")
            break

        if query == "":
            print("질문이 비어 있습니다.")
            continue

        print_top_results(search_products(query))


if __name__ == "__main__":
    main()
