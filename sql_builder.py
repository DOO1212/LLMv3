"""Build parameterized SQL queries from parsed search conditions."""

from __future__ import annotations

import re

from config import QUALITY_DEFECT_VALUES, QUALITY_OK_VALUES
from config.search import compress_keyword_tokens


SELECT_INVENTORY_ITEMS = """
    SELECT
        id,
        source_file,
        sheet_name,
        row_index,
        product_name,
        product_code,
        price,
        stock,
        safety_stock,
        inbound_date,
        last_outbound_date,
        warehouse,
        category,
        brand,
        supplier,
        status,
        description,
        quality_status,
        raw_text,
        raw_json
    FROM inventory_items
    WHERE 1 = 1
"""


ALLOWED_OPERATORS = {"<=", ">=", "<", ">", "="}


def build_search_sql(parsed_query):
    sql = SELECT_INVENTORY_ITEMS
    params = []

    keyword = parsed_query.get("keyword")
    if keyword is not None:
        keyword_terms = compress_keyword_tokens(
            [t for t in re.split(r"\s+", str(keyword).strip()) if t]
        )
        if not keyword_terms:
            keyword_terms = [str(keyword).strip()]
        for term in keyword_terms:
            sql += " AND (product_name LIKE ? OR raw_text LIKE ?)"
            like_keyword = f"%{term}%"
            params.extend([like_keyword, like_keyword])

    product_code = parsed_query.get("product_code")
    if product_code is not None:
        sql += " AND product_code LIKE ?"
        params.append(f"%{product_code}%")

    warehouse = parsed_query.get("warehouse")
    if warehouse is not None:
        sql += " AND warehouse LIKE ?"
        params.append(f"%{warehouse}%")

    supplier = parsed_query.get("supplier")
    if supplier is not None:
        sql += " AND supplier LIKE ?"
        params.append(f"%{supplier}%")

    status = parsed_query.get("status")
    if status is not None:
        sql += " AND status LIKE ?"
        params.append(f"%{status}%")

    category = parsed_query.get("category")
    if category is not None:
        sql += " AND category LIKE ?"
        params.append(f"%{category}%")

    price = parsed_query.get("price")
    if price is not None:
        operator = price["operator"]
        if operator not in ALLOWED_OPERATORS:
            raise ValueError(f"지원하지 않는 가격 연산자입니다: {operator}")

        sql += f" AND price {operator} ?"
        params.append(price["value"])

    stock = parsed_query.get("stock")
    if stock is not None:
        operator = stock["operator"]
        if operator not in ALLOWED_OPERATORS:
            raise ValueError(f"지원하지 않는 재고 연산자입니다: {operator}")

        sql += f" AND stock {operator} ?"
        params.append(stock["value"])

    safety_stock = parsed_query.get("safety_stock")
    if safety_stock is not None:
        operator = safety_stock["operator"]
        if operator not in ALLOWED_OPERATORS:
            raise ValueError(f"지원하지 않는 안전재고 연산자입니다: {operator}")

        sql += f" AND safety_stock {operator} ?"
        params.append(safety_stock["value"])

    if parsed_query.get("stock_below_safety"):
        sql += (
            " AND stock IS NOT NULL AND safety_stock IS NOT NULL"
            " AND stock < safety_stock"
        )

    quality_flag = parsed_query.get("quality_flag")
    if quality_flag == "defect":
        placeholders = ",".join("?" * len(QUALITY_DEFECT_VALUES))
        sql += f" AND quality_status IN ({placeholders})"
        params.extend(QUALITY_DEFECT_VALUES)
    elif quality_flag == "good":
        placeholders = ",".join("?" * len(QUALITY_OK_VALUES))
        sql += f" AND quality_status IN ({placeholders})"
        params.extend(QUALITY_OK_VALUES)

    quality_status = parsed_query.get("quality_status")
    if quality_status is not None:
        sql += " AND quality_status LIKE ?"
        params.append(f"%{quality_status}%")

    inbound_date = parsed_query.get("inbound_date")
    if inbound_date is not None:
        sql += " AND inbound_date LIKE ?"
        params.append(f"%{inbound_date}%")

    last_outbound_date = parsed_query.get("last_outbound_date")
    if last_outbound_date is not None:
        sql += " AND last_outbound_date LIKE ?"
        params.append(f"%{last_outbound_date}%")

    sort = parsed_query.get("sort")
    if sort == "price_desc":
        sql += " ORDER BY price IS NULL, price DESC, product_name ASC"
    else:
        sql += " ORDER BY price IS NULL, price ASC, product_name ASC"

    sql += " LIMIT ?"
    params.append(parsed_query.get("limit", 5))

    return sql, params
