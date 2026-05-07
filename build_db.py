"""Build the SQLite inventory database from Excel files."""

from __future__ import annotations

import glob
import json
import os
import re
import sqlite3

import pandas as pd

from config import DATA_DIR, SQLITE_PATH, TARGET_FIELDS
from mapping_state import get_merged_column_synonyms


def normalize_text(text):
    text = str(text).strip().lower()
    return re.sub(r"[\s_\-]+", "", text)


def map_column_name(col_name):
    norm_col = normalize_text(col_name)
    column_synonyms = get_merged_column_synonyms()
    is_identifier_col = (
        "코드" in norm_col
        or norm_col.endswith("id")
        or norm_col.endswith("아이디")
    )

    for target_field, synonym_list in column_synonyms.items():
        for synonym in synonym_list:
            if norm_col == normalize_text(synonym):
                return target_field

    for target_field, synonym_list in column_synonyms.items():
        if is_identifier_col and target_field in {"product_name", "stock"}:
            continue

        if "재고금액" in norm_col and target_field in {"price", "stock"}:
            continue

        if "안전재고" in norm_col and target_field == "stock":
            continue

        for synonym in synonym_list:
            if normalize_text(synonym) in norm_col:
                return target_field

    return None


def clean_value(value):
    if pd.isna(value):
        return None

    if hasattr(value, "isoformat"):
        return value.isoformat()

    text = str(value).strip()
    if text == "":
        return None

    return text


def parse_int(value):
    value = clean_value(value)
    if value is None:
        return None

    cleaned = re.sub(r"[^0-9.\-]", "", str(value).replace(",", ""))
    if cleaned in ("", "-", ".", "-."):
        return None

    try:
        return int(float(cleaned))
    except ValueError:
        return None


def first_mapped_value(row, mapped_columns, field):
    for original_col, mapped_field in mapped_columns.items():
        if mapped_field != field:
            continue

        value = clean_value(row.get(original_col, None))
        if value is not None:
            return value

    return None


def row_to_text(row, mapped_columns):
    parts = []

    for field in TARGET_FIELDS:
        for original_col, mapped_field in mapped_columns.items():
            if mapped_field != field:
                continue

            value = clean_value(row.get(original_col, None))
            if value is not None:
                parts.append(f"{field}: {value}")

    for original_col in row.index:
        if original_col in mapped_columns:
            continue

        value = clean_value(row.get(original_col, None))
        if value is not None:
            parts.append(f"{original_col}: {value}")

    return " / ".join(parts)


def row_to_structured(row, mapped_columns):
    return {
        "product_name": first_mapped_value(row, mapped_columns, "product_name"),
        "product_code": first_mapped_value(row, mapped_columns, "product_code"),
        "price": parse_int(first_mapped_value(row, mapped_columns, "price")),
        "stock": parse_int(first_mapped_value(row, mapped_columns, "stock")),
        "safety_stock": parse_int(first_mapped_value(row, mapped_columns, "safety_stock")),
        "inbound_date": first_mapped_value(row, mapped_columns, "inbound_date"),
        "last_outbound_date": first_mapped_value(
            row, mapped_columns, "last_outbound_date"
        ),
        "warehouse": first_mapped_value(row, mapped_columns, "warehouse"),
        "category": first_mapped_value(row, mapped_columns, "category"),
        "brand": first_mapped_value(row, mapped_columns, "brand"),
        "supplier": first_mapped_value(row, mapped_columns, "supplier"),
        "status": first_mapped_value(row, mapped_columns, "status"),
        "description": first_mapped_value(row, mapped_columns, "description"),
        "quality_status": first_mapped_value(row, mapped_columns, "quality_status"),
    }


def row_to_raw_data(row):
    raw_data = {}

    for col in row.index:
        raw_data[str(col)] = clean_value(row[col])

    return raw_data


def load_all_excel_rows(data_dir):
    records = []
    excel_files = glob.glob(os.path.join(data_dir, "*.xlsx"))
    excel_files += glob.glob(os.path.join(data_dir, "*.xls"))

    print(f"발견한 엑셀 파일 수: {len(excel_files)}")

    for file_path in excel_files:
        print(f"\n처리 중 파일: {file_path}")
        xls = pd.ExcelFile(file_path)

        for sheet_name in xls.sheet_names:
            print(f"  - 시트 처리 중: {sheet_name}")
            df = pd.read_excel(file_path, sheet_name=sheet_name)

            if df.empty:
                print("    -> 빈 시트라서 건너뜀")
                continue

            mapped_columns = {}
            for col in df.columns:
                mapped = map_column_name(col)
                if mapped is not None:
                    mapped_columns[col] = mapped

            print(f"    -> 자동 매핑 결과: {mapped_columns}")

            for row_idx, row in df.iterrows():
                text = row_to_text(row, mapped_columns)
                if text.strip() == "":
                    continue

                records.append(
                    {
                        "source_file": os.path.basename(file_path),
                        "sheet_name": sheet_name,
                        "row_index": int(row_idx),
                        "raw_text": text,
                        "raw_json": row_to_raw_data(row),
                        "structured_data": row_to_structured(row, mapped_columns),
                    }
                )

    return records


def save_records_to_sqlite(records, sqlite_path):
    conn = sqlite3.connect(sqlite_path)

    try:
        conn.execute("DROP TABLE IF EXISTS inventory_items")
        conn.execute(
            """
            CREATE TABLE inventory_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_file TEXT NOT NULL,
                sheet_name TEXT NOT NULL,
                row_index INTEGER NOT NULL,
                product_name TEXT,
                product_code TEXT,
                price INTEGER,
                stock INTEGER,
                safety_stock INTEGER,
                inbound_date TEXT,
                last_outbound_date TEXT,
                warehouse TEXT,
                category TEXT,
                brand TEXT,
                supplier TEXT,
                status TEXT,
                description TEXT,
                quality_status TEXT,
                raw_text TEXT NOT NULL,
                raw_json TEXT NOT NULL
            )
            """
        )

        rows = []
        for record in records:
            structured = record["structured_data"]
            rows.append(
                (
                    record["source_file"],
                    record["sheet_name"],
                    record["row_index"],
                    structured.get("product_name"),
                    structured.get("product_code"),
                    structured.get("price"),
                    structured.get("stock"),
                    structured.get("safety_stock"),
                    structured.get("inbound_date"),
                    structured.get("last_outbound_date"),
                    structured.get("warehouse"),
                    structured.get("category"),
                    structured.get("brand"),
                    structured.get("supplier"),
                    structured.get("status"),
                    structured.get("description"),
                    structured.get("quality_status"),
                    record["raw_text"],
                    json.dumps(record["raw_json"], ensure_ascii=False),
                )
            )

        conn.executemany(
            """
            INSERT INTO inventory_items (
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
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

        conn.execute("CREATE INDEX idx_inventory_product_name ON inventory_items(product_name)")
        conn.execute("CREATE INDEX idx_inventory_price ON inventory_items(price)")
        conn.execute("CREATE INDEX idx_inventory_stock ON inventory_items(stock)")
        conn.execute("CREATE INDEX idx_inventory_category ON inventory_items(category)")
        conn.commit()
    finally:
        conn.close()


def main():
    records = load_all_excel_rows(DATA_DIR)

    if not records:
        print("처리할 엑셀 데이터가 없습니다.")
        return

    save_records_to_sqlite(records, SQLITE_PATH)

    print(f"\nSQLite 저장 완료: {SQLITE_PATH}")
    print(f"저장 row 수: {len(records)}")


if __name__ == "__main__":
    main()
