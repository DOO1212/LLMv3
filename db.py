"""SQLite database helpers."""

from __future__ import annotations

import os
import sqlite3

from config import SQLITE_PATH


def get_connection(sqlite_path=SQLITE_PATH):
    if not os.path.exists(sqlite_path):
        raise FileNotFoundError(f"SQLite DB가 없습니다: {sqlite_path}")

    conn = sqlite3.connect(sqlite_path)
    conn.row_factory = sqlite3.Row
    return conn


def run_sql(sql, params=None, sqlite_path=SQLITE_PATH):
    params = params or []
    conn = get_connection(sqlite_path)

    try:
        return conn.execute(sql, params).fetchall()
    finally:
        conn.close()
