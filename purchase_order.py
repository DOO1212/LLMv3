"""부족 재고 발주서 의도 감지 및 발주 후보 행 조회."""

from __future__ import annotations

import json
import re
from pathlib import Path

from query_parser import extract_product_keyword, parse_query
from runtime_flags import is_model_eval_mode

LOW_STOCK_PO_BASE = """
    SELECT
        id,
        product_name,
        product_code,
        warehouse,
        category,
        supplier,
        stock,
        safety_stock,
        source_file
    FROM inventory_items
    WHERE stock IS NOT NULL
      AND safety_stock IS NOT NULL
      AND stock < safety_stock
      AND (quality_status IS NULL OR TRIM(quality_status) = '')
"""

# 발주·부족 문구 제거 후 품목 키워드 추출용 (긴 패턴 우선)
_PO_NOISE_PATTERNS = (
    r"발주(?:서)?(?:\s*좀)?\s*(?:작성해줘|작성해주세요|작성해|작성|써줘|써주세요|써|정리해줘|정리해주세요|정리해|정리|뽑아줘|뽑아주세요|뽑아|만들어줘|만들어주세요|만들어|작성부탁해|작성부탁|부탁해|부탁)?",
    r"주문서(?:\s*좀)?\s*(?:작성해줘|작성해주세요|작성해|작성|써줘|써주세요|써|정리해줘|정리해주세요|정리해|정리|뽑아줘|뽑아주세요|뽑아|만들어줘|만들어주세요|만들어|작성부탁해|작성부탁|부탁해|부탁)?",
    r"구매\s*요청",
    r"부족한\s*재고",
    r"부족한",
    r"부족\s*재고",
    r"재고가\s*부족",
    r"재고\s*부족",
    r"부족\s*품목",
    r"재고(?:가)?\s*없는\s*것(?:들)?",
    r"재고(?:가)?\s*없(?:는|음)",
    r"품절",
    r"품목(?:들)?\s*목록",
    r"low\s*stock",
    r"stock\s*out",
)

_PO_TAIL_WORDS = (
    "목록",
    "상품들",
    "제품들",
    "품목들",
    "상품",
    "제품",
    "품목",
    "해줘",
    "해주세요",
    "작성",
    "작성해줘",
    "좀",
    "써줘",
    "써주세요",
    "작성해줘",
    "작성해주세요",
    "만들어줘",
    "만들어주세요",
    "써",
    "정리",
    "정리해",
    "정리해줘",
    "정리해주세요",
    "뽑아",
    "뽑아줘",
    "뽑아주세요",
    "만들어",
    "작성해",
    "작성해주세요",
    "작성부탁",
    "작성부탁해",
    "부탁",
    "부탁해",
    "부탁해요",
    "있으면",
    "있음",
    "있을때",
    "있을 때",
    "중에",
    "없는",
    "없는것",
    "없는것들",
    "것",
    "것들",
    "부족한",
    "부족",
)

_PO_GENERIC_KEYWORDS = {
    "상품",
    "제품",
    "품목",
    "상품들",
    "제품들",
    "품목들",
    "인",
    "의",
    "이",
    "가",
    "은",
    "는",
    "만",
}

_LEARNED_RULES_PATH = Path("output/feedback_learned_rules.json")
_LEARNED_RULES_CACHE_MTIME: float | None = None
_LEARNED_RULES_CACHE: dict = {}


def _load_learned_rules() -> dict:
    if is_model_eval_mode():
        return {}
    global _LEARNED_RULES_CACHE_MTIME, _LEARNED_RULES_CACHE
    try:
        stat = _LEARNED_RULES_PATH.stat()
    except FileNotFoundError:
        _LEARNED_RULES_CACHE_MTIME = None
        _LEARNED_RULES_CACHE = {}
        return {}

    mtime = stat.st_mtime
    if _LEARNED_RULES_CACHE_MTIME == mtime:
        return _LEARNED_RULES_CACHE

    try:
        payload = json.loads(_LEARNED_RULES_PATH.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            payload = {}
    except Exception:
        payload = {}
    _LEARNED_RULES_CACHE_MTIME = mtime
    _LEARNED_RULES_CACHE = payload
    return payload


def _normalize_po_keyword(keyword: str | None) -> str | None:
    """발주 품목 키워드의 과도한 접미를 정리한다. (예: 라면들 -> 라면)"""
    if not keyword:
        return None
    kw = " ".join(str(keyword).split()).strip()
    if not kw:
        return None
    # 한국어 조사/복수 접미로 인한 매칭 실패를 줄인다.
    suffixes = (
        "들은",
        "들을",
        "들이",
        "들",
        "은",
        "는",
        "이",
        "가",
        "을",
        "를",
        "의",
        "만",
    )
    changed = True
    while changed:
        changed = False
        for suffix in suffixes:
            if kw.endswith(suffix) and len(kw) > len(suffix):
                kw = kw[: -len(suffix)].strip()
                changed = True
                break
    return kw or None


def wants_purchase_order_for_low_stock(query: str) -> bool:
    """발주·주문서 작성 + 재고 부족(안전재고 대비) 맥락이 함께 있을 때."""
    if not query or not str(query).strip():
        return False

    text = str(query).strip()
    compact = re.sub(r"\s+", "", text)

    purchase = bool(
        re.search(r"발주(서)?|주문서|구매요청|발\s*주", text)
    )
    learned = _load_learned_rules()
    learned_tokens = learned.get("learned_low_stock_tokens") or []
    learned_fragment = "|".join(
        re.escape(str(tok).strip()) for tok in learned_tokens if str(tok).strip()
    )
    base_fragment = (
        r"부족|안전재고|재고부족|미달|재고없|품절|부족한\s*재고|재고가\s*부족|"
        r"재고가?\s*없(?:는|음)|stock\s*out|low\s*stock"
    )
    low_fragment = f"{base_fragment}|{learned_fragment}" if learned_fragment else base_fragment
    low = bool(re.search(low_fragment, compact, re.I))
    # "없는 라면들 발주서"처럼 '재고'가 생략된 표현도 발주 문맥에서는 부족재고 의도로 본다.
    if not is_model_eval_mode() and not low and re.search(r"(?:^|\s)없는(?:\s|$)", text):
        low = True
    return purchase and low


def strip_purchase_order_noise(query: str) -> str:
    text = str(query).strip()
    learned = _load_learned_rules()
    learned_patterns = learned.get("learned_noise_patterns") or []
    for pattern in (*_PO_NOISE_PATTERNS, *learned_patterns):
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)
    text = re.sub(
        r"(?:작성해줘|작성해주세요|작성해|작성|써줘|써주세요|써|정리해줘|정리해주세요|정리해|정리|뽑아줘|뽑아주세요|뽑아|만들어줘|만들어주세요|만들어|작성부탁해|작성부탁|부탁해|부탁|있으면|있음|있을때|있을\s*때)",
        " ",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\s+", " ", text).strip()
    learned_tail_words = learned.get("learned_tail_words") or []
    for w in (*_PO_TAIL_WORDS, *learned_tail_words):
        text = re.sub(rf"(?:^|\s){re.escape(w)}(?=\s|$)", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def po_scope_keyword(query: str) -> str | None:
    """질문에서 품목 검색어 후보 (예: '주방세제' → DB의 '주방 세제'와 매칭되도록 SQL에서 공백 무시)."""
    noise = strip_purchase_order_noise(query)
    if not noise:
        return None
    parts = [p for p in noise.split() if p and not p.endswith("창고")]
    compact = " ".join(parts).strip()
    if not compact:
        return None
    kw = extract_product_keyword(compact)
    if kw and kw.strip():
        kw = _normalize_po_keyword(kw)
        if not kw:
            return None
        if kw in _PO_GENERIC_KEYWORDS:
            return None
        return kw
    compact = _normalize_po_keyword(compact)
    if not compact:
        return None
    if compact in _PO_GENERIC_KEYWORDS:
        return None
    return compact if len(compact) <= 60 else None


def _compact_like_pattern(keyword: str) -> str:
    """공백 제거 후 부분 일치 (주방세제 ↔ 주방 세제)."""
    inner = re.sub(r"\s+", "", keyword)
    return f"%{inner}%" if inner else "%"


def purchase_order_filters(query: str, limit: int) -> dict:
    """parse_query + 발주용 품목 키워드."""
    parsed = parse_query(query, limit=limit)
    scope_kw = po_scope_keyword(query)
    return {
        "keyword": scope_kw,
        "category": parsed.get("category"),
        "warehouse": parsed.get("warehouse"),
        "supplier": parsed.get("supplier"),
        "product_code": parsed.get("product_code"),
    }


def build_low_stock_po_sql(filters: dict, limit: int) -> tuple[str, list]:
    """필터를 반영한 발주용 부족 재고 SQL과 파라미터."""
    sql = LOW_STOCK_PO_BASE
    params: list = []

    kw = filters.get("keyword")
    if kw:
        like = _compact_like_pattern(kw)
        sql += (
            " AND ("
            "REPLACE(IFNULL(product_name, ''), ' ', '') LIKE ?"
            " OR REPLACE(IFNULL(category, ''), ' ', '') LIKE ?"
            " OR REPLACE(IFNULL(product_code, ''), ' ', '') LIKE ?"
            ")"
        )
        params.extend([like, like, like])

    cat = filters.get("category")
    if cat:
        sql += " AND category LIKE ?"
        params.append(f"%{cat}%")

    wh = filters.get("warehouse")
    if wh:
        sql += " AND warehouse LIKE ?"
        params.append(f"%{wh}%")

    sup = filters.get("supplier")
    if sup:
        sql += " AND supplier LIKE ?"
        params.append(f"%{sup}%")

    pc = filters.get("product_code")
    if pc:
        sql += " AND product_code LIKE ?"
        params.append(f"%{pc}%")

    sql += (
        " ORDER BY (safety_stock - stock) DESC, product_name ASC, id ASC"
        "\n    LIMIT ?"
    )
    params.append(int(limit))
    return sql.strip(), params


def fetch_low_stock_order_lines(
    sqlite_path, limit=2000, filters: dict | None = None
) -> tuple[list, str, list]:
    """재고(품질결과 없는 행)만: 실제 재고 부족 후보. (lines, sql, params) 반환."""
    from db import run_sql

    sql, params = build_low_stock_po_sql(filters or {}, limit)
    rows = run_sql(sql, params, sqlite_path)
    lines = []
    for row in rows:
        stock = int(row["stock"] or 0)
        safe = int(row["safety_stock"] or 0)
        gap = max(0, safe - stock)
        lines.append(
            {
                "품목명": row["product_name"] or "",
                "상품코드": row["product_code"] or "",
                "창고": row["warehouse"] or "",
                "카테고리": row["category"] or "",
                "공급업체": row["supplier"] or "",
                "현재고": stock,
                "안전재고": safe,
                "부족수량": gap,
                "발주수량": gap,
            }
        )
    return lines, sql, params
