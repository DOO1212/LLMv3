"""Use a local Llama model to parse difficult queries into SQL-safe filters."""

from __future__ import annotations

import json
import re

from config import KEYWORD_STRIP_WORDS, TOP_K
from llm_fallback import generate_text
from runtime_flags import is_model_eval_mode


ALLOWED_OPERATORS = {"<=", ">=", "<", ">", "="}
ALLOWED_SORTS = {"price_asc", "price_desc"}


def build_parser_prompt(query, limit=TOP_K):
    if is_model_eval_mode():
        return f"""
너는 query parser다.
사용자 질문을 아래 JSON 스키마로 변환하라.
설명 없이 JSON만 출력하라.

스키마:
{{
  "keyword": string|null,
  "product_code": string|null,
  "warehouse": string|null,
  "supplier": string|null,
  "status": string|null,
  "category": string|null,
  "price": {{"operator": "<=|>=|<|>|=", "value": number}}|null,
  "stock": {{"operator": "<=|>=|<|>|=", "value": number}}|null,
  "safety_stock": {{"operator": "<=|>=|<|>|=", "value": number}}|null,
  "stock_below_safety": boolean|null,
  "quality_flag": "defect"|"good"|null,
  "quality_status": string|null,
  "inbound_date": string|null,
  "last_outbound_date": string|null,
  "requested_fields": string[],
  "sort": "price_asc"|"price_desc",
  "limit": number
}}

사용자 질문:
{query}
""".strip()

    return f"""
너는 재고 검색용 query parser다.
사용자 질문을 SQLite 검색 조건 JSON으로 변환하라.
반드시 JSON만 출력하고 설명 문장은 쓰지 마라.

출력 스키마:
{{
  "keyword": string|null,
  "product_code": string|null,
  "warehouse": string|null,
  "supplier": string|null,
  "status": string|null,
  "category": string|null,
  "price": {{"operator": "<=|>=|<|>|=", "value": number}}|null,
  "stock": {{"operator": "<=|>=|<|>|=", "value": number}}|null,
  "safety_stock": {{"operator": "<=|>=|<|>|=", "value": number}}|null,
  "stock_below_safety": boolean|null,
  "quality_flag": "defect"|"good"|null,
  "quality_status": string|null,
  "inbound_date": string|null,
  "last_outbound_date": string|null,
  "requested_fields": string[],
  "sort": "price_asc"|"price_desc",
  "limit": number
}}

규칙:
- 상품명/품목명은 keyword에 넣어라. 예: "충전기", "패딩", "릴레이"
- 상품코드는 product_code에 넣어라.
- 창고 조건은 warehouse에 넣어라. 예: "대전 중부 물류허브"
- 공급업체 조건은 supplier에 넣어라.
- 상태 조건은 status에 넣어라. (재고: 정상/부족/품절, 생산라인: 생산중/대기/점검중 등 데이터 값 그대로)
- 생산 실시간 데이터에는 **품질 판정(quality_status)만** 있다. 허용 값: 불량·NG(불량군), 양호·OK·정상(양호군). 불량건수·불량율·원인 컬럼은 **없다**.
- "불량이냐/불량만/품질 불량" 같은 질문은 quality_flag를 "defect"로. "양호만/불량 아닌"은 quality_flag "good". 구체 값이 있으면 quality_status에 넣어라(예: "불량").
- 카테고리 조건은 category에 넣어라. 예: "생활용품", "전자", "의류", "식품", "PCB", "조립"
- "전자 제품", "전자기기"처럼 카테고리만 가리키면 category에 "전자" 등으로 넣고 keyword는 비워라.
- "알려줘", "찾아줘", "보여줘", "추천해줘", "검색해줘" 같은 요청어는 keyword에서 제외하라.
- 가격/단가 조건은 price에 넣어라.
- 재고/수량을 숫자와 비교할 때만 stock에 넣어라.
- 안전재고를 숫자와 비교할 때만 safety_stock에 넣어라.
- **실재고(재고)와 안전재고를 서로 비교**하는 경우(예: "실재고가 안전재고보다 적다", "재고가 안전재고 미만")에는
  stock_below_safety를 true로 두고, stock/safety_stock에는 절대 넣지 마라. (컬럼 대 컬럼 비교는 DB에서 따로 처리된다.)
- 입고일 조건은 inbound_date에 넣어라.
- 최근출고일 조건은 last_outbound_date에 넣어라.
- "종류 뭐 있어", "뭐가 있어", "어떤 상품 있어" 같은 탐색 표현은 keyword에 넣지 말고, 핵심 상품명만 남겨라.
- 가격 조건이 아니라 재고/단가/가격 정보를 알려달라는 요청이면 requested_fields에 "stock", "price"를 넣어라.
- 상품명이나 모델명에 들어 있는 숫자와 영문은 함부로 버리지 마라. 예: "4K 모니터", "437", "ABC-123"
- "만원"은 10000, "천원"은 1000, "억"은 100000000으로 변환하라.
- "이상"은 >=, "초과"는 >, "이하"는 <=, "미만"은 < 로 변환하라.
- 정렬 요청이 없으면 sort는 "price_asc"로 둬라.
- limit은 {limit}으로 둬라.

다국어 입력(동남아·영어 등):
- 사용자는 한국어가 아닌 언어로 질문할 수 있다. 의미만 파악해 같은 JSON 스키마로 출력하라.
- DB category 값은 한국어 고정: 의류, 전자, 식품, 생활용품, 사무용품, 부품.
  (예: English electronics → category "전자", apparel/clothing → "의류", food → "식품")
- status: 재고 테이블이면 정상/부족/품절, 생산 테이블이면 생산중/대기/점검중 등 DB 값으로 매핑.
- 창고·공급업체는 데이터에 적힌 한글 명칭을 keyword/warehouse/supplier에 맞게 넣어라.

사용자 질문:
{query}
""".strip()


def extract_json_object(text):
    text = text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
        text = re.sub(r"```$", "", text).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or start > end:
        return None

    return text[start:end + 1]


def normalize_condition(condition):
    if not isinstance(condition, dict):
        return None

    operator = condition.get("operator")
    value = condition.get("value")
    if operator not in ALLOWED_OPERATORS:
        return None

    try:
        value = int(float(value))
    except (TypeError, ValueError):
        return None

    return {"operator": operator, "value": value}


def _compact_query(text):
    return re.sub(r"\s+", "", str(text or ""))


def wants_stock_below_safety(query):
    """질문이 재고 < 안전재고(컬럼 비교) 의도인지 휴리스틱으로 판별한다."""
    q = _compact_query(query)
    if not q:
        return False
    if re.search(
        r"(실재고|재고수량|현재고|재고).{0,16}(안전재고).{0,10}(적|낮|부족|미만|작)"
        r"|(안전재고).{0,16}(실재고|재고수량|재고|현재고).{0,10}(적|낮|부족|미만|작)"
        r"|(안전재고보다).{0,12}(실재고|재고)"
        r"|(실재고|재고).{0,12}(안전재고보다)",
        q,
    ):
        return True
    return False


def is_bogus_stock_safety_pair(stock, safety_stock):
    """LLM이 컬럼 비교를 stock<0 & safety>0 으로 잘못 쪼갠 패턴."""
    if not stock or not safety_stock:
        return False
    if stock.get("operator") != "<" or stock.get("value") != 0:
        return False
    if safety_stock.get("operator") not in {">", ">="}:
        return False
    if safety_stock.get("value") != 0:
        return False
    return True


def normalize_llm_parse(data, limit=TOP_K):
    if not isinstance(data, dict):
        return None

    def normalize_optional_text(value):
        if value is None:
            return None

        value = str(value).strip()
        return value or None

    keyword = data.get("keyword")
    if keyword is not None:
        keyword = str(keyword).strip()
        for strip_word in KEYWORD_STRIP_WORDS:
            keyword = keyword.replace(strip_word, "")
        keyword = re.sub(r"\s+", " ", keyword).strip()
        if keyword == "":
            keyword = None

    product_code = normalize_optional_text(data.get("product_code"))
    warehouse = normalize_optional_text(data.get("warehouse"))
    supplier = normalize_optional_text(data.get("supplier"))
    status = normalize_optional_text(data.get("status"))
    category = normalize_optional_text(data.get("category"))
    quality_status = normalize_optional_text(data.get("quality_status"))
    inbound_date = normalize_optional_text(data.get("inbound_date"))
    last_outbound_date = normalize_optional_text(data.get("last_outbound_date"))

    sort = data.get("sort", "price_asc")
    if sort not in ALLOWED_SORTS:
        sort = "price_asc"

    try:
        parsed_limit = int(data.get("limit", limit))
    except (TypeError, ValueError):
        parsed_limit = limit

    requested_fields = data.get("requested_fields", [])
    if not isinstance(requested_fields, list):
        requested_fields = []
    requested_fields = [
        field for field in (str(item).strip() for item in requested_fields)
        if field in {"stock", "price"}
    ]

    stock_cond = normalize_condition(data.get("stock"))
    safety_cond = normalize_condition(data.get("safety_stock"))

    sb = data.get("stock_below_safety")
    eval_mode = is_model_eval_mode()
    if not eval_mode and is_bogus_stock_safety_pair(stock_cond, safety_cond):
        stock_below_safety = True
    elif sb is True:
        stock_below_safety = True
    elif sb is False:
        stock_below_safety = False
    else:
        stock_below_safety = None

    if stock_below_safety:
        stock_cond = None
        safety_cond = None

    if not eval_mode:
        cat_synonym_keyword = {
            "전자 제품": "전자",
            "전자제품": "전자",
            "전자기기": "전자",
        }
        if keyword in cat_synonym_keyword and category is None:
            category = cat_synonym_keyword[keyword]
            keyword = None
        elif keyword in cat_synonym_keyword and category is not None:
            keyword = None

    qf = data.get("quality_flag")
    quality_flag = qf if qf in ("defect", "good") else None

    return {
        "keyword": keyword,
        "product_code": product_code,
        "warehouse": warehouse,
        "supplier": supplier,
        "status": status,
        "category": category,
        "quality_status": quality_status,
        "quality_flag": quality_flag,
        "price": normalize_condition(data.get("price")),
        "stock": stock_cond,
        "safety_stock": safety_cond,
        "stock_below_safety": bool(stock_below_safety) if stock_below_safety is not None else False,
        "inbound_date": inbound_date,
        "last_outbound_date": last_outbound_date,
        "requested_fields": requested_fields,
        "sort": sort,
        "limit": parsed_limit,
    }


def parse_query_with_llm(query, limit=TOP_K):
    response_text = generate_text(
        build_parser_prompt(query, limit),
        max_new_tokens=512,
        temperature=0.0,
    )
    if not response_text:
        return None
    json_text = extract_json_object(response_text)
    if json_text is None:
        return None

    try:
        parsed = json.loads(json_text)
    except json.JSONDecodeError:
        return None

    normalized = normalize_llm_parse(parsed, limit)
    if normalized is None:
        return None

    if not is_model_eval_mode() and wants_stock_below_safety(query):
        normalized["stock_below_safety"] = True
        normalized["stock"] = None
        normalized["safety_stock"] = None

    return normalized
