"""Parse natural-language inventory queries into search conditions."""

from __future__ import annotations

import re

from config import KEYWORD_STRIP_WORDS, TOP_K


def parse_korean_money(text):
    if text is None:
        return None

    text = str(text).strip().replace(",", "").replace(" ", "").replace("원", "")
    total = 0

    eok = re.search(r"(\d+(?:\.\d+)?)억", text)
    if eok:
        total += int(float(eok.group(1)) * 100000000)

    man = re.search(r"(\d+(?:\.\d+)?)만", text)
    if man:
        total += int(float(man.group(1)) * 10000)

    cheon = re.search(r"(\d+(?:\.\d+)?)천", text)
    if cheon:
        total += int(float(cheon.group(1)) * 1000)

    if total > 0:
        return total

    num = re.search(r"\d+(?:\.\d+)?", text)
    if not num:
        return None

    return int(float(num.group(0)))


def extract_price_condition(query):
    compact_query = query.replace(" ", "")

    price_match = re.search(
        r"(?:가격|단가)(?:이|가)?(\d+(?:\.\d+)?(?:억|만|천|원)?|억원|만원|천원|억|만|천)(이상|초과|이하|미만)",
        compact_query,
    )
    if not price_match:
        price_match = re.search(
            r"(\d+(?:\.\d+)?(?:억|만|천|원)?|억원|만원|천원|억|만|천)(이상|초과|이하|미만)(?:인)?(?:가격|단가)",
            compact_query,
        )

    if not price_match:
        price_match = re.search(
            r"(\d+(?:\.\d+)?(?:억원|만원|천원|억|만|천|원)|억원|만원|천원|억|만|천)(이상|초과|이하|미만)(?:인|의)?",
            compact_query,
        )

    if not price_match:
        return None

    money_text = price_match.group(1)
    condition_word = price_match.group(2)

    if condition_word in ["이상", "초과"]:
        operator = ">="
    else:
        operator = "<="

    if condition_word in ["초과", "미만"]:
        operator = ">" if condition_word == "초과" else "<"

    if money_text in ["만원", "만"]:
        return {"operator": operator, "value": 10000}
    if money_text in ["천원", "천"]:
        return {"operator": operator, "value": 1000}
    if money_text in ["억원", "억"]:
        return {"operator": operator, "value": 100000000}

    value = parse_korean_money(money_text)
    if value is not None:
        return {"operator": operator, "value": value}

    return None


def extract_quality_flag(query):
    """config.quality 기준으로 불량/양호 질문만 구분한다. 세부 불량 데이터는 없다."""
    compact = query.replace(" ", "")
    if "품질결과" in compact or "품질판정" in compact:
        return None
    if "재고" in compact and "품질" not in compact and "QC" not in compact:
        if "정상" in compact and "불량" not in compact:
            return None
    if "불량아님" in compact or "불량이아닌" in compact or "불량제외" in compact:
        return "good"
    if "양호" in compact or re.search(r"(?:^|\s)OK(?:만|인|$)", query, re.I):
        return "good"
    if "품질" in compact and "정상" in compact and "불량" not in compact:
        return "good"
    if "불량" in compact:
        return "defect"
    if re.search(r"(?:^|\s)NG(?:\s|$|인|건)", query, re.I):
        return "defect"
    return None


def extract_stock_condition(query):
    compact_query = query.replace(" ", "")

    if "재고" not in compact_query:
        return None

    match = re.search(r"(?<!안전)재고(?:가|수량)?(\d+)(?:개|EA|ea)?(이상|초과|이하|미만)", compact_query)
    if not match:
        match = re.search(r"(\d+)(?:개|EA|ea)?(이상|초과|이하|미만).*(?<!안전)재고", compact_query)

    if not match:
        return None

    if match.lastindex == 2 and match.group(1).isdigit():
        value = int(match.group(1))
        condition_word = match.group(2)
    else:
        value = int(match.group(1))
        condition_word = match.group(2)

    if condition_word == "이상":
        operator = ">="
    elif condition_word == "초과":
        operator = ">"
    elif condition_word == "이하":
        operator = "<="
    else:
        operator = "<"

    return {"operator": operator, "value": value}


def extract_safety_stock_condition(query):
    compact_query = query.replace(" ", "")

    if "안전재고" not in compact_query:
        return None

    match = re.search(r"안전재고(?:가|수량)?(\d+)(?:개|EA|ea)?(이상|초과|이하|미만)", compact_query)
    if not match:
        match = re.search(r"(\d+)(?:개|EA|ea)?(이상|초과|이하|미만).*안전재고", compact_query)

    if not match:
        return None

    value = int(match.group(1))
    condition_word = match.group(2)

    if condition_word == "이상":
        operator = ">="
    elif condition_word == "초과":
        operator = ">"
    elif condition_word == "이하":
        operator = "<="
    else:
        operator = "<"

    return {"operator": operator, "value": value}


def extract_labeled_value(query, label):
    patterns = [
        rf"([^\s].*?)\s*{label}(?:에|에서|의|만|인|으로|를|은|는|이|가)?(?:\s|$)",
        rf"{label}(?:는|은|이|가|:)?\s*([^\s].*?)(?:\s|$)",
    ]

    for pattern in patterns:
        match = re.search(pattern, query)
        if not match:
            continue

        value = match.group(1).strip()
        value = re.sub(r"(?:에있는|에 있는|에서|만|상품|제품|품목|인|의)$", "", value).strip()
        if value:
            return value

    return None


def extract_warehouse_value(query):
    return extract_labeled_value(query, "창고")


def extract_supplier_value(query):
    return extract_labeled_value(query, "공급업체")


def extract_status_value(query):
    value = extract_labeled_value(query, "상태")
    if value is None:
        value = extract_labeled_value(query, "가동상태")
    if value is None:
        compact = query.replace(" ", "")
        if "생산중" in compact:
            return "생산중"
        if "점검중" in compact:
            return "점검중"
        if "계획정지" in compact:
            return "계획정지"
        if "대기" in compact and ("라인" in compact or "공정" in compact):
            return "대기"
        return None
    return re.sub(r"(?:인|인 상품|상품)$", "", value).strip() or None


def extract_quality_status_value(query):
    value = extract_labeled_value(query, "품질결과")
    if value is None:
        value = extract_labeled_value(query, "품질판정")
    return value


def extract_product_code_value(query):
    patterns = [
        r"상품코드(?:는|은|이|가|:)?\s*([A-Za-z0-9\-_]+)",
        r"([A-Za-z0-9\-_]+)\s*상품코드",
    ]

    for pattern in patterns:
        match = re.search(pattern, query)
        if match:
            return match.group(1).strip()

    return None


def extract_date_value(query, label):
    patterns = [
        rf"{label}(?:은|는|이|가|:)?\s*([0-9]{{4}}[-./][0-9]{{1,2}}[-./][0-9]{{1,2}})",
        rf"([0-9]{{4}}[-./][0-9]{{1,2}}[-./][0-9]{{1,2}})\s*{label}",
    ]

    for pattern in patterns:
        match = re.search(pattern, query)
        if match:
            return match.group(1).strip()

    return None


def extract_requested_fields(query):
    requested_fields = []

    if "재고" in query and extract_stock_condition(query) is None:
        requested_fields.append("stock")

    if ("단가" in query or "가격" in query) and extract_price_condition(query) is None:
        requested_fields.append("price")

    return requested_fields


def extract_category_keyword(query):
    match = re.search(r"([^\s]+)\s*카테고리", query)
    if not match:
        return None

    category = match.group(1).strip()
    return category or None


def extract_product_keyword(query):
    keyword = query
    keyword = re.sub(r"[^\s]+\s*카테고리", " ", keyword)
    keyword = re.sub(r"[^\s].*?\s*창고(?:에|에서|의|만|인|으로|를|은|는|이|가)?", " ", keyword)
    keyword = re.sub(r"[^\s].*?\s*공급업체(?:의|만|인|를|은|는|이|가)?", " ", keyword)
    keyword = re.sub(r"상태(?:는|은|이|가|:)?\s*[^\s]+", " ", keyword)
    keyword = re.sub(r"상품코드(?:는|은|이|가|:)?\s*[A-Za-z0-9\-_]+", " ", keyword)
    keyword = re.sub(r"[A-Za-z0-9\-_]+\s*상품코드", " ", keyword)
    keyword = re.sub(r"(?:입고일|최근출고일)(?:은|는|이|가|:)?\s*[0-9]{4}[-./][0-9]{1,2}[-./][0-9]{1,2}", " ", keyword)
    keyword = re.sub(r"[0-9]{4}[-./][0-9]{1,2}[-./][0-9]{1,2}\s*(?:입고일|최근출고일)", " ", keyword)
    keyword = re.sub(
        r"재고(?:가|수량)?\s*\d+\s*(?:개|EA|ea)?\s*(?:이상|초과|이하|미만)(?:이고|이며|인|의)?",
        " ",
        keyword,
    )
    keyword = re.sub(
        r"안전재고(?:가|수량)?\s*\d+\s*(?:개|EA|ea)?\s*(?:이상|초과|이하|미만)(?:이고|이며|인|의)?",
        " ",
        keyword,
    )
    keyword = re.sub(
        r"\d+\s*(?:개|EA|ea)?\s*(?:이상|초과|이하|미만)\s*재고(?:인|의)?",
        " ",
        keyword,
    )
    keyword = re.sub(
        r"\d+\s*(?:개|EA|ea)?\s*(?:이상|초과|이하|미만)\s*안전재고(?:인|의)?",
        " ",
        keyword,
    )
    keyword = re.sub(r"(?:재고와|재고|단가와|단가|가격과|가격)\s*(?:알려줘|알려주세요|보여줘|보여주세요)?", " ", keyword)
    keyword = re.sub(r"(?:종류|뭐\s*있어|뭐가\s*있어|어떤\s*것|어떤\s*상품|어떤\s*제품)", " ", keyword)
    keyword = re.sub(
        r"(?:가격|단가)(?:이|가)?\s*(?:\d+(?:\.\d+)?\s*)?(?:억원|만원|천원|억|만|천|원)?\s*(?:이상|초과|이하|미만)(?:이고|이며|인|의)?",
        " ",
        keyword,
    )
    keyword = re.sub(
        r"(?:\d+(?:\.\d+)?\s*)?(?:억원|만원|천원|억|만|천|원)?\s*(?:이상|초과|이하|미만)\s*(?:가격|단가)(?:인|의)?",
        " ",
        keyword,
    )
    keyword = re.sub(
        r"\d+(?:\.\d+)?\s*(?:억원|만원|천원|억|만|천|원)\s*(?:이상|초과|이하|미만)(?:이고|이며|인|의)?",
        " ",
        keyword,
    )
    keyword = re.sub(r"(^|\s)(?:이고|이며|인|의|이|가)(?=\s|$)", " ", keyword)
    keyword = re.sub(
        r"(^|\s)(?:상품만|제품만|품목만|상품들|제품들|품목들|상품|제품|품목|있는|만|정상|안전)(?=\s|$)",
        " ",
        keyword,
    )

    for word in KEYWORD_STRIP_WORDS:
        keyword = keyword.replace(word, "")

    keyword = re.sub(r"\b(?:개|EA|ea)\b", " ", keyword)
    keyword = re.sub(r"[,:()?!]+", " ", keyword)
    keyword = re.sub(r"\s+", " ", keyword)
    keyword = keyword.strip()

    return keyword or None


def parse_query(query, limit=TOP_K):
    parsed = {
        "keyword": extract_product_keyword(query),
        "product_code": extract_product_code_value(query),
        "warehouse": extract_warehouse_value(query),
        "supplier": extract_supplier_value(query),
        "status": extract_status_value(query),
        "category": extract_category_keyword(query),
        "safety_stock": extract_safety_stock_condition(query),
        "inbound_date": extract_date_value(query, "입고일"),
        "last_outbound_date": extract_date_value(query, "최근출고일"),
        "requested_fields": extract_requested_fields(query),
        "price": extract_price_condition(query),
        "stock": extract_stock_condition(query),
        "quality_flag": extract_quality_flag(query),
        "quality_status": extract_quality_status_value(query),
        "stock_below_safety": False,
        "sort": "price_asc",
        "limit": limit,
    }

    if parsed.get("status") in {"생산중", "대기", "점검중", "계획정지"}:
        parsed["keyword"] = None

    qf = parsed.get("quality_flag")
    if qf in {"defect", "good"} and parsed.get("keyword") in {
        "불량",
        "NG",
        "ng",
        "양호",
        "OK",
        "ok",
    }:
        parsed["keyword"] = None

    if qf == "good" and parsed.get("keyword"):
        stripped = re.sub(r"만$", "", parsed["keyword"]).strip()
        if stripped in {"양호", "OK", "ok"}:
            parsed["keyword"] = None

    if parsed.get("quality_status"):
        parsed["keyword"] = None

    return parsed
