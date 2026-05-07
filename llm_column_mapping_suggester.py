"""Suggest standard-field mappings for Excel headers using a local LLM."""

from __future__ import annotations

import glob
import json
import os
import urllib.error
import urllib.request
from datetime import datetime, timezone

import pandas as pd

from config import COLUMN_SYNONYMS, DATA_DIR, OUTPUT_DIR
from llm_fallback import DEFAULT_LLAMA_MODEL, DEFAULT_OLLAMA_URL, get_ollama_timeout

SCAN_REPORT_PATH = os.path.join(OUTPUT_DIR, "header_scan_report.json")
SUGGESTION_JSON_PATH = os.path.join(OUTPUT_DIR, "header_llm_mapping_suggestions.json")
SUGGESTION_TXT_PATH = os.path.join(OUTPUT_DIR, "header_llm_mapping_suggestions.txt")


def load_scan_report(path=SCAN_REPORT_PATH):
    with open(path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def find_excel_path(source_file, data_dir=DATA_DIR):
    direct_path = os.path.join(data_dir, source_file)
    if os.path.exists(direct_path):
        return direct_path

    matches = glob.glob(os.path.join(data_dir, "**", source_file), recursive=True)
    if matches:
        return matches[0]

    return None


def sample_header_values(header_item, sample_size=5):
    values = []
    seen = set()

    for occurrence in header_item.get("occurrences", []):
        file_path = find_excel_path(occurrence["source_file"])
        if file_path is None:
            continue

        sheet_name = occurrence["sheet_name"]
        try:
            df = pd.read_excel(file_path, sheet_name=sheet_name, usecols=[header_item["header"]])
        except Exception:
            continue

        for value in df[header_item["header"]].tolist():
            if pd.isna(value):
                continue

            text = str(value).strip()
            if text == "" or text in seen:
                continue

            seen.add(text)
            values.append(text)
            if len(values) >= sample_size:
                return values

    return values


def build_prompt(header_item):
    standard_fields = []
    for field, synonyms in COLUMN_SYNONYMS.items():
        standard_fields.append(
            {
                "field": field,
                "known_synonyms": synonyms,
            }
        )

    return f"""
너는 엑셀 컬럼 표준화 도우미다.
목표는 미매핑 헤더를 표준 필드에 자동 확정하는 것이 아니라, 후보를 신중하게 추천하는 것이다.

표준 필드 목록:
{json.dumps(standard_fields, ensure_ascii=False, indent=2)}

입력으로 헤더의 이름, 정규화 값, 샘플 값이 주어진다.
샘플 값을 보고 의미를 추론하라.

반드시 JSON만 출력하고 설명 문장은 쓰지 마라.
출력 형식:
{{
  "header": "원본 헤더명",
  "recommended_field": "product_name|price|description|stock|category|brand|warehouse|supplier|status|null",
  "confidence": "high|medium|low",
  "reason": "짧은 한국어 설명",
  "review_needed": true,
  "alternative_fields": ["후보1", "후보2"]
}}

규칙:
- 확실하지 않으면 recommended_field는 null로 두어라.
- 식별자성 컬럼(예: 코드, ID)은 기존 표준 필드에 억지로 맞추지 마라.
- 날짜 컬럼(예: 입고일, 최근출고일)은 현재 표준 필드에 맞는 것이 없으면 null로 두어라.
- 재고금액처럼 계산 파생 컬럼이면 null로 둘 수 있다.
- 안전재고는 stock과 비슷해 보이더라도 의미가 다르면 review_needed를 true로 하고 신중히 추천하라.
- alternative_fields에는 최대 2개만 넣어라.

검토 대상 헤더:
{json.dumps(header_item, ensure_ascii=False, indent=2)}
""".strip()


def call_ollama(prompt):
    ollama_url = os.environ.get("OLLAMA_URL", DEFAULT_OLLAMA_URL)
    model = os.environ.get("LLAMA_MODEL", DEFAULT_LLAMA_MODEL)
    timeout = get_ollama_timeout()
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
    }
    request = urllib.request.Request(
        ollama_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=timeout) as response:
        response_data = json.loads(response.read().decode("utf-8"))

    return model, response_data.get("response", "")


def extract_json_object(text):
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or start > end:
        return None
    return text[start:end + 1]


def build_unmapped_payload(scan_report):
    unmapped_headers = []

    for item in scan_report["headers"]:
        if item["mapped_field"] is not None:
            continue

        unmapped_headers.append(
            {
                "header": item["header"],
                "normalized_header": item["normalized_header"],
                "occurrence_count": item["occurrence_count"],
                "sample_values": sample_header_values(item),
            }
        )

    return unmapped_headers


def suggest_header_mapping(header_item):
    prompt = build_prompt(header_item)
    _, response_text = call_ollama(prompt)
    json_text = extract_json_object(response_text)
    if json_text is None:
        return {
            "header": header_item["header"],
            "recommended_field": None,
            "confidence": "low",
            "reason": "LLM 응답에서 JSON을 찾지 못함",
            "review_needed": True,
            "alternative_fields": [],
        }

    try:
        parsed = json.loads(json_text)
    except json.JSONDecodeError:
        return {
            "header": header_item["header"],
            "recommended_field": None,
            "confidence": "low",
            "reason": "LLM 응답 JSON 파싱 실패",
            "review_needed": True,
            "alternative_fields": [],
        }

    return {
        "header": parsed.get("header", header_item["header"]),
        "recommended_field": parsed.get("recommended_field"),
        "confidence": parsed.get("confidence", "low"),
        "reason": parsed.get("reason", ""),
        "review_needed": bool(parsed.get("review_needed", True)),
        "alternative_fields": parsed.get("alternative_fields", []),
    }


def write_reports(report):
    with open(SUGGESTION_JSON_PATH, "w", encoding="utf-8") as fp:
        json.dump(report, fp, ensure_ascii=False, indent=2)

    lines = [
        "LLM Column Mapping Suggestions",
        f"generated_at_utc: {report['generated_at_utc']}",
        f"model: {report['model']}",
        f"scan_report_path: {report['scan_report_path']}",
        "",
    ]

    for item in report["suggestions"]:
        recommended = item["recommended_field"] or "NO_MATCH"
        alternatives = ", ".join(item.get("alternative_fields", [])) or "-"
        lines.append(f"- {item['header']} -> {recommended}")
        lines.append(f"  confidence: {item['confidence']}")
        lines.append(f"  review_needed: {item['review_needed']}")
        lines.append(f"  alternatives: {alternatives}")
        lines.append(f"  reason: {item['reason']}")
        sample_values = ", ".join(item.get("sample_values", [])) or "-"
        lines.append(f"  sample_values: {sample_values}")

    with open(SUGGESTION_TXT_PATH, "w", encoding="utf-8") as fp:
        fp.write("\n".join(lines) + "\n")


def main():
    scan_report = load_scan_report()
    unmapped_headers = build_unmapped_payload(scan_report)

    if not unmapped_headers:
        print("미매핑 헤더가 없습니다.")
        return

    try:
        model = os.environ.get("LLAMA_MODEL", DEFAULT_LLAMA_MODEL)
        suggestions = [suggest_header_mapping(item) for item in unmapped_headers]
    except FileNotFoundError:
        raise
    except urllib.error.URLError as error:
        print(f"Ollama 호출 실패: {error.reason}")
        return
    suggestion_map = {item["header"]: item for item in suggestions if "header" in item}

    enriched_suggestions = []
    for header_item in unmapped_headers:
        suggestion = suggestion_map.get(header_item["header"], {})
        enriched_suggestions.append(
            {
                "header": header_item["header"],
                "normalized_header": header_item["normalized_header"],
                "sample_values": header_item["sample_values"],
                "recommended_field": suggestion.get("recommended_field"),
                "confidence": suggestion.get("confidence", "low"),
                "reason": suggestion.get("reason", ""),
                "review_needed": bool(suggestion.get("review_needed", True)),
                "alternative_fields": suggestion.get("alternative_fields", []),
            }
        )

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "scan_report_path": SCAN_REPORT_PATH,
        "suggestions": enriched_suggestions,
    }
    write_reports(report)

    print(f"LLM 컬럼 매핑 제안 완료: {len(enriched_suggestions)}개")
    print(f"JSON 리포트: {SUGGESTION_JSON_PATH}")
    print(f"텍스트 리포트: {SUGGESTION_TXT_PATH}")


if __name__ == "__main__":
    main()
