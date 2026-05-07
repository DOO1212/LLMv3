"""Review first-seen Excel headers with LLM suggestions and yes/no approval."""

from __future__ import annotations

import os
import urllib.error

from llm_column_mapping_suggester import (
    build_unmapped_payload,
    load_scan_report,
    suggest_header_mapping,
)
from mapping_state import (
    add_override_synonym,
    load_known_headers,
    save_known_headers,
)
from scan_excel_headers import REPORT_JSON_PATH, build_header_report, write_json_report
from config import DATA_DIR


def collect_current_headers(scan_report):
    return [item["header"] for item in scan_report.get("headers", [])]


def ensure_scan_report():
    if os.path.exists(REPORT_JSON_PATH):
        return load_scan_report(REPORT_JSON_PATH)

    report = build_header_report(DATA_DIR)
    write_json_report(report, REPORT_JSON_PATH)
    return report


def find_new_headers(scan_report):
    known_headers = set(load_known_headers())
    current_headers = collect_current_headers(scan_report)
    new_headers = [h for h in current_headers if h not in known_headers]
    return new_headers, current_headers


def ask_yes_no(prompt):
    while True:
        answer = input(prompt).strip().lower()
        if answer in {"y", "yes"}:
            return "yes"
        if answer in {"n", "no", ""}:
            return "no"
        if answer in {"a", "all"}:
            return "all_yes"
        if answer in {"s", "skip"}:
            return "skip_all"
        print("입력은 y/yes, n/no, a/all(남은 항목 모두 승인), s/skip(남은 항목 모두 건너뜀) 중 하나로 해주세요.")


def main():
    scan_report = ensure_scan_report()
    new_headers, current_headers = find_new_headers(scan_report)

    if not new_headers:
        print("새로운 컬럼이 없습니다.")
        return

    print(f"새로운 컬럼 {len(new_headers)}개를 발견했습니다.")
    print(", ".join(new_headers))
    print("")

    unmapped_payload = build_unmapped_payload(scan_report)
    payload_by_header = {item["header"]: item for item in unmapped_payload}

    approved_count = 0
    auto_mode = None

    for header_name in new_headers:
        header_payload = payload_by_header.get(
            header_name,
            {
                "header": header_name,
                "normalized_header": header_name,
                "occurrence_count": 0,
                "sample_values": [],
            },
        )
        try:
            suggestion = suggest_header_mapping(header_payload)
        except urllib.error.URLError as error:
            print(f"LLM 추천 호출 실패: {error.reason}")
            break

        print("=" * 72)
        print(f"컬럼: {header_name}")
        print(f"샘플값: {', '.join(header_payload.get('sample_values', [])) or '-'}")
        print(f"추천 필드: {suggestion.get('recommended_field')}")
        print(f"신뢰도: {suggestion.get('confidence')}")
        print(f"사유: {suggestion.get('reason')}")
        alternatives = suggestion.get("alternative_fields") or []
        print(f"대안: {', '.join(alternatives) if alternatives else '-'}")

        recommended_field = suggestion.get("recommended_field")
        if not recommended_field:
            print("추천 필드가 null이라 자동 승인 대상이 아닙니다.")
            continue

        if auto_mode == "all_yes":
            decision = "yes"
            print("자동 모드: 남은 항목 모두 승인(all yes)")
        elif auto_mode == "skip_all":
            decision = "no"
            print("자동 모드: 남은 항목 모두 건너뜀(skip all)")
        else:
            decision = ask_yes_no(
                f"'{header_name}'를 '{recommended_field}'로 오버라이드에 추가할까요? "
                "[y/N/a=all/s=skip]: "
            )

        if decision == "all_yes":
            auto_mode = "all_yes"
            decision = "yes"
        elif decision == "skip_all":
            auto_mode = "skip_all"
            decision = "no"

        if decision != "yes":
            continue

        added = add_override_synonym(recommended_field, header_name)
        if added:
            approved_count += 1
            print("오버라이드에 추가했습니다.")
        else:
            print("이미 오버라이드에 등록되어 있습니다.")

    save_known_headers(current_headers)

    print("")
    print(f"검토 완료. 승인 추가 수: {approved_count}")
    print("현재 헤더 목록을 known_headers.json에 갱신했습니다.")


if __name__ == "__main__":
    main()
