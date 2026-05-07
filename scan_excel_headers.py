"""Scan Excel header rows and report how they map to standard fields."""

from __future__ import annotations

import glob
import json
import os
from collections import defaultdict
from datetime import datetime, timezone

import pandas as pd

from build_db import map_column_name, normalize_text
from config import DATA_DIR, OUTPUT_DIR

REPORT_JSON_PATH = os.path.join(OUTPUT_DIR, "header_scan_report.json")
REPORT_TXT_PATH = os.path.join(OUTPUT_DIR, "header_scan_report.txt")


def iter_excel_sheets(data_dir):
    excel_files = sorted(glob.glob(os.path.join(data_dir, "*.xlsx")))
    excel_files += sorted(glob.glob(os.path.join(data_dir, "*.xls")))

    for file_path in excel_files:
        xls = pd.ExcelFile(file_path)
        for sheet_name in xls.sheet_names:
            yield file_path, sheet_name


def build_header_report(data_dir):
    header_occurrences = defaultdict(list)
    files_seen = set()
    sheets_seen = 0

    for file_path, sheet_name in iter_excel_sheets(data_dir):
        files_seen.add(os.path.basename(file_path))
        sheets_seen += 1

        df = pd.read_excel(file_path, sheet_name=sheet_name, nrows=0)
        for column_name in df.columns:
            header_text = str(column_name).strip()
            header_occurrences[header_text].append(
                {
                    "source_file": os.path.basename(file_path),
                    "sheet_name": sheet_name,
                }
            )

    headers = []
    mapped_count = 0
    unmapped_count = 0

    for header_text in sorted(header_occurrences):
        mapped_field = map_column_name(header_text)
        if mapped_field is None:
            unmapped_count += 1
        else:
            mapped_count += 1

        headers.append(
            {
                "header": header_text,
                "normalized_header": normalize_text(header_text),
                "mapped_field": mapped_field,
                "occurrence_count": len(header_occurrences[header_text]),
                "occurrences": header_occurrences[header_text],
            }
        )

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_dir": data_dir,
        "files_scanned": len(files_seen),
        "sheets_scanned": sheets_seen,
        "unique_header_count": len(headers),
        "mapped_header_count": mapped_count,
        "unmapped_header_count": unmapped_count,
        "headers": headers,
    }


def write_json_report(report, output_path):
    with open(output_path, "w", encoding="utf-8") as fp:
        json.dump(report, fp, ensure_ascii=False, indent=2)


def write_text_report(report, output_path):
    lines = [
        "Excel Header Scan Report",
        f"generated_at_utc: {report['generated_at_utc']}",
        f"data_dir: {report['data_dir']}",
        f"files_scanned: {report['files_scanned']}",
        f"sheets_scanned: {report['sheets_scanned']}",
        f"unique_header_count: {report['unique_header_count']}",
        f"mapped_header_count: {report['mapped_header_count']}",
        f"unmapped_header_count: {report['unmapped_header_count']}",
        "",
        "[Header Mapping]",
    ]

    for item in report["headers"]:
        mapped_label = item["mapped_field"] or "UNMAPPED"
        lines.append(
            f"- {item['header']} -> {mapped_label} "
            f"(normalized={item['normalized_header']}, occurrences={item['occurrence_count']})"
        )
        for occurrence in item["occurrences"]:
            lines.append(
                f"  - {occurrence['source_file']} / {occurrence['sheet_name']}"
            )

    with open(output_path, "w", encoding="utf-8") as fp:
        fp.write("\n".join(lines) + "\n")


def main():
    report = build_header_report(DATA_DIR)
    write_json_report(report, REPORT_JSON_PATH)
    write_text_report(report, REPORT_TXT_PATH)

    print(f"헤더 스캔 완료: 파일 {report['files_scanned']}개, 시트 {report['sheets_scanned']}개")
    print(f"고유 헤더 수: {report['unique_header_count']}")
    print(f"매핑 완료: {report['mapped_header_count']}개")
    print(f"미매핑: {report['unmapped_header_count']}개")
    print(f"JSON 리포트: {REPORT_JSON_PATH}")
    print(f"텍스트 리포트: {REPORT_TXT_PATH}")


if __name__ == "__main__":
    main()
