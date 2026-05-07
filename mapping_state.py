"""Persist mapping state for newly discovered Excel headers."""

from __future__ import annotations

import json
import os
from copy import deepcopy

from config import COLUMN_SYNONYMS, OUTPUT_DIR

KNOWN_HEADERS_PATH = os.path.join(OUTPUT_DIR, "known_headers.json")
MAPPING_OVERRIDES_PATH = os.path.join(OUTPUT_DIR, "column_mapping_overrides.json")


def _load_json(path, default):
    if not os.path.exists(path):
        return deepcopy(default)

    with open(path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def _save_json(path, data):
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(data, fp, ensure_ascii=False, indent=2)


def load_known_headers():
    return _load_json(KNOWN_HEADERS_PATH, [])


def save_known_headers(headers):
    deduped = sorted({str(h).strip() for h in headers if str(h).strip() != ""})
    _save_json(KNOWN_HEADERS_PATH, deduped)


def load_mapping_overrides():
    data = _load_json(MAPPING_OVERRIDES_PATH, {})
    if not isinstance(data, dict):
        return {}

    normalized = {}
    for field, values in data.items():
        if not isinstance(values, list):
            continue
        normalized[field] = [str(v).strip() for v in values if str(v).strip() != ""]
    return normalized


def save_mapping_overrides(overrides):
    cleaned = {}
    for field, values in overrides.items():
        deduped = sorted({str(v).strip() for v in values if str(v).strip() != ""})
        if deduped:
            cleaned[field] = deduped
    _save_json(MAPPING_OVERRIDES_PATH, cleaned)


def add_override_synonym(field, header_name):
    header_name = str(header_name).strip()
    if header_name == "":
        return False
    if field not in COLUMN_SYNONYMS:
        return False

    overrides = load_mapping_overrides()
    field_values = set(overrides.get(field, []))
    before = len(field_values)
    field_values.add(header_name)
    overrides[field] = sorted(field_values)
    save_mapping_overrides(overrides)
    return len(field_values) > before


def get_merged_column_synonyms():
    merged = {field: list(values) for field, values in COLUMN_SYNONYMS.items()}
    overrides = load_mapping_overrides()

    for field, values in overrides.items():
        if field not in merged:
            continue
        existing = set(merged[field])
        for value in values:
            if value not in existing:
                merged[field].append(value)
                existing.add(value)
    return merged
