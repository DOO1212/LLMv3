"""실시간 생산 스냅샷 샘플 엑셀(~1000행). 품질은 판정값(양호/불량)만 — 불량 건수·율·원인 컬럼 없음."""

from __future__ import annotations

import random
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

random.seed(42)

LINES = [
    "1라인 SMT",
    "2라인 조립",
    "3라인 포장",
    "4라인 사출",
    "5라인 검사",
]
CATEGORIES = ["PCB", "조립", "포장", "사출", "금형"]
SUPPLIERS = [
    "삼성전자 DS",
    "LG이노텍",
    "코스모텍",
    "한국LM",
    "대덕전자",
    "이수페타시스",
]
STATUSES = (
    ["생산중"] * 40
    + ["대기"] * 35
    + ["점검중"] * 15
    + ["계획정지"] * 10
)

# config/quality.py 기준: QUALITY_DEFECT_VALUES, QUALITY_OK_VALUES 와 맞출 것
QUALITY_CHOICES_DEFECT = ("불량", "NG")
QUALITY_CHOICES_OK = ("양호", "OK", "정상")


def pick_quality(status: str) -> str:
    if status == "생산중":
        return random.choices(
            QUALITY_CHOICES_DEFECT + QUALITY_CHOICES_OK,
            weights=[12, 5, 38, 20, 25],
            k=1,
        )[0]
    if status == "점검중":
        return random.choices(
            QUALITY_CHOICES_OK + QUALITY_CHOICES_DEFECT,
            weights=[40, 25, 30, 3, 2],
            k=1,
        )[0]
    return random.choices(
        QUALITY_CHOICES_OK + QUALITY_CHOICES_DEFECT,
        weights=[45, 30, 22, 2, 1],
        k=1,
    )[0]


def random_date(start: datetime, end: datetime) -> datetime:
    delta = end - start
    sec = random.randint(0, int(delta.total_seconds()))
    return start + timedelta(seconds=sec)


def main():
    rows = []
    base_start = datetime(2025, 11, 1)
    base_end = datetime(2026, 4, 30)
    now_hint = datetime(2026, 5, 6, 14, 0, 0)

    for i in range(1000):
        cat = random.choice(CATEGORIES)
        status = random.choice(STATUSES)
        line = random.choice(LINES)
        quality = pick_quality(status)

        target = random.randint(800, 2200)
        if status == "생산중":
            qty = random.randint(int(target * 0.55), int(target * 0.98))
        elif status == "대기":
            qty = random.randint(0, int(target * 0.15))
        else:
            qty = random.randint(0, int(target * 0.4))

        cost = random.randint(1200, 89000)
        t0 = random_date(base_start, base_end)
        if status == "생산중":
            t_update = now_hint - timedelta(minutes=random.randint(1, 180))
        else:
            t_update = t0 + timedelta(hours=random.randint(2, 72))

        rows.append(
            {
                "품목명": f"{cat} 유닛 AU-{4200 + i:04d}",
                "생산LOT": f"LOT-{2026}{i:05d}",
                "제품군": cat,
                "생산라인": line,
                "가동상태": status,
                "금일누적생산수": qty,
                "목표수량": target,
                "품질결과": quality,
                "자재공급사": random.choice(SUPPLIERS),
                "작업시작일": t0.strftime("%Y-%m-%d %H:%M:%S"),
                "실시간갱신시각": t_update.strftime("%Y-%m-%d %H:%M:%S"),
                "제조원가": cost,
                "상세비고": f"배치 {i % 17 + 1} / 채널 {(i % 6) + 1}",
            }
        )

    df = pd.DataFrame(rows)
    path = DATA_DIR / "production_line_1000.xlsx"
    df.to_excel(path, index=False, sheet_name="Sheet1")
    print(f"Wrote {len(df)} rows -> {path}")


if __name__ == "__main__":
    main()
