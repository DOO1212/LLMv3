"""실시간 생산 데이터의 품질 판정 기준(앱 고정). 엑셀·DB에는 판정 결과 문자열만 둔다."""

from __future__ import annotations

from typing import Final

# quality_status 컬럼에 저장되는 값 — 불량으로 볼 문자열
QUALITY_DEFECT_VALUES: Final[tuple[str, ...]] = ("불량", "NG")

# 양호로 볼 문자열
QUALITY_OK_VALUES: Final[tuple[str, ...]] = ("양호", "OK", "정상")
