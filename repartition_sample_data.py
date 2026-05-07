"""Repartition sample Excel inventory data into smaller category files."""

from __future__ import annotations

from pathlib import Path
import shutil

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
BACKUP_DIR = DATA_DIR / "_backup_original_5000"
ROWS_PER_FILE = 1000

# Ten smaller files: four categories split into A/B, two kept as single files.
PARTITIONS = [
    ("inventory_apparel_5000.xlsx", "inventory_apparel_a_1000.xlsx", 0),
    ("inventory_apparel_5000.xlsx", "inventory_apparel_b_1000.xlsx", 1),
    ("inventory_electronics_5000.xlsx", "inventory_electronics_a_1000.xlsx", 2),
    ("inventory_electronics_5000.xlsx", "inventory_electronics_b_1000.xlsx", 3),
    ("inventory_food_5000.xlsx", "inventory_food_a_1000.xlsx", 4),
    ("inventory_food_5000.xlsx", "inventory_food_b_1000.xlsx", 5),
    ("inventory_household_5000.xlsx", "inventory_household_a_1000.xlsx", 6),
    ("inventory_household_5000.xlsx", "inventory_household_b_1000.xlsx", 7),
    ("inventory_office_5000.xlsx", "inventory_office_1000.xlsx", 8),
    ("inventory_parts_5000.xlsx", "inventory_parts_1000.xlsx", 9),
]


def backup_original_files(source_files: set[str]) -> None:
    BACKUP_DIR.mkdir(exist_ok=True)

    for file_name in source_files:
        source_path = DATA_DIR / file_name
        backup_path = BACKUP_DIR / file_name
        if source_path.exists() and not backup_path.exists():
            shutil.move(str(source_path), str(backup_path))


def write_partitions() -> list[tuple[str, int]]:
    written: list[tuple[str, int]] = []

    for source_name, output_name, seed in PARTITIONS:
        source_path = BACKUP_DIR / source_name
        if not source_path.exists():
            raise FileNotFoundError(f"원본 파일이 없습니다: {source_path}")

        df = pd.read_excel(source_path)
        sample_size = min(ROWS_PER_FILE, len(df))
        sampled_df = df.sample(n=sample_size, random_state=seed).reset_index(drop=True)
        output_path = DATA_DIR / output_name
        sampled_df.to_excel(output_path, index=False)
        written.append((output_name, len(sampled_df)))

    return written


def remove_old_top_level_xlsx() -> None:
    for file_path in DATA_DIR.glob("*.xlsx"):
        file_path.unlink()


def main() -> None:
    source_files = {source_name for source_name, _, _ in PARTITIONS}
    backup_original_files(source_files)
    remove_old_top_level_xlsx()
    written = write_partitions()

    print("새 데이터 파일 생성 완료")
    for file_name, row_count in written:
        print(f"{file_name}: {row_count} rows")


if __name__ == "__main__":
    main()
