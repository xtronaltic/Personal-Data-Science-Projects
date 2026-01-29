#!/usr/bin/env python3
"""Data anonymization pipeline for retail panel data.

Anonymizes categorical columns while preserving numeric (FACT) columns.
Generates consistent mappings across all input files for referential integrity.

Usage:
    python scripts/anonymize_data.py --input Dataset/ --output public_data/ --seed 42
    python scripts/anonymize_data.py --input Dataset/Historical_Data.csv --output examples/sample_data.csv
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any


def _ensure_src_on_path() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    return repo_root


REPO_ROOT = _ensure_src_on_path()

import pandas as pd

CATEGORICAL_PATTERNS: dict[str, str] = {
    "markets": "MARKET",
    "manufacturer": "MFR",
    "category": "CAT",
    "trademark": "TM",
    "brand": "BRAND",
    "Markets": "MARKET",
    "Manufacturer": "MFR",
    "Category": "CAT",
    "Trademark": "TM",
    "Brand": "BRAND",
}

NUMERIC_COLUMN_PATTERNS: list[str] = [
    "$",
    "dollars",
    "units",
    "eq",
    "acv",
    "tdp",
    "price",
    "p10",
    "p50",
    "p90",
    "horizon",
    "date",
    "periods",
    "week",
]


def is_numeric_column(col: str) -> bool:
    """Check if column should be preserved (numeric/FACT column)."""
    col_lower = col.lower()
    for pattern in NUMERIC_COLUMN_PATTERNS:
        if pattern in col_lower:
            return True
    return False


def get_categorical_prefix(col: str) -> str | None:
    """Get anonymization prefix for a categorical column."""
    col_lower = col.lower()
    for pattern, prefix in CATEGORICAL_PATTERNS.items():
        if pattern.lower() in col_lower or col == pattern:
            return prefix
    if "manufacturer" in col_lower:
        return "MFR"
    if "category" in col_lower:
        return "CAT"
    if "trademark" in col_lower:
        return "TM"
    if "brand" in col_lower:
        return "BRAND"
    if "market" in col_lower:
        return "MARKET"
    return None


class Anonymizer:
    """Consistent anonymization across multiple files."""

    def __init__(self, seed: int | None = None):
        self.seed = seed
        self.mappings: dict[str, dict[str, str]] = {}
        self.counters: dict[str, int] = {}
        self._rng = random.Random(seed)

    def _get_or_create_token(self, column: str, value: Any, prefix: str) -> str:
        """Get existing token or create new one for a value."""
        if column not in self.mappings:
            self.mappings[column] = {}
            self.counters[column] = 0

        str_value = str(value) if pd.notna(value) else "__NULL__"

        if str_value in self.mappings[column]:
            return self.mappings[column][str_value]

        self.counters[column] += 1
        token = f"{prefix}_{self.counters[column]:03d}"
        self.mappings[column][str_value] = token
        return token

    def anonymize_column(self, series: pd.Series, column: str) -> pd.Series:
        """Anonymize a single column."""
        prefix = get_categorical_prefix(column)
        if prefix is None:
            return series

        return series.apply(
            lambda x: self._get_or_create_token(column, x, prefix) if pd.notna(x) else x
        )

    def anonymize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Anonymize all categorical columns and rename headers to neutral names."""
        df_out = df.copy()
        rename_map: dict[str, str] = {}

        for col in df.columns:
            if is_numeric_column(col):
                continue

            prefix = get_categorical_prefix(col)
            if prefix is not None:
                # Determine new generic column name
                generic_name = prefix.title()
                if prefix == "TM":
                    generic_name = "Trademark"
                elif prefix == "MFR":
                    generic_name = "Manufacturer"
                elif prefix == "CAT":
                    generic_name = "Category"
                elif prefix == "MARKET":
                    generic_name = "Markets"

                if generic_name != col:
                    rename_map[col] = generic_name

                print(f"  Anonymizing column: {col} -> {generic_name} (values {prefix}_XXX)")
                df_out[col] = self.anonymize_column(df[col], col)

        if rename_map:
            df_out = df_out.rename(columns=rename_map)
            print(f"  Renamed columns: {rename_map}")

        return df_out

    def get_mapping_dict(self) -> dict[str, dict[str, str]]:
        """Get the complete mapping dictionary."""
        return {col: dict(mapping) for col, mapping in self.mappings.items()}

    def save_mapping(self, path: Path) -> None:
        """Save mapping to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "seed": self.seed,
                    "mappings": self.get_mapping_dict(),
                    "counters": dict(self.counters),
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        print(f"Saved mapping to: {path}")


def read_input_file(path: Path) -> pd.DataFrame:
    """Read Excel or CSV file."""
    suffix = path.suffix.lower()
    if suffix == ".xlsx":
        return pd.read_excel(path)
    elif suffix == ".csv":
        return pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")


def get_input_files(input_path: Path) -> list[Path]:
    """Get list of input files from path (file or directory)."""
    if input_path.is_file():
        return [input_path]

    if input_path.is_dir():
        files = []
        for ext in ["*.xlsx", "*.csv"]:
            files.extend(input_path.glob(ext))
        return sorted(files)

    raise FileNotFoundError(f"Input path not found: {input_path}")


def determine_output_path(
    input_file: Path, output_arg: Path, is_single_file: bool
) -> Path:
    """Determine output path for a given input file."""
    if is_single_file and output_arg.suffix == ".csv":
        return output_arg

    output_dir = output_arg if output_arg.is_dir() or not output_arg.suffix else output_arg.parent
    return output_dir / f"{input_file.stem}_anonymized.csv"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Anonymize retail panel data while preserving numeric columns.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/anonymize_data.py --input Dataset/ --output public_data/ --seed 42
  python scripts/anonymize_data.py --input Dataset/Historical_Data.csv --output examples/sample_data.csv
        """,
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=REPO_ROOT / "Dataset",
        help="Input file or directory (default: Dataset/)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=REPO_ROOT / "public_data",
        help="Output file or directory (default: public_data/)",
    )
    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=None,
        help="Random seed for reproducible anonymization",
    )
    parser.add_argument(
        "--mapping",
        "-m",
        type=Path,
        default=REPO_ROOT / "mappings" / "anonymization_map.json",
        help="Path to save mapping file (default: mappings/anonymization_map.json)",
    )

    args = parser.parse_args()

    try:
        input_files = get_input_files(args.input)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if not input_files:
        print(f"No .xlsx or .csv files found in: {args.input}", file=sys.stderr)
        return 1

    print(f"Found {len(input_files)} input file(s)")
    for f in input_files:
        print(f"  - {f.name}")

    anonymizer = Anonymizer(seed=args.seed)
    is_single_file = len(input_files) == 1

    output_dir = args.output if args.output.is_dir() or not args.output.suffix else args.output.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    for input_file in input_files:
        print(f"\nProcessing: {input_file.name}")

        try:
            df = read_input_file(input_file)
            print(f"  Loaded {len(df):,} rows, {len(df.columns)} columns")
        except Exception as e:
            print(f"  Error reading file: {e}", file=sys.stderr)
            continue

        df_anon = anonymizer.anonymize_dataframe(df)

        output_path = determine_output_path(input_file, args.output, is_single_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_anon.to_csv(output_path, index=False)
        print(f"  Saved: {output_path}")

    anonymizer.save_mapping(args.mapping)

    total_mappings = sum(len(m) for m in anonymizer.mappings.values())
    print("\nAnonymization complete:")
    print(f"  - {len(input_files)} file(s) processed")
    print(f"  - {total_mappings} unique values mapped across {len(anonymizer.mappings)} columns")
    print(f"  - Mapping saved to: {args.mapping}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
