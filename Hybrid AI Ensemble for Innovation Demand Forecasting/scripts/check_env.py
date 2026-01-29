from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


def _ensure_src_on_path() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    return repo_root


repo_root = _ensure_src_on_path()

from retail_forecast.panel import derive_week_ending_date, read_panel_xlsx  # noqa: E402


def main() -> None:
    panel_path = repo_root / "Dataset" / "Historical_Data.csv"

    if not panel_path.exists():
        raise SystemExit(f"Missing required dataset: {panel_path}")

    df = read_panel_xlsx(panel_path)

    print(f"Historical_Data.csv: {panel_path}")
    print(f"shape={df.shape}")
    print(f"columns={list(df.columns)}")
    print("head(2):")
    print(df.head(2).to_string(index=False))

    date = derive_week_ending_date(df["Periods"])
    dt = pd.to_datetime(date, errors="coerce")

    print(f"derived_date_min={dt.min().date()}")
    print(f"derived_date_max={dt.max().date()}")

    dow_counts = dt.dt.day_name().value_counts(dropna=False).sort_index()
    print("day_of_week_counts:")
    print(dow_counts.to_string())

    print(f"unique_markets={df['Markets'].nunique(dropna=True)}")
    print(f"unique_trademarks={df['Trademark'].nunique(dropna=True)}")
    print(f"unique_brands={df['Brand'].nunique(dropna=True)}")
    print(f"unique_weeks={pd.Series(date).nunique(dropna=True)}")


if __name__ == "__main__":
    main()
