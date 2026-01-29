from __future__ import annotations

from pathlib import Path
import sys


def _ensure_src_on_path() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    return repo_root


repo_root = _ensure_src_on_path()

import pandas as pd  # noqa: E402

from retail_forecast.curve_library import (  # noqa: E402
    build_distribution_library,
    build_velocity_library,
    save_curve_libraries,
)
from retail_forecast.decomposition import add_decomposition_columns  # noqa: E402
from retail_forecast.io import load_panel  # noqa: E402


def _print_counts_per_trademark(df_lib: pd.DataFrame, label: str) -> None:
    if df_lib.empty:
        print(f"{label}: empty")
        return
    counts = df_lib.groupby("trademark", dropna=False)["series_id"].nunique().sort_values(ascending=False)
    print(f"{label} series_count_per_trademark:")
    print(counts.to_string())


def main() -> None:
    df = load_panel(repo_root / "Dataset" / "Historical_Data.csv")
    df = add_decomposition_columns(df)

    dist_lib = build_distribution_library(df)
    vel_lib = build_velocity_library(df)

    dist_path, vel_path = save_curve_libraries(dist_lib, vel_lib, artifacts_dir=repo_root / "artifacts")

    _print_counts_per_trademark(dist_lib, "dist_lib")
    print("dist_lib head(3):")
    print(dist_lib.head(3).to_string(index=False))

    _print_counts_per_trademark(vel_lib, "vel_lib")
    print("vel_lib head(3):")
    print(vel_lib.head(3).to_string(index=False))

    print(f"dist_curve_library_parquet={dist_path}")
    print(f"vel_curve_library_parquet={vel_path}")


if __name__ == "__main__":
    main()
