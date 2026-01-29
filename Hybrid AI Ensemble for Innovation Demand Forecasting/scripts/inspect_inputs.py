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

from retail_forecast.io import fingerprint_df, load_new_innovation, load_panel  # noqa: E402


def _tiny_summary(df: pd.DataFrame, label: str) -> None:
    print(f"{label} shape={df.shape}")
    print(f"{label} columns={list(df.columns)}")
    dates = pd.to_datetime(df["date"], errors="coerce") if "date" in df.columns else None
    if dates is not None:
        print(f"{label} date_range={dates.min().date()}..{dates.max().date()}")


def main() -> None:
    panel_df = load_panel(repo_root / "Dataset" / "Historical_Data.csv")
    asof = panel_df.attrs.get("asof_date")
    print(f"Panel asof_date={asof.date() if hasattr(asof, 'date') else asof}")
    _tiny_summary(panel_df, "Panel")
    print(f"Panel fingerprint={fingerprint_df(panel_df)}")

    new_df = load_new_innovation()
    if new_df is None:
        print("New_Innovations: not found")
        return

    asof_new = new_df.attrs.get("asof_date")
    print(f"New_Innovations asof_date={asof_new.date() if hasattr(asof_new, 'date') else asof_new}")
    _tiny_summary(new_df, "New_Innovations")
    print(f"New_Innovations fingerprint={fingerprint_df(new_df)}")


if __name__ == "__main__":
    main()
