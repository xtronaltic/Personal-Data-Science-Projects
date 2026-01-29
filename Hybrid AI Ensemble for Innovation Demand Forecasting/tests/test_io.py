from __future__ import annotations

from pathlib import Path

import pandas as pd

from retail_forecast.io import derive_date_from_periods, load_panel


def test_derive_date_from_periods_simple() -> None:
    s = pd.Series(["1 w/e 01/27/24"])
    out = derive_date_from_periods(s)
    assert out.iloc[0] == pd.Timestamp("2024-01-27")


def test_load_panel_asof_date_filtering(tmp_path: Path) -> None:
    # Create a minimal RetailPanel-like file with two weeks, then filter as-of the first week.
    df = pd.DataFrame(
        {
            "Markets": ["M1", "M1"],
            "Periods": ["1 w/e 01/27/24", "2 w/e 02/03/24"],
            "Manufacturer": ["K", "K"],
            "Category": ["C", "C"],
            "Trademark": ["T", "T"],
            "Brand": ["B", "B"],
            "$": [1.0, 2.0],
            "Units": [1.0, 2.0],
            "EQ": [1.0, 2.0],
            "Avg Unit Price": [1.0, 1.0],
            "%ACV": [0.1, 0.1],
            "TDP": [0.1, 0.1],
            "$ / $MM ACV": [10.0, 10.0],
            "Units / $MM ACV": [10.0, 10.0],
            "EQ / $MM ACV": [10.0, 10.0],
        }
    )
    xlsx = tmp_path / "Historical_Data.csv"
    df.to_excel(xlsx, index=False, engine="openpyxl")

    out = load_panel(xlsx, asof_date="2024-01-27")
    assert out.shape[0] == 1
    assert out["date"].max() == pd.Timestamp("2024-01-27")
    assert out.attrs["asof_date"] == pd.Timestamp("2024-01-27")
