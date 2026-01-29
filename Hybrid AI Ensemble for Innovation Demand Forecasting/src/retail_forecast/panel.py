from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import pandas as pd


PANEL_COLUMNS_EXACT = [
    "Markets",
    "Periods",
    "Manufacturer",
    "Category",
    "Trademark",
    "Brand",
    "$",
    "Units",
    "EQ",
    "Avg Unit Price",
    "%ACV",
    "TDP",
    "$ / $MM ACV",
    "Units / $MM ACV",
    "EQ / $MM ACV",
]

_W_E_RE = re.compile(r"\bw/e\s*(?P<date>\d{1,2}/\d{1,2}/\d{2,4})\b", flags=re.IGNORECASE)


def read_panel_xlsx(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, engine="openpyxl")
    cols = list(df.columns)
    if cols != PANEL_COLUMNS_EXACT:
        missing = [c for c in PANEL_COLUMNS_EXACT if c not in cols]
        extra = [c for c in cols if c not in PANEL_COLUMNS_EXACT]
        raise ValueError(
            "Historical_Data.csv columns must match exactly. "
            f"Missing={missing} Extra={extra} OrderMatches={cols == PANEL_COLUMNS_EXACT}"
        )
    return df


def derive_week_ending_date(periods: Iterable[str]) -> pd.Series:
    extracted: list[str | None] = []
    for p in periods:
        if p is None or (isinstance(p, float) and pd.isna(p)):
            extracted.append(None)
            continue
        m = _W_E_RE.search(str(p))
        extracted.append(m.group("date") if m else None)

    s = pd.Series(extracted, dtype="string")

    # Deterministic parsing: M/D/YY is typical; allow M/D/YYYY.
    dt = pd.to_datetime(s, format="%m/%d/%y", errors="coerce")
    dt_4 = pd.to_datetime(s, format="%m/%d/%Y", errors="coerce")
    dt = dt.fillna(dt_4)

    if dt.isna().any():
        bad = int(dt.isna().sum())
        raise ValueError(
            f"Failed to parse {bad} Periods into dates. "
            "Expected like '1 w/e 01/27/24'."
        )

    # Business rule: derived date is week-ending Saturday.
    non_sat = dt.dt.weekday != 5
    if non_sat.any():
        sample = pd.Series(periods, dtype="string")[non_sat].head(3).tolist()
        raise ValueError(f"Derived week-ending dates not Saturday for some rows. Sample Periods={sample}")

    return dt.dt.date
