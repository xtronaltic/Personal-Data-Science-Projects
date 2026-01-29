from __future__ import annotations

import hashlib
import warnings
from pathlib import Path
from typing import Iterable

import pandas as pd

from .constants import (
    HIERARCHY_COLS,
    NEW_INNOVATION_REQUIRED_COLUMNS,
    PANEL_REQUIRED_COLUMNS,
    SOURCE_TO_SNAKE,
)


_DATE_RE = r"(\d{1,2}/\d{1,2}/\d{2,4})"


def derive_date_from_periods(periods: pd.Series) -> pd.Series:
    """Extract a date from RetailPanel-style period strings and normalize to midnight.

    Expected examples: "1 w/e 01/27/24".

    Returns
    - pandas Series of dtype datetime64[ns], normalized to 00:00:00.
    """

    if periods is None:
        raise ValueError("periods cannot be None")

    extracted = periods.astype("string").str.extract(_DATE_RE, expand=False)

    # Deterministic parsing: M/D/YY is typical; allow M/D/YYYY.
    dt = pd.to_datetime(extracted, format="%m/%d/%y", errors="coerce")
    dt_4 = pd.to_datetime(extracted, format="%m/%d/%Y", errors="coerce")
    dt = dt.fillna(dt_4)

    if dt.isna().any():
        bad_n = int(dt.isna().sum())
        sample = periods[dt.isna()].astype("string").head(3).tolist()
        raise ValueError(
            f"Failed to parse {bad_n} Periods into dates (example Periods={sample})."
        )

    return dt.dt.normalize()


def _validate_required_columns(df: pd.DataFrame, required: Iterable[str], *, label: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{label}: missing required columns after standardization: {missing}")


def _validate_non_negative(df: pd.DataFrame, cols: Iterable[str], *, label: str) -> None:
    for c in cols:
        if c not in df.columns:
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        # allow NaN; enforce no negative real values
        neg = (s < 0) & (~s.isna())
        if neg.any():
            bad_n = int(neg.sum())
            raise ValueError(f"{label}: column '{c}' has {bad_n} negative values")


def _warn_or_fail_if_not_saturday(dates: pd.Series, *, label: str) -> None:
    # Monday=0 ... Saturday=5
    is_sat = dates.dt.weekday == 5
    not_sat = ~is_sat
    not_sat_n = int(not_sat.sum())
    if not_sat_n == 0:
        return

    frac = not_sat_n / max(int(len(dates)), 1)
    sample = dates[not_sat].head(3).dt.strftime("%Y-%m-%d").tolist()

    msg = f"{label}: {not_sat_n} dates are not Saturday (sample={sample})"
    if frac > 0.05:
        raise ValueError(msg + f"; exceeds 5% threshold (frac={frac:.3%})")

    warnings.warn(msg + f"; below 5% threshold (frac={frac:.3%})", stacklevel=2)


def load_panel(path: str | Path | None = None, asof_date: str | None = None) -> pd.DataFrame:
    """Load RetailPanel historical exemplars with standardization + as-of filtering.

    - Renames columns to snake_case.
    - Always derives `date` from `periods`.
    - Validates required columns.
    - Validates non-negative dollars/units/eq.
    - Warns if some dates aren't Saturday; fails if >5% aren't Saturday.
    - Filters to `date <= asof_date` and stores `df.attrs['asof_date']`.
    """

    if path is None:
        path = "./Dataset/Historical_Data.csv"
    
    p = Path(path)
    if not p.exists():
        # fallback to original name if exists (for backward compatibility if user provides it)
        if p.name == "Historical_Data.csv":
            alt = p.parent / "Historical_Data.csv"
            if alt.exists():
                p = alt
        
    if not p.exists():
        raise FileNotFoundError(p)

    if p.suffix.lower() == ".csv":
        df = pd.read_csv(p)
    else:
        df = pd.read_excel(p, engine="openpyxl")

    df = df.rename(columns=SOURCE_TO_SNAKE)

    if "periods" not in df.columns:
        raise ValueError("Panel load: expected a 'Periods' column to derive date")

    df["date"] = derive_date_from_periods(df["periods"])

    # Normalize dtypes
    for col in ["dollars", "units", "eq", "avg_unit_price", "acv_pct", "tdp", "dollars_per_mm_acv", "units_per_mm_acv", "eq_per_mm_acv"]:
        if col in df.columns:
            # robust cleanup for currency/strings
            if df[col].dtype == "object":
                df[col] = (
                    df[col]
                    .astype(str)
                    .str.replace("$", "", regex=False)
                    .str.replace(",", "", regex=False)
                    .str.replace("%", "", regex=False)  # also handle %
                )
            df[col] = pd.to_numeric(df[col], errors="coerce")

    _validate_required_columns(df, PANEL_REQUIRED_COLUMNS, label="Panel load")
    _validate_non_negative(df, ["dollars", "units", "eq"], label="Panel load")
    _warn_or_fail_if_not_saturday(df["date"], label="Panel load")

    if asof_date is None:
        asof_ts = df["date"].max()
    else:
        asof_ts = pd.to_datetime(asof_date, errors="raise").normalize()

    df = df[df["date"] <= asof_ts].copy()
    df = df.sort_values(HIERARCHY_COLS + ["date"], kind="mergesort")
    df.attrs["asof_date"] = asof_ts

    return df


def load_new_innovation(
    path_candidates: list[str | Path] | str | Path | None = None,
) -> pd.DataFrame | None:
    """Load optional early-week input for a single new brand innovation.

    Supports either:
    - ./Dataset/New_Innovations.csv
    - ./Dataset/New_Innovations.csv

    Returns None if no candidate file exists.
    """

    if path_candidates is None:
        path_candidates = [
            "./Dataset/New_Innovations.csv",
            "./Dataset/New_Innovations.csv",
            "./Dataset/New_Innovations.csv",
        ]
    
    if isinstance(path_candidates, (str, Path)):
        path_candidates = [path_candidates]

    chosen: Path | None = None
    for cand in path_candidates:
        p = Path(cand)
        if p.exists() and p.is_file():
            chosen = p
            break

    if chosen is None:
        return None

    if chosen.suffix.lower() == ".csv":
        df = pd.read_csv(chosen)
    else:
        df = pd.read_excel(chosen, engine="openpyxl")

    df = df.rename(columns=SOURCE_TO_SNAKE)

    # Allow the file to provide date directly, but if periods exists we always derive.
    if "periods" in df.columns:
        df["date"] = derive_date_from_periods(df["periods"])
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    else:
        raise ValueError("New_Innovations load: expected 'Periods' or 'date' column")

    # Normalize dtypes
    for col in ["dollars", "units", "eq", "avg_unit_price", "acv_pct", "tdp", "dollars_per_mm_acv", "units_per_mm_acv", "eq_per_mm_acv"]:
        if col in df.columns:
            # robust cleanup for currency/strings
            if df[col].dtype == "object":
                df[col] = (
                    df[col]
                    .astype(str)
                    .str.replace("$", "", regex=False)
                    .str.replace(",", "", regex=False)
                    .str.replace("%", "", regex=False)
                )
            df[col] = pd.to_numeric(df[col], errors="coerce")

    _validate_required_columns(df, NEW_INNOVATION_REQUIRED_COLUMNS, label="New_Innovations load")
    _validate_non_negative(df, ["dollars", "units", "eq"], label="New_Innovations load")
    _warn_or_fail_if_not_saturday(df["date"], label="New_Innovations load")

    df = df.sort_values(HIERARCHY_COLS + ["date"], kind="mergesort")
    df.attrs["asof_date"] = df["date"].max()

    return df


def fingerprint_df(df: pd.DataFrame) -> str:
    """Stable dataset fingerprint.

    Includes:
    - rowcount
    - max date
    - sha256 over sorted key columns (identifiers + numeric)

    The dataframe is sorted by stable keys before hashing to ensure
    deterministic output regardless of input row order.
    """

    # Stable sort keys: use full hierarchy + date
    sort_keys = HIERARCHY_COLS + ["date"]
    sort_cols = [c for c in sort_keys if c in df.columns]

    work = df.copy()
    if sort_cols:
        work = work.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

    rowcount = int(work.shape[0])
    max_date = None
    if "date" in work.columns:
        max_date = pd.to_datetime(work["date"], errors="coerce").max()

    # Columns included in hash: full hierarchy identifiers + key numeric
    id_cols = HIERARCHY_COLS
    num_cols = [
        "dollars",
        "units",
        "eq",
        "acv_pct",
        "tdp",
        "dollars_per_mm_acv",
        "units_per_mm_acv",
        "eq_per_mm_acv",
    ]
    hash_cols = [c for c in id_cols + num_cols if c in work.columns]

    block = work[hash_cols].copy()

    # Normalize dtypes for stability
    for c in block.columns:
        if c in num_cols:
            block[c] = pd.to_numeric(block[c], errors="coerce")
        else:
            block[c] = block[c].astype("string").fillna("")

    h = pd.util.hash_pandas_object(block, index=False)
    digest = hashlib.sha256(h.values.tobytes()).hexdigest()

    max_date_str = "" if max_date is pd.NaT or max_date is None else str(max_date.normalize().date())
    final = hashlib.sha256(f"{rowcount}|{max_date_str}|{digest}".encode("utf-8")).hexdigest()
    return final

