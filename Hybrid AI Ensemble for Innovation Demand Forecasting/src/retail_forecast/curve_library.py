from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from .constants import HIERARCHY_COLS, HIERARCHY_SEPARATOR


HORIZON_WEEKS_DEFAULT = 26


def _series_id(df: pd.DataFrame) -> pd.Series:
    """Build series_id from full hierarchy: markets||manufacturer||category||trademark||brand"""
    parts = []
    for col in HIERARCHY_COLS:
        if col in df.columns:
            parts.append(df[col].astype("string").fillna(""))
        else:
            parts.append(pd.Series([""] * len(df), index=df.index))
    return parts[0].str.cat(parts[1:], sep=HIERARCHY_SEPARATOR)


def _collapse_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure one row per (series_id, date) with sensible aggregations."""

    required = HIERARCHY_COLS + ["date"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"curve_library: missing required columns: {missing}")

    out = df.copy()
    out["series_id"] = _series_id(out)
    out["date"] = pd.to_datetime(out["date"], errors="raise").dt.normalize()

    sum_cols = [c for c in ["dollars", "units", "eq"] if c in out.columns]
    mean_cols = [
        c
        for c in [
            "acv_pct",
            "tdp",
            "dist_acv",
            "dist_tdp",
            "dollars_per_mm_acv",
            "units_per_mm_acv",
            "eq_per_mm_acv",
            "vel_dollars",
            "vel_units",
            "vel_eq",
            "avg_unit_price",
        ]
        if c in out.columns
    ]

    agg: dict[str, str] = {}
    for c in sum_cols:
        agg[c] = "sum"
    for c in mean_cols:
        agg[c] = "mean"

    keys = ["series_id"] + HIERARCHY_COLS + ["date"]
    out = out.groupby(keys, dropna=False, sort=False).agg(agg).reset_index()
    return out


def _find_launch_dates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in ["dollars", "units", "eq", "acv_pct", "tdp", "dist_acv", "dist_tdp"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    # Prefer dist_* when present; fall back to acv_pct/tdp.
    dist_acv_col = "dist_acv" if "dist_acv" in out.columns else "acv_pct"
    dist_tdp_col = "dist_tdp" if "dist_tdp" in out.columns else "tdp"

    nonzero = (
        (out.get("dollars", 0) > 0)
        | (out.get("units", 0) > 0)
        | (out.get("eq", 0) > 0)
        | (out.get(dist_acv_col, 0) > 0)
        | (out.get(dist_tdp_col, 0) > 0)
    )

    first = (
        out.loc[nonzero, ["series_id", "date"]]
        .sort_values(["series_id", "date"], kind="mergesort")
        .groupby("series_id", sort=False, dropna=False)["date"]
        .min()
        .rename("launch_date")
        .reset_index()
    )

    return first


def _add_weeks_since_launch(df: pd.DataFrame, launch_df: pd.DataFrame) -> pd.DataFrame:
    out = df.merge(launch_df, on="series_id", how="inner", validate="m:1").copy()
    out["weeks_since_launch"] = ((out["date"] - out["launch_date"]).dt.days // 7).astype("int64")
    return out


def _vector_for_weeks(g: pd.DataFrame, value_col: str, length: int) -> list[float | None]:
    s = (
        g.set_index("weeks_since_launch")[value_col]
        .reindex(range(length))
        .astype("float")
        .tolist()
    )
    # Convert NaN to None for JSON stability
    out: list[float | None] = []
    for v in s:
        if v != v:  # NaN check
            out.append(None)
        else:
            out.append(float(v))
    return out


def _slope(vec: list[float | None]) -> float | None:
    clean = [v for v in vec if v is not None]
    if len(clean) < 2:
        return None
    first = vec[0]
    last = vec[len(vec) - 1]
    if first is None or last is None:
        return None
    if len(vec) <= 1:
        return None
    return float((last - first) / (len(vec) - 1))


def _normalize_peak(vec: list[float | None]) -> list[float | None]:
    clean = [v for v in vec if v is not None]
    if not clean:
        return [None for _ in vec]
    peak = max(clean)
    if peak <= 0:
        return [None for _ in vec]
    return [None if v is None else float(v / peak) for v in vec]


def _require_min_weeks(g: pd.DataFrame, min_weeks: int) -> bool:
    # Require that we have at least min_weeks rows within the window (0..)
    return int(g["weeks_since_launch"].nunique()) >= int(min_weeks)


def build_distribution_library(
    df: pd.DataFrame,
    min_weeks: int = 12,
    max_weeks: int = 52,
    early_weeks: int = 4,
) -> pd.DataFrame:
    """Build distribution curve library aligned by launch week.

    Targets:
    - dist_acv (uses dist_acv if present else acv_pct)
    - dist_tdp (uses dist_tdp if present else tdp)

    Stores:
    - early vectors dist_acv_0..{early_weeks-1}, dist_tdp_0..{early_weeks-1}
    - slopes
    - normalized shapes (peak-normalized)
    - future vectors (length = early_weeks + 14) as JSON strings
    """

    base = _collapse_weekly(df)
    launch = _find_launch_dates(base)
    aligned = _add_weeks_since_launch(base, launch)

    aligned = aligned[(aligned["weeks_since_launch"] >= 0) & (aligned["weeks_since_launch"] < max_weeks)].copy()

    dist_acv_col = "dist_acv" if "dist_acv" in aligned.columns else "acv_pct"
    dist_tdp_col = "dist_tdp" if "dist_tdp" in aligned.columns else "tdp"

    aligned[dist_acv_col] = pd.to_numeric(aligned.get(dist_acv_col), errors="coerce")
    aligned[dist_tdp_col] = pd.to_numeric(aligned.get(dist_tdp_col), errors="coerce")

    future_len = early_weeks + HORIZON_WEEKS_DEFAULT

    rows: list[dict[str, Any]] = []
    for sid, g in aligned.groupby("series_id", sort=False, dropna=False):
        if not _require_min_weeks(g, min_weeks=min_weeks):
            continue

        g = g.sort_values("weeks_since_launch", kind="mergesort")
        early_acv = _vector_for_weeks(g, dist_acv_col, early_weeks)
        early_tdp = _vector_for_weeks(g, dist_tdp_col, early_weeks)

        future_acv = _vector_for_weeks(g, dist_acv_col, future_len)
        future_tdp = _vector_for_weeks(g, dist_tdp_col, future_len)

        row: dict[str, Any] = {
            "series_id": str(sid),
            "markets": str(g["markets"].iloc[0]) if "markets" in g.columns else "",
            "manufacturer": str(g["manufacturer"].iloc[0]) if "manufacturer" in g.columns else "",
            "category": str(g["category"].iloc[0]) if "category" in g.columns else "",
            "trademark": str(g["trademark"].iloc[0]) if "trademark" in g.columns else "",
            "brand": str(g["brand"].iloc[0]) if "brand" in g.columns else "",
            "launch_date": str(pd.to_datetime(g["launch_date"].iloc[0]).date()),
            "n_weeks_observed": int(g["weeks_since_launch"].nunique()),
            "max_week": int(g["weeks_since_launch"].max()),
            "early_slope_dist_acv": _slope(early_acv),
            "early_slope_dist_tdp": _slope(early_tdp),
            "future_dist_acv_json": json.dumps(future_acv),
            "future_dist_tdp_json": json.dumps(future_tdp),
            "future_dist_acv_norm_json": json.dumps(_normalize_peak(future_acv)),
            "future_dist_tdp_norm_json": json.dumps(_normalize_peak(future_tdp)),
        }

        for i in range(early_weeks):
            row[f"dist_acv_{i}"] = early_acv[i]
            row[f"dist_tdp_{i}"] = early_tdp[i]

        rows.append(row)

    return pd.DataFrame(rows)


def build_velocity_library(
    df: pd.DataFrame,
    min_weeks: int = 12,
    max_weeks: int = 52,
    early_weeks: int = 4,
) -> pd.DataFrame:
    """Build velocity curve library aligned by launch week.

    Targets:
    - vel_dollars, vel_units, vel_eq (expects these columns exist)

    Stores:
    - early vectors vel_*_0..{early_weeks-1}
    - slopes
    - normalized shapes (peak-normalized) and raw future vectors as JSON strings
    - price context: early_mean_avg_unit_price
    - distribution context: early_mean_acv_pct, early_mean_tdp
    """

    base = _collapse_weekly(df)
    launch = _find_launch_dates(base)
    aligned = _add_weeks_since_launch(base, launch)

    aligned = aligned[(aligned["weeks_since_launch"] >= 0) & (aligned["weeks_since_launch"] < max_weeks)].copy()

    required = ["vel_dollars", "vel_units", "vel_eq"]
    missing = [c for c in required if c not in aligned.columns]
    if missing:
        raise ValueError(
            "build_velocity_library: missing required decomposition columns. "
            "Run add_decomposition_columns() first. "
            f"Missing={missing}"
        )

    for c in ["vel_dollars", "vel_units", "vel_eq", "avg_unit_price", "acv_pct", "tdp"]:
        if c in aligned.columns:
            aligned[c] = pd.to_numeric(aligned[c], errors="coerce")

    future_len = early_weeks + HORIZON_WEEKS_DEFAULT

    rows: list[dict[str, Any]] = []
    for sid, g in aligned.groupby("series_id", sort=False, dropna=False):
        if not _require_min_weeks(g, min_weeks=min_weeks):
            continue

        g = g.sort_values("weeks_since_launch", kind="mergesort")

        early_vd = _vector_for_weeks(g, "vel_dollars", early_weeks)
        early_vu = _vector_for_weeks(g, "vel_units", early_weeks)
        early_ve = _vector_for_weeks(g, "vel_eq", early_weeks)

        future_vd = _vector_for_weeks(g, "vel_dollars", future_len)
        future_vu = _vector_for_weeks(g, "vel_units", future_len)
        future_ve = _vector_for_weeks(g, "vel_eq", future_len)

        # Context features from early window
        early_mask = g["weeks_since_launch"].between(0, early_weeks - 1)
        early_mean_avg_unit_price = float(g.loc[early_mask, "avg_unit_price"].mean()) if "avg_unit_price" in g.columns else None
        early_mean_acv_pct = float(g.loc[early_mask, "acv_pct"].mean()) if "acv_pct" in g.columns else None
        early_mean_tdp = float(g.loc[early_mask, "tdp"].mean()) if "tdp" in g.columns else None

        row: dict[str, Any] = {
            "series_id": str(sid),
            "markets": str(g["markets"].iloc[0]) if "markets" in g.columns else "",
            "manufacturer": str(g["manufacturer"].iloc[0]) if "manufacturer" in g.columns else "",
            "category": str(g["category"].iloc[0]) if "category" in g.columns else "",
            "trademark": str(g["trademark"].iloc[0]) if "trademark" in g.columns else "",
            "brand": str(g["brand"].iloc[0]) if "brand" in g.columns else "",
            "launch_date": str(pd.to_datetime(g["launch_date"].iloc[0]).date()),
            "n_weeks_observed": int(g["weeks_since_launch"].nunique()),
            "max_week": int(g["weeks_since_launch"].max()),
            "early_slope_vel_dollars": _slope(early_vd),
            "early_slope_vel_units": _slope(early_vu),
            "early_slope_vel_eq": _slope(early_ve),
            "early_mean_avg_unit_price": early_mean_avg_unit_price,
            "early_mean_acv_pct": early_mean_acv_pct,
            "early_mean_tdp": early_mean_tdp,
            "future_vel_dollars_json": json.dumps(future_vd),
            "future_vel_units_json": json.dumps(future_vu),
            "future_vel_eq_json": json.dumps(future_ve),
            "future_vel_dollars_norm_json": json.dumps(_normalize_peak(future_vd)),
            "future_vel_units_norm_json": json.dumps(_normalize_peak(future_vu)),
            "future_vel_eq_norm_json": json.dumps(_normalize_peak(future_ve)),
        }

        for i in range(early_weeks):
            row[f"vel_dollars_{i}"] = early_vd[i]
            row[f"vel_units_{i}"] = early_vu[i]
            row[f"vel_eq_{i}"] = early_ve[i]

        rows.append(row)

    return pd.DataFrame(rows)


def save_curve_libraries(
    dist_lib: pd.DataFrame,
    vel_lib: pd.DataFrame,
    artifacts_dir: str | Path = "./artifacts",
) -> tuple[Path, Path]:
    out_dir = Path(artifacts_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dist_path = out_dir / "dist_curve_library.parquet"
    vel_path = out_dir / "vel_curve_library.parquet"

    dist_lib.to_parquet(dist_path, index=False)
    vel_lib.to_parquet(vel_path, index=False)

    return dist_path, vel_path
