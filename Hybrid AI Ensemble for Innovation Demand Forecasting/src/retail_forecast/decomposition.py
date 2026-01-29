from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd


DistType = Literal["acv", "tdp"]


def add_decomposition_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add standardized decomposition columns.

    Expects (canonical names):
    - dollars, units, eq
    - acv_pct, tdp
    - dollars_per_mm_acv, units_per_mm_acv, eq_per_mm_acv
    - avg_unit_price

    Adds:
    - dist_acv, dist_tdp
    - vel_dollars, vel_units, vel_eq
    """

    required = [
        "dollars",
        "units",
        "eq",
        "tdp",
        "dollars_per_mm_acv",
        "units_per_mm_acv",
        "eq_per_mm_acv",
        "avg_unit_price",
    ]

    # Allow legacy column name 'pct_acv' and normalize it to 'acv_pct'
    df_out = df.copy()
    if "acv_pct" not in df_out.columns and "pct_acv" in df_out.columns:
        df_out = df_out.rename(columns={"pct_acv": "acv_pct"})

    required = [*required, "acv_pct"]
    missing = [c for c in required if c not in df_out.columns]
    if missing:
        raise ValueError(f"add_decomposition_columns: missing required columns: {missing}")

    df_out["dist_acv"] = pd.to_numeric(df_out["acv_pct"], errors="coerce")
    df_out["dist_tdp"] = pd.to_numeric(df_out["tdp"], errors="coerce")

    df_out["vel_dollars"] = pd.to_numeric(df_out["dollars_per_mm_acv"], errors="coerce")
    df_out["vel_units"] = pd.to_numeric(df_out["units_per_mm_acv"], errors="coerce")
    df_out["vel_eq"] = pd.to_numeric(df_out["eq_per_mm_acv"], errors="coerce")

    return df_out


@dataclass(frozen=True)
class RobustStats:
    median: float
    iqr: float
    count: int


def _robust_median_iqr(x: pd.Series) -> RobustStats:
    import numpy as np
    x = pd.to_numeric(x, errors="coerce")
    x = x[(~x.isna()) & np.isfinite(x) & (x > 0)]
    if x.empty:
        return RobustStats(median=float("nan"), iqr=float("nan"), count=0)

    q1 = float(x.quantile(0.25))
    q3 = float(x.quantile(0.75))
    return RobustStats(median=float(x.median()), iqr=q3 - q1, count=int(x.shape[0]))


def estimate_market_scalers(df: pd.DataFrame) -> pd.DataFrame:
    """Estimate market/category scalers using robust medians.

    Goal (using ACV distribution):
      dollars ≈ k_dollars(market, category) * vel_dollars * dist_acv
      units   ≈ k_units(market, category)   * vel_units   * dist_acv
      eq      ≈ k_eq(market, category)      * vel_eq      * dist_acv

    Uses only rows where dist_acv>0 and vel>0.
    Groups by (markets, category).
    """

    required = [
        "markets",
        "category",
        "dollars",
        "units",
        "eq",
        "dist_acv",
        "vel_dollars",
        "vel_units",
        "vel_eq",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"estimate_market_scalers: missing required columns: {missing}")

    base = df.copy()
    base["dist_acv"] = pd.to_numeric(base["dist_acv"], errors="coerce")
    base["vel_dollars"] = pd.to_numeric(base["vel_dollars"], errors="coerce")
    base["vel_units"] = pd.to_numeric(base["vel_units"], errors="coerce")
    base["vel_eq"] = pd.to_numeric(base["vel_eq"], errors="coerce")

    rows = base[(base["dist_acv"] > 0)].copy()

    def compute_group(g: pd.DataFrame) -> pd.Series:
        # Filter vel>0 per metric to avoid div-by-zero / inf
        mask_d = (g["vel_dollars"] > 0) & (g["dist_acv"] > 0)
        mask_u = (g["vel_units"] > 0) & (g["dist_acv"] > 0)
        mask_e = (g["vel_eq"] > 0) & (g["dist_acv"] > 0)

        kd = _robust_median_iqr(
            g.loc[mask_d, "dollars"] / (g.loc[mask_d, "vel_dollars"] * g.loc[mask_d, "dist_acv"])
        )
        ku = _robust_median_iqr(
            g.loc[mask_u, "units"] / (g.loc[mask_u, "vel_units"] * g.loc[mask_u, "dist_acv"])
        )
        ke = _robust_median_iqr(
            g.loc[mask_e, "eq"] / (g.loc[mask_e, "vel_eq"] * g.loc[mask_e, "dist_acv"])
        )
        return pd.Series(
            {
                "k_dollars": kd.median,
                "k_dollars_iqr": kd.iqr,
                "k_dollars_count": kd.count,
                "k_units": ku.median,
                "k_units_iqr": ku.iqr,
                "k_units_count": ku.count,
                "k_eq": ke.median,
                "k_eq_iqr": ke.iqr,
                "k_eq_count": ke.count,
            }
        )

    scaler_df = (
        rows.groupby(["markets", "category"], dropna=False, sort=True)
        .apply(compute_group, include_groups=False)
        .reset_index()
    )

    return scaler_df


def reconstruct_sales(
    df: pd.DataFrame,
    scaler_df: pd.DataFrame,
    use_dist: DistType = "acv",
) -> pd.DataFrame:
    """Reconstruct sales using distribution x velocity x scaler constants."""

    dist_col = "dist_acv" if use_dist == "acv" else "dist_tdp"
    required = [
        "markets",
        "category",
        "dollars",
        "units",
        "eq",
        dist_col,
        "vel_dollars",
        "vel_units",
        "vel_eq",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"reconstruct_sales: missing required columns: {missing}")

    needed_scalers = ["markets", "category", "k_dollars", "k_units", "k_eq"]
    missing_s = [c for c in needed_scalers if c not in scaler_df.columns]
    if missing_s:
        raise ValueError(f"reconstruct_sales: scaler_df missing columns: {missing_s}")

    out = df.merge(
        scaler_df[needed_scalers],
        on=["markets", "category"],
        how="left",
        validate="m:1",
    ).copy()

    out["recon_dollars"] = out["k_dollars"] * out["vel_dollars"] * out[dist_col]
    out["recon_units"] = out["k_units"] * out["vel_units"] * out[dist_col]
    out["recon_eq"] = out["k_eq"] * out["vel_eq"] * out[dist_col]

    # Error ratios vs actual (reconstructed / actual)
    for m in ["dollars", "units", "eq"]:
        denom = pd.to_numeric(out[m], errors="coerce")
        numer = pd.to_numeric(out[f"recon_{m}"], errors="coerce")
        out[f"recon_over_actual_{m}"] = numer / denom

    return out


def _ratio_stats(x: pd.Series) -> dict[str, Any]:
    x = pd.to_numeric(x, errors="coerce")
    x = x[(~x.isna()) & (x > 0)]
    if x.empty:
        return {"count": 0}
    return {
        "count": int(x.shape[0]),
        "median": float(x.median()),
        "p10": float(x.quantile(0.10)),
        "p90": float(x.quantile(0.90)),
        "iqr": float(x.quantile(0.75) - x.quantile(0.25)),
    }


def quality_report(
    df: pd.DataFrame,
    scaler_df: pd.DataFrame,
    outputs_dir: str | Path = "./outputs",
) -> dict[str, Any]:
    """Compute QC checks and write outputs/quality_checks.csv.

    Produces:
    - price consistency check: dollars ≈ units * avg_unit_price
    - reconstruction quality distribution by market/category
    """

    required = ["dollars", "units", "avg_unit_price", "markets", "category"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"quality_report: missing required columns: {missing}")

    recon = reconstruct_sales(df, scaler_df, use_dist="acv")

    units = pd.to_numeric(recon["units"], errors="coerce")
    aup = pd.to_numeric(recon["avg_unit_price"], errors="coerce")
    denom = units * aup
    price_ratio = pd.to_numeric(recon["dollars"], errors="coerce") / denom

    price_stats = _ratio_stats(price_ratio)

    group_cols = ["markets", "category"]
    qc_rows = (
        recon.groupby(group_cols, dropna=False)
        .agg(
            count=("dollars", "size"),
            dollars_ratio_median=("recon_over_actual_dollars", "median"),
            dollars_ratio_p10=("recon_over_actual_dollars", lambda s: float(pd.Series(s).quantile(0.10))),
            dollars_ratio_p90=("recon_over_actual_dollars", lambda s: float(pd.Series(s).quantile(0.90))),
            units_ratio_median=("recon_over_actual_units", "median"),
            units_ratio_p10=("recon_over_actual_units", lambda s: float(pd.Series(s).quantile(0.10))),
            units_ratio_p90=("recon_over_actual_units", lambda s: float(pd.Series(s).quantile(0.90))),
            eq_ratio_median=("recon_over_actual_eq", "median"),
            eq_ratio_p10=("recon_over_actual_eq", lambda s: float(pd.Series(s).quantile(0.10))),
            eq_ratio_p90=("recon_over_actual_eq", lambda s: float(pd.Series(s).quantile(0.90))),
        )
        .reset_index()
        .sort_values(group_cols, kind="mergesort")
    )

    out_dir = Path(outputs_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    qc_path = out_dir / "quality_checks.csv"
    qc_rows.to_csv(qc_path, index=False)

    summary = {
        "price_ratio_stats": price_stats,
        "qc_rows": int(qc_rows.shape[0]),
        "qc_path": str(qc_path),
    }

    return summary
