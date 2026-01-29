from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_ALPHA = 0.10


@dataclass(frozen=True)
class ConformalConfig:
    alpha: float = DEFAULT_ALPHA
    min_samples: int = 50
    per_trademark_min_samples: int = 200


def fit_conformal_deltas(
    backtest_df: pd.DataFrame,
    *,
    config: ConformalConfig = ConformalConfig(),
) -> pd.DataFrame:
    """Fit conformal deltas from backtest results.

    Expected input columns:
    - model_type, metric, horizon_step
    - y_true, p10, p50, p90
    - optional: trademark

    Output columns:
    - model_type, metric, horizon_step, trademark
    - delta_p10, delta_p90
    - n

    Calibration logic (conformalized quantiles for interval):
      Need calibrated interval [p10 - d10, p90 + d90] to achieve ~ (1-alpha) coverage.
      Let err_low  = p10 - y_true, err_high = y_true - p90.
      Choose d10 = quantile(err_low, 1-alpha), d90 = quantile(err_high, 1-alpha).
      These may be negative (narrowing) if intervals are overly wide.
    """

    required = {"model_type", "metric", "horizon_step", "y_true", "p10", "p90"}
    missing = sorted(required - set(backtest_df.columns))
    if missing:
        raise ValueError(f"fit_conformal_deltas: missing columns: {missing}")

    df = backtest_df.copy()
    df["horizon_step"] = pd.to_numeric(df["horizon_step"], errors="coerce").astype("Int64")

    df["y_true"] = pd.to_numeric(df["y_true"], errors="coerce")
    df["p10"] = pd.to_numeric(df["p10"], errors="coerce")
    df["p90"] = pd.to_numeric(df["p90"], errors="coerce")

    df = df.dropna(subset=["horizon_step", "y_true", "p10", "p90"]).copy()

    df["err_low"] = df["p10"] - df["y_true"]
    df["err_high"] = df["y_true"] - df["p90"]

    alpha = float(config.alpha)
    q = float(1.0 - alpha)

    group_cols = ["model_type", "metric", "horizon_step"]
    has_trademark = "trademark" in df.columns

    rows: list[dict[str, Any]] = []

    def agg(g: pd.DataFrame, trademark: str) -> dict[str, Any] | None:
        n = int(g.shape[0])
        if n < int(config.min_samples):
            return None
        d10 = float(pd.to_numeric(g["err_low"], errors="coerce").quantile(q))
        d90 = float(pd.to_numeric(g["err_high"], errors="coerce").quantile(q))
        return {
            "delta_p10": d10,
            "delta_p90": d90,
            "n": n,
            "trademark": trademark,
        }

    # Global deltas
    for (mt, m, hs), g in df.groupby(group_cols, dropna=False, sort=False):
        rec = agg(g, trademark="__GLOBAL__")
        if rec is None:
            continue
        rows.append({"model_type": mt, "metric": m, "horizon_step": int(hs), **rec})

    # Optional per-trademark deltas (only when enough samples)
    if has_trademark:
        for (mt, m, hs, tm), g in df.groupby([*group_cols, "trademark"], dropna=False, sort=False):
            if int(g.shape[0]) < int(config.per_trademark_min_samples):
                continue
            rec = agg(g, trademark=str(tm))
            if rec is None:
                continue
            rows.append({"model_type": mt, "metric": m, "horizon_step": int(hs), **rec})

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out = out.sort_values(["model_type", "metric", "horizon_step", "trademark"], kind="mergesort")
    return out.reset_index(drop=True)


def apply_conformal_calibration(
    forecast_df: pd.DataFrame,
    deltas_df: pd.DataFrame,
    *,
    model_type: str,
    trademark_col: str = "trademark",
    metric_col: str = "metric",
    date_col: str = "week_ending",
    p10_col: str = "p10",
    p50_col: str = "p50",
    p90_col: str = "p90",
) -> pd.DataFrame:
    """Apply conformal deltas to forecast intervals.

    Expects forecast_df to have per-week rows for each metric.
    Computes horizon_step per group as 1..H based on week_ending order.

    Uses per-trademark deltas if present for that trademark; falls back to __GLOBAL__.
    """

    required = {metric_col, date_col, p10_col, p90_col}
    missing = sorted(required - set(forecast_df.columns))
    if missing:
        raise ValueError(f"apply_conformal_calibration: forecast_df missing columns: {missing}")

    needed_deltas = {"model_type", "metric", "horizon_step", "trademark", "delta_p10", "delta_p90"}
    missing_d = sorted(needed_deltas - set(deltas_df.columns))
    if missing_d:
        raise ValueError(f"apply_conformal_calibration: deltas_df missing columns: {missing_d}")

    df = forecast_df.copy()

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.normalize()

    # Determine group keys for horizon stepping
    keys = [c for c in ["markets", trademark_col, "brand", metric_col] if c in df.columns]
    if not keys:
        keys = [metric_col]

    df = df.sort_values([*keys, date_col], kind="mergesort")
    df["horizon_step"] = df.groupby(keys, dropna=False, sort=False).cumcount() + 1

    # Filter deltas to model_type
    dd = deltas_df[deltas_df["model_type"].astype("string") == str(model_type)].copy()
    if dd.empty:
        return df

    # Build merge keys
    def pick_delta_row(row: pd.Series) -> tuple[float, float] | None:
        metric = str(row[metric_col])
        hs = int(row["horizon_step"])
        trademark = str(row.get(trademark_col, "__GLOBAL__"))

        # prefer trademark-specific
        hit = dd[
            (dd["metric"].astype("string") == metric)
            & (dd["horizon_step"].astype(int) == hs)
            & (dd["trademark"].astype("string") == trademark)
        ]
        if not hit.empty:
            r = hit.iloc[0]
            return float(r["delta_p10"]), float(r["delta_p90"])

        hit2 = dd[
            (dd["metric"].astype("string") == metric)
            & (dd["horizon_step"].astype(int) == hs)
            & (dd["trademark"].astype("string") == "__GLOBAL__")
        ]
        if not hit2.empty:
            r = hit2.iloc[0]
            return float(r["delta_p10"]), float(r["delta_p90"])

        return None

    delta_p10: list[float] = []
    delta_p90: list[float] = []
    for _, r in df.iterrows():
        pair = pick_delta_row(r)
        if pair is None:
            delta_p10.append(0.0)
            delta_p90.append(0.0)
        else:
            d10, d90 = pair
            delta_p10.append(float(d10))
            delta_p90.append(float(d90))

    df["delta_p10"] = delta_p10
    df["delta_p90"] = delta_p90

    df[p10_col] = pd.to_numeric(df[p10_col], errors="coerce") - pd.to_numeric(df["delta_p10"], errors="coerce")
    df[p90_col] = pd.to_numeric(df[p90_col], errors="coerce") + pd.to_numeric(df["delta_p90"], errors="coerce")

    # Maintain ordering if p50 exists
    if p50_col in df.columns:
        p50 = pd.to_numeric(df[p50_col], errors="coerce")
        p10 = pd.to_numeric(df[p10_col], errors="coerce")
        p90 = pd.to_numeric(df[p90_col], errors="coerce")
        df[p10_col] = np.minimum(p10, p50)
        df[p90_col] = np.maximum(p90, p50)

    # Non-negative guard for sales-like metrics
    df[p10_col] = pd.to_numeric(df[p10_col], errors="coerce").clip(lower=0)
    df[p90_col] = pd.to_numeric(df[p90_col], errors="coerce").clip(lower=0)

    return df


def save_conformal_deltas(deltas_df: pd.DataFrame, *, models_dir: str | Path, fingerprint: str) -> Path:
    out_dir = Path(models_dir) / "conformal" / str(fingerprint)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "deltas.parquet"
    deltas_df.to_parquet(out_path, index=False)
    return out_path


def load_conformal_deltas(*, models_dir: str | Path, fingerprint: str) -> pd.DataFrame | None:
    p = Path(models_dir) / "conformal" / str(fingerprint) / "deltas.parquet"
    if not p.exists():
        return None
    return pd.read_parquet(p)
