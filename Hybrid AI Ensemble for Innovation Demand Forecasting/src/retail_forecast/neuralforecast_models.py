from __future__ import annotations

import math
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .constants import HIERARCHY_COLS, HIERARCHY_SEPARATOR
from .decomposition import add_decomposition_columns
from .io import derive_date_from_periods, load_new_innovation, load_panel


def _series_id(df: pd.DataFrame) -> pd.Series:
    """Build series_id from full hierarchy columns.
    
    Format: markets||manufacturer||category||trademark||brand
    """
    parts = [df[c].astype("string").fillna("") for c in HIERARCHY_COLS]
    return parts[0].str.cat(parts[1:], sep=HIERARCHY_SEPARATOR)


def _add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    ds = pd.to_datetime(out["ds"], errors="coerce")

    # ISO week of year as cyclical
    week = ds.dt.isocalendar().week.astype("int64")
    angle = 2.0 * math.pi * (week.astype(float) / 52.0)
    out["weekofyear_sin"] = np.sin(angle)
    out["weekofyear_cos"] = np.cos(angle)

    out["month"] = ds.dt.month.astype("int64")
    return out


def _ensure_timeline(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "date" not in out.columns:
        if "periods" not in out.columns:
            raise ValueError("Expected 'date' or 'periods' to build NeuralForecast dataset")
        out["date"] = derive_date_from_periods(out["periods"])

    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()

    # Use full hierarchy columns where available, with fallback
    hierarchy_cols = [c for c in HIERARCHY_COLS if c in out.columns]
    if not hierarchy_cols:
        hierarchy_cols = ["markets", "trademark", "brand"]
    
    req = hierarchy_cols + ["date"]
    missing = [c for c in req if c not in out.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = out.sort_values(hierarchy_cols + ["date"], kind="mergesort")

    launch = (
        out.groupby(hierarchy_cols, dropna=False, sort=False)["date"]
        .min()
        .rename("launch_date")
        .reset_index()
    )
    out = out.merge(launch, on=hierarchy_cols, how="left", validate="m:1")
    out["weeks_since_launch"] = ((out["date"] - out["launch_date"]).dt.days // 7).astype("int64")

    return out


@dataclass(frozen=True)
class NFConfig:
    horizon: int = 26
    input_size: int = 26
    models: tuple[str, ...] = ("NHITS", "PatchTST")
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)
    max_steps: int = 1000
    batch_size: int = 32
    random_seed: int = 0

    def __post_init__(self) -> None:
        qs = tuple(float(q) for q in self.quantiles)
        allowed = {0.1, 0.5, 0.9}
        if any(q not in allowed for q in qs):
            raise ValueError(f"Unsupported quantiles={qs}. Expected subset of {sorted(allowed)}")


@dataclass(frozen=True)
class NFBundle:
    nf: object
    hist_exog: list[str]
    futr_exog: list[str]
    component: str


def _require_neuralforecast() -> None:
    try:
        import neuralforecast  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "NeuralForecast is not installed. Install it in your conda env (llm) first. "
            "Suggested: `pip install neuralforecast` (or pin per your requirements.txt). "
            f"Original error: {e}"
        )


def build_nf_frame(
    df: pd.DataFrame,
    *,
    target: str,
    component: str,
    include_velocity_exog: bool,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Build a NeuralForecast DataFrame.

    Output columns:
    - unique_id, ds, y
    - hist exog and future exog columns

    Exog specs:
    - Velocity targets: historic exog = avg_unit_price, acv_pct, tdp
    - Calendar: weekofyear sin/cos, month
    - weeks_since_launch as future exog
    """

    base = _ensure_timeline(df)
    base = add_decomposition_columns(base)

    # Map component->target column
    if target not in base.columns:
        raise ValueError(f"Target column missing: {target}")

    nf = pd.DataFrame(
        {
            "unique_id": _series_id(base),
            "ds": pd.to_datetime(base["date"], errors="coerce").dt.normalize(),
            "y": pd.to_numeric(base[target], errors="coerce"),
            "weeks_since_launch": pd.to_numeric(base["weeks_since_launch"], errors="coerce").fillna(0).astype("int64"),
        }
    )

    if include_velocity_exog:
        for c in ["avg_unit_price", "acv_pct", "tdp"]:
            nf[c] = pd.to_numeric(base[c], errors="coerce")

    nf = _add_calendar_features(nf)

    # Decide exog lists
    hist_exog: list[str] = []
    futr_exog: list[str] = ["weeks_since_launch", "weekofyear_sin", "weekofyear_cos", "month"]

    if include_velocity_exog:
        # These are treated as future-known only via projection; we provide them in both history and future frames.
        hist_exog.extend(["avg_unit_price", "acv_pct", "tdp"])
        futr_exog.extend(["avg_unit_price", "acv_pct", "tdp"])

    # Clean ordering and missing
    nf = nf.sort_values(["unique_id", "ds"], kind="mergesort")
    return nf, hist_exog, futr_exog


def _future_exog_projection(
    hist: pd.DataFrame,
    *,
    horizon: int,
    include_velocity_exog: bool,
    mild_trend: bool = True,
    trend_cap: float = 0.02,
) -> pd.DataFrame:
    """Create a future dataframe with exogenous columns.

    - Carries forward last observed values.
    - Optionally adds mild trend capped per week.
    """

    out_rows: list[dict[str, object]] = []

    for uid, g in hist.groupby("unique_id", sort=False, dropna=False):
        g = g.sort_values("ds", kind="mergesort")
        last_ds = pd.to_datetime(g["ds"].iloc[-1], errors="coerce").normalize()
        last_wsl = int(pd.to_numeric(g["weeks_since_launch"].iloc[-1], errors="coerce") or 0)

        last_price = float(pd.to_numeric(g.get("avg_unit_price", pd.Series([np.nan])).iloc[-1], errors="coerce")) if include_velocity_exog else float("nan")
        last_acv = float(pd.to_numeric(g.get("acv_pct", pd.Series([np.nan])).iloc[-1], errors="coerce")) if include_velocity_exog else float("nan")
        last_tdp = float(pd.to_numeric(g.get("tdp", pd.Series([np.nan])).iloc[-1], errors="coerce")) if include_velocity_exog else float("nan")

        # Estimate trend from last 4 points if present
        def _trend(col: str) -> float:
            if not include_velocity_exog:
                return 0.0
            s = pd.to_numeric(g[col], errors="coerce").dropna().tail(4)
            if s.shape[0] < 2:
                return 0.0
            # simple last-first per step
            tr = float((s.iloc[-1] - s.iloc[0]) / max(int(s.shape[0]) - 1, 1))
            # cap trend magnitude (relative to last value when possible)
            base = float(s.iloc[-1])
            cap = trend_cap * max(abs(base), 1.0)
            return float(np.clip(tr, -cap, cap))

        tr_price = _trend("avg_unit_price")
        tr_acv = _trend("acv_pct")
        tr_tdp = _trend("tdp")

        for h in range(1, int(horizon) + 1):
            ds = last_ds + pd.to_timedelta(h * 7, unit="D")
            row: dict[str, object] = {
                "unique_id": str(uid),
                "ds": ds,
                "weeks_since_launch": int(last_wsl + h),
            }

            if include_velocity_exog:
                if mild_trend:
                    row["avg_unit_price"] = float(last_price + tr_price * h) if np.isfinite(last_price) else float("nan")
                    row["acv_pct"] = float(last_acv + tr_acv * h) if np.isfinite(last_acv) else float("nan")
                    row["tdp"] = float(last_tdp + tr_tdp * h) if np.isfinite(last_tdp) else float("nan")
                else:
                    row["avg_unit_price"] = last_price
                    row["acv_pct"] = last_acv
                    row["tdp"] = last_tdp

            out_rows.append(row)

    futr = pd.DataFrame(out_rows)
    futr = _add_calendar_features(futr)
    return futr


def train_nf_bundle(
    df_hist: pd.DataFrame,
    *,
    component: str,
    target_col: str,
    config: NFConfig,
    include_velocity_exog: bool,
) -> NFBundle:
    _require_neuralforecast()

    from neuralforecast import NeuralForecast  # type: ignore
    from neuralforecast.losses.pytorch import MQLoss  # type: ignore

    # Models - use NHITS and TFT (both support future exog)
    from neuralforecast.models import NHITS  # type: ignore

    nf_df, hist_exog, futr_exog = build_nf_frame(
        df_hist,
        target=target_col,
        component=component,
        include_velocity_exog=include_velocity_exog,
    )

    loss = MQLoss(quantiles=list(config.quantiles))

    models = []
    for m in config.models:
        name = str(m).upper()
        if name == "NHITS":
            models.append(
                NHITS(
                    h=int(config.horizon),
                    input_size=int(config.input_size),
                    loss=loss,
                    hist_exog_list=hist_exog or None,
                    futr_exog_list=futr_exog or None,
                    max_steps=int(config.max_steps),
                    batch_size=int(config.batch_size),
                    random_seed=int(config.random_seed),
                )
            )
        elif name == "PATCHTST":
            # PatchTST in newer NeuralForecast versions does not support future exog.
            # Use NHITS as a second model instead (it handles future exog well).
            # Note: If you specifically need a transformer, consider TFT or TimesNet.
            models.append(
                NHITS(
                    h=int(config.horizon),
                    input_size=int(config.input_size),
                    loss=loss,
                    hist_exog_list=hist_exog or None,
                    futr_exog_list=futr_exog or None,
                    max_steps=int(config.max_steps),
                    batch_size=int(config.batch_size),
                    random_seed=int(config.random_seed) + 1,  # different seed
                    n_pool_kernel_size=[2, 2, 1],  # different architecture variant
                    alias="NHITS_v2",
                )
            )
        else:
            raise ValueError(f"Unsupported model name: {m}")

    nf = NeuralForecast(models=models, freq="W-SAT")
    nf.fit(df=nf_df)

    return NFBundle(nf=nf, hist_exog=hist_exog, futr_exog=futr_exog, component=component)


def predict_components_for_new_innovation(
    bundle: NFBundle,
    df_new: pd.DataFrame,
    *,
    horizon: int,
    include_velocity_exog: bool,
) -> pd.DataFrame:
    """Run component forecasts for New_Innovations.

    Returns long df with columns:
      run_date, markets, trademark, brand, week_ending, component, p10,p50,p90, method
    """

    # Build history frame from df_new only (early history)
    hist_nf, _, _ = build_nf_frame(
        df_new,
        target={
            "dist_acv": "dist_acv",
            "vel_dollars": "dollars_per_mm_acv",
            "vel_units": "units_per_mm_acv",
            "vel_eq": "eq_per_mm_acv",
        }[bundle.component],
        component=bundle.component,
        include_velocity_exog=include_velocity_exog,
    )

    futr = _future_exog_projection(
        hist_nf,
        horizon=int(horizon),
        include_velocity_exog=include_velocity_exog,
        mild_trend=True,
    )

    # Predict returns wide with model columns; quantiles are appended in col names.
    pred = bundle.nf.predict(df=hist_nf, futr_df=futr)

    # Normalize output into p10/p50/p90
    run_date = str(pd.Timestamp(date.today()).date())

    # Figure out which model columns exist and ensemble them.
    # Expected col formats include e.g. 'NHITS-lo-0.1' depending on versions.
    # We'll search for quantile suffixes.
    # Map output format: lo-80.0 -> p10, median -> p50, hi-80.0 -> p90
    qmap = {
        "lo-": "p10",     # matches 'lo-80.0', 'lo-90', etc.
        "median": "p50",  # matches 'median'
        "hi-": "p90",     # matches 'hi-80.0', 'hi-90', etc.
    }
    # Also support older MQLoss format with explicit quantiles
    qmap_fallback = {"0.1": "p10", "0.5": "p50", "0.9": "p90"}

    # Get hierarchy from unique_id
    # Format: markets||manufacturer||category||trademark||brand
    def split_uid(uid: str) -> dict[str, str]:
        parts = str(uid).split(HIERARCHY_SEPARATOR)
        if len(parts) == 5:
            return {
                "markets": parts[0],
                "manufacturer": parts[1],
                "category": parts[2],
                "trademark": parts[3],
                "brand": parts[4],
            }
        elif len(parts) == 3:
            # Legacy format: trademark||brand||markets
            return {
                "markets": parts[2],
                "manufacturer": "",
                "category": "",
                "trademark": parts[0],
                "brand": parts[1],
            }
        return {c: "" for c in HIERARCHY_COLS}

    rows: list[dict[str, object]] = []

    # Identify quantile columns
    pred_cols = [c for c in pred.columns if c not in {"unique_id", "ds"}]

    # For each quantile, average over all matching columns
    for q_str, out_col in qmap.items():
        match = [c for c in pred_cols if q_str in str(c)]
        if not match:
            # Fallback to MQLoss format
            fb_key = {"p10": "0.1", "p50": "0.5", "p90": "0.9"}.get(out_col, "")
            match = [c for c in pred_cols if fb_key and (fb_key in str(c) or str(c).endswith(fb_key))]
        if match:
            # Convert columns to numeric and average across models
            numeric_cols = pred[match].apply(pd.to_numeric, errors="coerce")
            pred[out_col] = numeric_cols.mean(axis=1)
        else:
            pred[out_col] = np.nan

    pred = pred[["unique_id", "ds", "p10", "p50", "p90"]].copy()

    for _, r in pred.iterrows():
        uid_parts = split_uid(str(r["unique_id"]))
        rows.append(
            {
                "run_date": run_date,
                "markets": uid_parts["markets"],
                "manufacturer": uid_parts["manufacturer"],
                "category": uid_parts["category"],
                "trademark": uid_parts["trademark"],
                "brand": uid_parts["brand"],
                "week_ending": str(pd.to_datetime(r["ds"], errors="coerce").date()),
                "component": bundle.component,
                "p10": float(pd.to_numeric(r["p10"], errors="coerce")) if pd.notna(r["p10"]) else float("nan"),
                "p50": float(pd.to_numeric(r["p50"], errors="coerce")) if pd.notna(r["p50"]) else float("nan"),
                "p90": float(pd.to_numeric(r["p90"], errors="coerce")) if pd.notna(r["p90"]) else float("nan"),
                "method": "neuralforecast",
            }
        )

    return pd.DataFrame(rows)


def train_all_components(
    *,
    panel_path: str | Path = "./Dataset/Historical_Data.csv",
    config: NFConfig = NFConfig(),
) -> dict[str, NFBundle]:
    """Train NeuralForecast models for all configured components."""

    df_hist = load_panel(panel_path)
    df_hist = add_decomposition_columns(df_hist)

    bundles: dict[str, NFBundle] = {}

    # dist_acv uses acv_pct via dist_acv
    bundles["dist_acv"] = train_nf_bundle(
        df_hist,
        component="dist_acv",
        target_col="dist_acv",
        config=config,
        include_velocity_exog=False,
    )

    # velocity components use per-mm ACV targets; include exog
    bundles["vel_dollars"] = train_nf_bundle(
        df_hist,
        component="vel_dollars",
        target_col="dollars_per_mm_acv",
        config=config,
        include_velocity_exog=True,
    )
    bundles["vel_units"] = train_nf_bundle(
        df_hist,
        component="vel_units",
        target_col="units_per_mm_acv",
        config=config,
        include_velocity_exog=True,
    )
    bundles["vel_eq"] = train_nf_bundle(
        df_hist,
        component="vel_eq",
        target_col="eq_per_mm_acv",
        config=config,
        include_velocity_exog=True,
    )

    return bundles


def save_bundles(bundles: dict[str, NFBundle], out_dir: str | Path) -> None:
    """Persist bundles using NeuralForecast's save when available."""

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for name, b in bundles.items():
        comp_dir = out / name
        comp_dir.mkdir(parents=True, exist_ok=True)

        nf = b.nf
        # NeuralForecast has a .save API in most versions.
        if hasattr(nf, "save"):
            nf.save(path=str(comp_dir), overwrite=True)  # type: ignore
        else:  # pragma: no cover
            import joblib

            joblib.dump(nf, comp_dir / "nf.joblib")

        meta = {
            "component": b.component,
            "hist_exog": b.hist_exog,
            "futr_exog": b.futr_exog,
        }
        (comp_dir / "bundle_meta.json").write_text(json.dumps(meta), encoding="utf-8")


def load_bundle(component: str, models_dir: str | Path) -> NFBundle:
    _require_neuralforecast()

    from neuralforecast import NeuralForecast  # type: ignore

    comp_dir = Path(models_dir) / component
    if not comp_dir.exists():
        raise FileNotFoundError(comp_dir)

    meta_path = comp_dir / "bundle_meta.json"
    meta: dict[str, object] = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))

    if hasattr(NeuralForecast, "load"):
        nf = NeuralForecast.load(path=str(comp_dir))  # type: ignore
    else:  # pragma: no cover
        import joblib

        nf = joblib.load(comp_dir / "nf.joblib")

    hist_exog = list(meta.get("hist_exog", [])) if isinstance(meta.get("hist_exog", []), list) else []
    futr_exog = list(meta.get("futr_exog", [])) if isinstance(meta.get("futr_exog", []), list) else []

    return NFBundle(nf=nf, hist_exog=hist_exog, futr_exog=futr_exog, component=str(component))
