from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from .analog_forecaster import forecast_new_innovation_analog
from .constants import HIERARCHY_COLS, HIERARCHY_SEPARATOR
from .curve_library import build_distribution_library, build_velocity_library, save_curve_libraries
from .decomposition import add_decomposition_columns, estimate_market_scalers
from .ensemble import EnsembleConfig, forecast_new_innovation_hybrid_ensemble
from .io import fingerprint_df, load_panel
from .similarity import fit_indices_by_hierarchy


ModelType = Literal["analog", "ensemble"]


@dataclass(frozen=True)
class BacktestConfig:
    early_weeks: int = 4
    horizon: int = 26
    top_k: int = 50
    n_sims: int = 2000
    include_ensemble: bool = True
    nf_models_dir: str = "./models/neuralforecast/h26_in26"
    use_timesfm_hybrid: bool = False  # Use TimesFM + analog hybrid instead of NF ensemble


def _series_id(df: pd.DataFrame) -> pd.Series:
    """Build series_id from full hierarchy columns."""
    parts = [df[c].astype("string").fillna("") for c in HIERARCHY_COLS]
    return parts[0].str.cat(parts[1:], sep=HIERARCHY_SEPARATOR)


def _pinball(y: np.ndarray, q: np.ndarray, tau: float) -> np.ndarray:
    # pinball loss for quantile tau
    u = y - q
    return np.maximum(tau * u, (tau - 1.0) * u)


def _compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute WAPE (p50), pinball (p10/p50/p90), coverage for each horizon_step."""

    required = {"model_type", "metric", "horizon_step", "y_true", "p10", "p50", "p90"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"_compute_metrics: missing columns: {missing}")

    base = df.copy()

    for c in ["y_true", "p10", "p50", "p90"]:
        base[c] = pd.to_numeric(base[c], errors="coerce")

    base["abs_err_p50"] = (base["y_true"] - base["p50"]).abs()
    # MAPE per observation (avoid div-by-zero with small epsilon)
    base["ape_p50"] = base["abs_err_p50"] / base["y_true"].abs().clip(lower=1e-9)
    base["pinball_0.1"] = _pinball(base["y_true"].to_numpy(), base["p10"].to_numpy(), 0.1)
    base["pinball_0.5"] = _pinball(base["y_true"].to_numpy(), base["p50"].to_numpy(), 0.5)
    base["pinball_0.9"] = _pinball(base["y_true"].to_numpy(), base["p90"].to_numpy(), 0.9)

    base["covered_10_90"] = (base["y_true"] >= base["p10"]) & (base["y_true"] <= base["p90"])

    def agg(g: pd.DataFrame) -> pd.Series:
        denom = float(pd.to_numeric(g["y_true"], errors="coerce").abs().sum())
        # WAPE / WMAPE: Σ|error| / Σ|actual| (volume-weighted, standard for retail forecasting)
        wmape = float(pd.to_numeric(g["abs_err_p50"], errors="coerce").sum()) / max(denom, 1e-9)
        # MAPE: mean of individual percentage errors (unweighted)
        mape = float(pd.to_numeric(g["ape_p50"], errors="coerce").mean())
        return pd.Series(
            {
                "n": int(g.shape[0]),
                "wmape_p50": wmape,  # Weighted MAPE (industry standard)
                "wape_p50": wmape,   # Alias for backward compatibility
                "mape_p50": mape,    # Unweighted MAPE
                "pinball_0.1": float(pd.to_numeric(g["pinball_0.1"], errors="coerce").mean()),
                "pinball_0.5": float(pd.to_numeric(g["pinball_0.5"], errors="coerce").mean()),
                "pinball_0.9": float(pd.to_numeric(g["pinball_0.9"], errors="coerce").mean()),
                "coverage_10_90": float(pd.to_numeric(g["covered_10_90"], errors="coerce").mean()),
            }
        )

    out = (
        base.groupby(["model_type", "metric", "horizon_step"], dropna=False, sort=False)
        .apply(agg, include_groups=False)
        .reset_index()
        .sort_values(["model_type", "metric", "horizon_step"], kind="mergesort")
    )
    return out


def _prepare_fold_artifacts(
    df_train: pd.DataFrame,
    *,
    artifacts_dir: Path,
    models_dir: Path,
) -> None:
    import json

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    train = add_decomposition_columns(df_train)

    # scalers
    scalers = estimate_market_scalers(train)
    scalers.to_parquet(artifacts_dir / "market_scalers.parquet", index=False)

    # curve libs
    dist_lib = build_distribution_library(train)
    vel_lib = build_velocity_library(train)
    save_curve_libraries(dist_lib, vel_lib, artifacts_dir=artifacts_dir)

    # similarity - capture fingerprints
    similarity_root = models_dir / "similarity"
    dist_result = fit_indices_by_hierarchy(dist_lib, kind="dist", models_root=similarity_root)
    vel_result = fit_indices_by_hierarchy(vel_lib, kind="vel", models_root=similarity_root)

    # Write latest.json so analog_forecaster can find the fingerprints
    latest_json = similarity_root / "latest.json"
    similarity_root.mkdir(parents=True, exist_ok=True)
    latest_json.write_text(
        json.dumps({
            "dist_fingerprint": dist_result["fingerprint"],
            "vel_fingerprint": vel_result["fingerprint"],
        }, indent=2),
        encoding="utf-8",
    )


def _build_df_new_from_brand(df_brand: pd.DataFrame, early_weeks: int) -> pd.DataFrame:
    # Keep only first `early_weeks` weeks per market for the heldout brand.
    dfb = df_brand.copy()
    dfb["date"] = pd.to_datetime(dfb["date"], errors="coerce").dt.normalize()

    rows: list[pd.DataFrame] = []
    for group_values, g in dfb.groupby(HIERARCHY_COLS, dropna=False, sort=False):
        g = g.sort_values("date", kind="mergesort")
        g = g.head(int(early_weeks)).copy()
        rows.append(g)

    out = pd.concat(rows, ignore_index=True) if rows else dfb.head(0)
    out.attrs["asof_date"] = out["date"].max() if "date" in out.columns and not out.empty else None
    return out


def _extract_actuals(df_brand: pd.DataFrame) -> pd.DataFrame:
    g = df_brand.copy()
    g["week_ending"] = pd.to_datetime(g["date"], errors="coerce").dt.normalize().dt.date.astype("string")
    cols = HIERARCHY_COLS + ["week_ending", "dollars", "units", "eq"]
    return g[cols].copy()


def run_leave_one_brand_out_backtest(
    *,
    panel_path: str | Path | None = None,
    config: BacktestConfig = BacktestConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Run leave-one-brand-out within trademark backtest.

    Returns:
      - backtest_rows: per-observation rows with y_true and forecast quantiles (used for conformal)
      - metrics: aggregated metrics by horizon step
      - fingerprint: panel fingerprint used for storing conformal deltas
    """

    df = load_panel(panel_path)
    fp = fingerprint_df(df)

    required = {"markets", "trademark", "brand", "date", "dollars", "units", "eq"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Panel missing required columns for backtest: {missing}")

    out_rows: list[dict[str, Any]] = []

    # Iterate trademarks and brands
    for tm, g_tm in df.groupby("trademark", dropna=False, sort=False):
        tm_s = str(tm)
        brands = g_tm["brand"].astype("string").dropna().unique().tolist()
        for br in brands:
            br_s = str(br)

            df_brand = g_tm[g_tm["brand"].astype("string") == br_s].copy()
            df_train = df[(df["trademark"].astype("string") == tm_s) & (df["brand"].astype("string") != br_s)].copy()

            if df_train.empty or df_brand.empty:
                continue

            df_new = _build_df_new_from_brand(df_brand, early_weeks=int(config.early_weeks))
            if df_new.empty:
                continue

            actuals = _extract_actuals(df_brand)

            # Build fold artifacts in a temp directory
            with tempfile.TemporaryDirectory(prefix="panel_backtest_") as td:
                td_path = Path(td)
                artifacts_dir = td_path / "artifacts"
                models_dir = td_path / "models"

                _prepare_fold_artifacts(df_train, artifacts_dir=artifacts_dir, models_dir=models_dir)

                # ANALOG
                df_new.attrs["artifacts_dir"] = str(artifacts_dir)
                df_new.attrs["models_root"] = str(models_dir / "similarity")
                df_new.attrs["asof_date_used"] = df_new.attrs.get("asof_date")

                forecast_a, _explain = forecast_new_innovation_analog(
                    df_new,
                    horizon=int(config.horizon),
                    early_weeks=int(config.early_weeks),
                    top_k=int(config.top_k),
                    n_sims=int(config.n_sims),
                )

                out_rows.extend(_join_forecast_to_actuals(forecast_a, actuals, model_type="analog"))

                # ENSEMBLE (optional)
                if config.include_ensemble:
                    try:
                        ens_cfg = EnsembleConfig(
                            horizon=int(config.horizon),
                            n_sims=int(config.n_sims),
                            top_k=int(config.top_k),
                            nf_models_dir=str(config.nf_models_dir),
                            artifacts_dir=str(artifacts_dir),
                            models_root_similarity=str(models_dir / "similarity"),
                        )
                        forecast_e, _explain_e = forecast_new_innovation_hybrid_ensemble(df_new, config=ens_cfg)
                        out_rows.extend(_join_forecast_to_actuals(forecast_e, actuals, model_type="ensemble"))
                    except Exception:
                        # Skip ensemble if NF isn't available
                        pass

    backtest_rows = pd.DataFrame(out_rows)
    if backtest_rows.empty:
        return backtest_rows, backtest_rows, fp

    metrics = _compute_metrics(backtest_rows)
    return backtest_rows, metrics, fp


def _join_forecast_to_actuals(forecast_df: pd.DataFrame, actuals_df: pd.DataFrame, *, model_type: str) -> list[dict[str, Any]]:
    required_f = set(HIERARCHY_COLS) | {"week_ending", "metric", "p10", "p50", "p90"}
    if not required_f.issubset(set(forecast_df.columns)):
        return []

    f = forecast_df.copy()
    f["week_ending"] = f["week_ending"].astype("string")

    a = actuals_df.copy()
    a["week_ending"] = a["week_ending"].astype("string")

    merge_keys = HIERARCHY_COLS + ["week_ending"]
    merged = f.merge(
        a,
        on=merge_keys,
        how="left",
        validate="m:1",
    )

    # horizon_step per group
    merged["week_ending_dt"] = pd.to_datetime(merged["week_ending"], errors="coerce")
    keys = HIERARCHY_COLS + ["metric"]
    merged = merged.sort_values([*keys, "week_ending_dt"], kind="mergesort")
    merged["horizon_step"] = merged.groupby(keys, dropna=False, sort=False).cumcount() + 1

    out: list[dict[str, Any]] = []
    for _, r in merged.iterrows():
        metric = str(r["metric"])
        y_true = None
        if metric in {"dollars", "units", "eq"}:
            y_true = r.get(metric)
        if y_true is None or (isinstance(y_true, float) and np.isnan(y_true)):
            continue

        row_dict = {
            "model_type": str(model_type),
            "metric": metric,
            "horizon_step": int(r["horizon_step"]),
            "week_ending": str(r["week_ending"]),
            "y_true": float(pd.to_numeric(y_true, errors="coerce")),
            "p10": float(pd.to_numeric(r.get("p10"), errors="coerce")),
            "p50": float(pd.to_numeric(r.get("p50"), errors="coerce")),
            "p90": float(pd.to_numeric(r.get("p90"), errors="coerce")),
        }
        # Add all hierarchy columns
        for c in HIERARCHY_COLS:
            row_dict[c] = str(r.get(c, ""))

        out.append(row_dict)

    return out
