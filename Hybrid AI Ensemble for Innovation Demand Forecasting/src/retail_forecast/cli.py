from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from retail_forecast.analog_forecaster import forecast_new_innovation_analog
from retail_forecast.backtest import BacktestConfig, run_leave_one_brand_out_backtest
from retail_forecast.conformal import (
    ConformalConfig,
    apply_conformal_calibration,
    fit_conformal_deltas,
    load_conformal_deltas,
    save_conformal_deltas,
)
from retail_forecast.curve_library import (
    build_distribution_library,
    build_velocity_library,
    save_curve_libraries,
)
from retail_forecast.decomposition import add_decomposition_columns, estimate_market_scalers
from retail_forecast.ensemble import EnsembleConfig, forecast_new_innovation_hybrid_ensemble
from retail_forecast.io import fingerprint_df, load_new_innovation, load_panel
from retail_forecast.similarity import fit_indices_by_trademark


def _repo_root() -> Path:
    # .../src/retail_forecast/cli.py -> repo root is parents[2]
    return Path(__file__).resolve().parents[2]


def _ensure_artifacts(*, repo_root: Path) -> None:
    artifacts = repo_root / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)

    # scalers
    scaler_path = artifacts / "market_scalers.parquet"
    if not scaler_path.exists():
        panel_df = load_panel(None)
        panel_df = add_decomposition_columns(panel_df)
        scalers = estimate_market_scalers(panel_df)
        scalers.to_parquet(scaler_path, index=False)

    # curve libraries
    dist_path = artifacts / "dist_curve_library.parquet"
    vel_path = artifacts / "vel_curve_library.parquet"
    if (not dist_path.exists()) or (not vel_path.exists()):
        panel_df = load_panel(None)
        panel_df = add_decomposition_columns(panel_df)
        dist_lib = build_distribution_library(panel_df)
        vel_lib = build_velocity_library(panel_df)
        save_curve_libraries(dist_lib, vel_lib, artifacts_dir=artifacts)

    # similarity indices
    sim_root = repo_root / "models" / "similarity"
    if not sim_root.exists() or not any(sim_root.rglob("index.joblib")):
        dist_lib = pd.read_parquet(dist_path)
        vel_lib = pd.read_parquet(vel_path)
        fit_indices_by_trademark(dist_lib, kind="dist", models_root=sim_root)
        fit_indices_by_trademark(vel_lib, kind="vel", models_root=sim_root)


def predict_analog_main() -> None:
    repo_root = _repo_root()

    ap = argparse.ArgumentParser(description="Analog forecast for New_Innovations.")
    ap.add_argument("--horizon", type=int, default=26)
    ap.add_argument("--n-sims", type=int, default=5000)
    ap.add_argument("--top-k", type=int, default=50)
    args = ap.parse_args()

    _ensure_artifacts(repo_root=repo_root)

    df_new = load_new_innovation()
    if df_new is None:
        raise SystemExit(
            "New_Innovations not found. Provide ./Dataset/New_Innovations.csv or ./Dataset/New_Innovations.csv"
        )

    df_new.attrs["artifacts_dir"] = str(repo_root / "artifacts")
    df_new.attrs["models_root"] = str(repo_root / "models" / "similarity")
    df_new.attrs.setdefault("asof_date_used", df_new.attrs.get("asof_date"))

    forecast_df, explain_df = forecast_new_innovation_analog(
        df_new,
        horizon=int(args.horizon),
        early_weeks=4,
        top_k=int(args.top_k),
        n_sims=int(args.n_sims),
    )

    out_dir = repo_root / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_forecast = out_dir / "new_brand_forecast.csv"
    out_explain = out_dir / "explain_neighbors.csv"
    out_forecast_cal = out_dir / "new_brand_forecast_calibrated.csv"

    forecast_df.to_csv(out_forecast, index=False)
    explain_df.to_csv(out_explain, index=False)

    # Calibrated (if deltas exist)
    try:
        fp = fingerprint_df(load_panel(None))
        deltas = load_conformal_deltas(models_dir=repo_root / "models", fingerprint=fp)
        if deltas is not None and not deltas.empty:
            cal = apply_conformal_calibration(forecast_df, deltas, model_type="analog")
            cal.to_csv(out_forecast_cal, index=False)
    except Exception:
        pass

    print("forecast head(5):")
    print(forecast_df.head(5).to_string(index=False))
    print(f"new_brand_forecast_csv={out_forecast}")
    if out_forecast_cal.exists():
        print(f"new_brand_forecast_calibrated_csv={out_forecast_cal}")
    print(f"explain_neighbors_csv={out_explain}")


def predict_ensemble_main() -> None:
    repo_root = _repo_root()

    ap = argparse.ArgumentParser(description="Hybrid ensemble forecast for New_Innovations.")
    ap.add_argument("--horizon", type=int, default=26)
    ap.add_argument("--n-sims", type=int, default=5000)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument(
        "--nf-models-dir",
        type=str,
        default=str(repo_root / "models" / "neuralforecast" / "h26_in26"),
    )
    args = ap.parse_args()

    _ensure_artifacts(repo_root=repo_root)

    df_new = load_new_innovation()
    if df_new is None:
        raise SystemExit(
            "New_Innovations not found. Provide ./Dataset/New_Innovations.csv or ./Dataset/New_Innovations.csv"
        )

    cfg = EnsembleConfig(
        horizon=int(args.horizon),
        n_sims=int(args.n_sims),
        top_k=int(args.top_k),
        nf_models_dir=str(args.nf_models_dir),
        artifacts_dir=str(repo_root / "artifacts"),
        models_root_similarity=str(repo_root / "models" / "similarity"),
    )

    df_new.attrs.setdefault("asof_date_used", df_new.attrs.get("asof_date"))

    forecast_df, explain_df = forecast_new_innovation_hybrid_ensemble(df_new, config=cfg)

    out_dir = repo_root / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_forecast = out_dir / "new_brand_forecast_ensemble.csv"
    out_explain = out_dir / "explain_ensemble.csv"
    out_forecast_cal = out_dir / "new_brand_forecast_ensemble_calibrated.csv"

    forecast_df.to_csv(out_forecast, index=False)
    explain_df.to_csv(out_explain, index=False)

    # Calibrated (if deltas exist)
    try:
        fp = fingerprint_df(load_panel(None))
        deltas = load_conformal_deltas(models_dir=repo_root / "models", fingerprint=fp)
        if deltas is not None and not deltas.empty:
            cal = apply_conformal_calibration(forecast_df, deltas, model_type="ensemble")
            cal.to_csv(out_forecast_cal, index=False)
    except Exception:
        pass

    print("forecast head(5):")
    print(forecast_df.head(5).to_string(index=False))
    print(f"new_brand_forecast_ensemble_csv={out_forecast}")
    if out_forecast_cal.exists():
        print(f"new_brand_forecast_ensemble_calibrated_csv={out_forecast_cal}")
    print(f"explain_ensemble_csv={out_explain}")


def backtest_and_calibrate_main() -> None:
    repo_root = _repo_root()

    ap = argparse.ArgumentParser(description="Run backtest and fit conformal calibration deltas.")
    ap.add_argument("--early-weeks", type=int, default=4)
    ap.add_argument("--horizon", type=int, default=26)
    ap.add_argument("--n-sims", type=int, default=2000)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--include-ensemble", action="store_true")
    ap.add_argument(
        "--nf-models-dir",
        type=str,
        default=str(repo_root / "models" / "neuralforecast" / "h26_in26"),
    )
    ap.add_argument("--min-samples", type=int, default=50)
    ap.add_argument("--per-trademark-min-samples", type=int, default=200)
    args = ap.parse_args()

    bt_cfg = BacktestConfig(
        early_weeks=int(args.early_weeks),
        horizon=int(args.horizon),
        top_k=int(args.top_k),
        n_sims=int(args.n_sims),
        include_ensemble=bool(args.include_ensemble),
        nf_models_dir=str(args.nf_models_dir),
    )

    rows, metrics, fp = run_leave_one_brand_out_backtest(
        panel_path=None,
        config=bt_cfg,
    )

    if rows.empty:
        print("backtest produced 0 rows; nothing to calibrate")
        return

    conf_cfg = ConformalConfig(
        alpha=0.10,
        min_samples=int(args.min_samples),
        per_trademark_min_samples=int(args.per_trademark_min_samples),
    )
    deltas = fit_conformal_deltas(rows, config=conf_cfg)

    out_path = save_conformal_deltas(deltas, models_dir=repo_root / "models", fingerprint=fp)

    out_dir = repo_root / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "backtest_metrics.csv"
    metrics.to_csv(metrics_path, index=False)

    print("deltas head(5):")
    print(deltas.head(5).to_string(index=False))
    print(f"deltas_parquet={out_path}")
    print(f"backtest_metrics_csv={metrics_path}")
