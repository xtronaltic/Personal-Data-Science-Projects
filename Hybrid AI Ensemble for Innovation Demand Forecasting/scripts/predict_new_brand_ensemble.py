from __future__ import annotations

from pathlib import Path
import sys


def _ensure_src_on_path() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    return repo_root


repo_root = _ensure_src_on_path()

import argparse  # noqa: E402

import pandas as pd  # noqa: E402

from retail_forecast.curve_library import (  # noqa: E402
    build_distribution_library,
    build_velocity_library,
    save_curve_libraries,
)
from retail_forecast.decomposition import add_decomposition_columns, estimate_market_scalers  # noqa: E402
from retail_forecast.ensemble import EnsembleConfig, forecast_new_innovation_hybrid_ensemble  # noqa: E402
from retail_forecast.io import load_new_innovation, load_panel  # noqa: E402
from retail_forecast.similarity import fit_indices_by_trademark  # noqa: E402
from retail_forecast.conformal import apply_conformal_calibration, load_conformal_deltas  # noqa: E402
from retail_forecast.io import fingerprint_df  # noqa: E402


def _ensure_artifacts() -> None:
    artifacts = repo_root / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)

    # scalers
    scaler_path = artifacts / "market_scalers.parquet"
    if not scaler_path.exists():
        panel_df = load_panel(repo_root / "Dataset" / "Historical_Data.csv")
        panel_df = add_decomposition_columns(panel_df)
        scalers = estimate_market_scalers(panel_df)
        scalers.to_parquet(scaler_path, index=False)

    # curve libraries
    dist_path = artifacts / "dist_curve_library.parquet"
    vel_path = artifacts / "vel_curve_library.parquet"
    if (not dist_path.exists()) or (not vel_path.exists()):
        panel_df = load_panel(repo_root / "Dataset" / "Historical_Data.csv")
        panel_df = add_decomposition_columns(panel_df)
        dist_lib = build_distribution_library(panel_df)
        vel_lib = build_velocity_library(panel_df)
        save_curve_libraries(dist_lib, vel_lib, artifacts_dir=artifacts)

    # similarity indices
    sim_root = repo_root / "models" / "similarity"
    # If empty, train
    if not sim_root.exists() or not any(sim_root.rglob("index.joblib")):
        dist_lib = pd.read_parquet(dist_path)
        vel_lib = pd.read_parquet(vel_path)
        fit_indices_by_trademark(dist_lib, kind="dist", models_root=sim_root)
        fit_indices_by_trademark(vel_lib, kind="vel", models_root=sim_root)


def main() -> None:
    ap = argparse.ArgumentParser(description="Hybrid ensemble forecast for New_Innovations.")
    ap.add_argument("--horizon", type=int, default=26)
    ap.add_argument("--n-sims", type=int, default=5000)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--nf-models-dir", type=str, default=str(repo_root / "models" / "neuralforecast" / "h26_in26"))
    args = ap.parse_args()

    _ensure_artifacts()

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

    # Attach aof_date_used for traceability when caller sets it
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
        fp = fingerprint_df(load_panel(repo_root / "Dataset" / "Historical_Data.csv"))
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


if __name__ == "__main__":
    main()
