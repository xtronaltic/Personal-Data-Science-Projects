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

from retail_forecast.analog_forecaster import forecast_new_innovation_analog  # noqa: E402
from retail_forecast.conformal import apply_conformal_calibration, load_conformal_deltas  # noqa: E402
from retail_forecast.io import fingerprint_df, load_new_innovation, load_panel  # noqa: E402
from retail_forecast.pipeline import run_full_pipeline  # noqa: E402


def _ensure_artifacts() -> dict:
    """Ensure all artifacts exist using pipeline (rebuild only if panel data changed)."""
    panel_df = load_panel(repo_root / "Dataset" / "Historical_Data.csv")
    result = run_full_pipeline(
        panel_df,
        artifacts_dir=repo_root / "artifacts",
        models_root=repo_root / "models" / "similarity",
        state_path=repo_root / "artifacts" / "state.json",
    )
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Analog forecast for New_Innovations.")
    ap.add_argument("--horizon", type=int, default=26)
    ap.add_argument("--n-sims", type=int, default=5000)
    ap.add_argument("--top-k", type=int, default=50)
    args = ap.parse_args()

    pipeline_result = _ensure_artifacts()
    print("--- Pipeline status ---")
    print(f"scalers_rebuilt: {pipeline_result['scalers_rebuilt']}")
    print(f"curve_libs_rebuilt: {pipeline_result['curve_libs_rebuilt']}")
    print(f"similarity_rebuilt: {pipeline_result['similarity_rebuilt']}")

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
        fp = fingerprint_df(load_panel(repo_root / "Dataset" / "Historical_Data.csv"))
        deltas = load_conformal_deltas(models_dir=repo_root / "models", fingerprint=fp)
        if deltas is not None and not deltas.empty:
            cal = apply_conformal_calibration(forecast_df, deltas, model_type="analog")
            cal.to_csv(out_forecast_cal, index=False)
    except Exception:
        pass

    print("forecast head(10):")
    print(forecast_df.head(10).to_string(index=False))
    print(f"new_brand_forecast_csv={out_forecast}")
    if out_forecast_cal.exists():
        print(f"new_brand_forecast_calibrated_csv={out_forecast_cal}")
    print(f"explain_neighbors_csv={out_explain}")


if __name__ == "__main__":
    main()
