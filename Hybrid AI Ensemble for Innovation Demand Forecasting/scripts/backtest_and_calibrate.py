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

from retail_forecast.backtest import BacktestConfig, run_leave_one_brand_out_backtest  # noqa: E402
from retail_forecast.conformal import (  # noqa: E402
    ConformalConfig,
    fit_conformal_deltas,
    save_conformal_deltas,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run backtest and fit conformal calibration deltas.")
    ap.add_argument("--early-weeks", type=int, default=4)
    ap.add_argument("--horizon", type=int, default=26)
    ap.add_argument("--n-sims", type=int, default=2000)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--include-ensemble", action="store_true")
    ap.add_argument("--nf-models-dir", type=str, default=str(repo_root / "models" / "neuralforecast" / "h26_in26"))
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
        panel_path=repo_root / "Dataset" / "Historical_Data.csv",
        config=bt_cfg,
    )

    if rows.empty:
        print("backtest produced 0 rows; nothing to calibrate")
        return

    # Fit deltas
    conf_cfg = ConformalConfig(
        alpha=0.10,
        min_samples=int(args.min_samples),
        per_trademark_min_samples=int(args.per_trademark_min_samples),
    )
    deltas = fit_conformal_deltas(rows, config=conf_cfg)

    out_path = save_conformal_deltas(deltas, models_dir=repo_root / "models", fingerprint=fp)

    # Save metrics for inspection
    out_dir = repo_root / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "backtest_metrics.csv"
    metrics.to_csv(metrics_path, index=False)

    print("deltas head(5):")
    print(deltas.head(5).to_string(index=False))
    print(f"deltas_parquet={out_path}")
    print(f"backtest_metrics_csv={metrics_path}")


if __name__ == "__main__":
    main()
