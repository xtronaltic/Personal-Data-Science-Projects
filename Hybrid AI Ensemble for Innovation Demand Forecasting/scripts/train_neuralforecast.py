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

from retail_forecast.io import load_new_innovation  # noqa: E402
from retail_forecast.neuralforecast_models import (  # noqa: E402
    NFConfig,
    predict_components_for_new_innovation,
    save_bundles,
    train_all_components,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train NeuralForecast component models and forecast New_Innovations.")
    ap.add_argument("--horizon", type=int, default=26)
    ap.add_argument("--input-size", type=int, default=26)
    ap.add_argument("--models", type=str, default="NHITS,PatchTST")
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    models = tuple([m.strip() for m in str(args.models).split(",") if m.strip()])

    cfg = NFConfig(
        horizon=int(args.horizon),
        input_size=int(args.input_size),
        models=models,
        max_steps=int(args.max_steps),
        batch_size=int(args.batch_size),
    )

    bundles = train_all_components(panel_path=repo_root / "Dataset" / "Historical_Data.csv", config=cfg)

    out_models = repo_root / "models" / "neuralforecast" / f"h{cfg.horizon}_in{cfg.input_size}"
    save_bundles(bundles, out_models)

    df_new = load_new_innovation()
    if df_new is None:
        print("New_Innovations not found; trained models only.")
        print(f"models_dir={out_models}")
        return

    forecasts: list[pd.DataFrame] = []

    # dist_acv
    forecasts.append(
        predict_components_for_new_innovation(
            bundles["dist_acv"],
            df_new,
            horizon=cfg.horizon,
            include_velocity_exog=False,
        )
    )

    # velocity components
    for comp in ["vel_dollars", "vel_units", "vel_eq"]:
        forecasts.append(
            predict_components_for_new_innovation(
                bundles[comp],
                df_new,
                horizon=cfg.horizon,
                include_velocity_exog=True,
            )
        )

    out_df = pd.concat(forecasts, ignore_index=True)

    out_path = repo_root / "outputs" / "neuralforecast_components_forecast.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    print("forecast head(10):")
    print(out_df.head(10).to_string(index=False))
    print(f"neuralforecast_components_forecast_csv={out_path}")
    print(f"models_dir={out_models}")


if __name__ == "__main__":
    main()
