#!/usr/bin/env python
"""Production runner for Innovation Forecast.

This script is the main entry point for running forecasts on new innovations.
It handles:
  - Dynamic as-of date detection from Historical_Data.csv
  - Fingerprint-based artifact/model rebuild only when data changes
  - State-of-the-Art Meta-Learner Ensemble Forecasting (Analog + Chronos + TimesFM)
  - V7 Context-Aware Conditional Calibration (80% Coverage, 50% narrower intervals)

Usage:
    python scripts/run_forecast.py --mode ensemble --calibrate
    python scripts/run_forecast.py --new_innovation_path ./my_new_data.csv

See --help for all options.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Production runner for new-innovation forecasting (Meta-Learner + V7).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Data & Paths
    parser.add_argument("--panel_path", type=str, default="./Dataset/Historical_Data.csv", help="Path to Historical_Data.csv.")
    parser.add_argument("--new_innovation_path", type=str, default=None, help="Path to New_Innovations file.")
    parser.add_argument("--artifacts_dir", type=str, default="./artifacts", help="Directory for artifacts.")
    parser.add_argument("--models_dir", type=str, default="./models", help="Directory for models.")
    parser.add_argument("--outputs_dir", type=str, default="./outputs", help="Directory for forecast outputs.")
    
    # Pipeline Control
    parser.add_argument("--train_if_needed", action="store_true", default=True, help="Rebuild artifacts if data changed.")
    parser.add_argument("--no_train", action="store_true", default=False, help="Force skip artifact rebuild.")
    
    # Forecast Settings
    parser.add_argument("--mode", type=str, choices=["analog", "ensemble"], default="ensemble", help="Forecasting mode.")
    parser.add_argument("--horizon", type=int, default=26, help="Forecast horizon in weeks.")
    parser.add_argument("--early_weeks", type=int, default=4, help="Weeks of observed data.")
    parser.add_argument("--top_k", type=int, default=50, help="Number of neighbors.")
    parser.add_argument("--n_sims", type=int, default=5000, help="Number of simulations.")
    
    # Calibration
    parser.add_argument("--calibrate", action="store_true", default=True, help="Apply V7 calibration.")
    parser.add_argument("--calibration_version", type=str, default="v7", help="Calibration version (v7 recommended).")

    args = parser.parse_args()

    # Paths
    panel_path = Path(args.panel_path)
    artifacts_dir = Path(args.artifacts_dir)
    models_dir = Path(args.models_dir)
    outputs_dir = Path(args.outputs_dir)
    similarity_root = models_dir / "similarity"
    
    # Ensure directories exist
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # Import modules
    try:
        from retail_forecast.utils import set_global_seeds
        set_global_seeds(42)
        
        from retail_forecast.io import fingerprint_df, load_panel
        from retail_forecast.pipeline import run_full_pipeline
        from retail_forecast.production import run_production_forecast, ProductionConfig
    except ImportError as e:
        print(f"ERROR: Could not import retail_forecast modules: {e}")
        sys.exit(1)

    # 1. Load panel data & Check Fingerprint
    print(f"[1/3] Checking Data & Artifacts...")
    if not panel_path.exists():
        print(f"ERROR: Panel data file not found: {panel_path}")
        sys.exit(1)
        
    df_panel = load_panel(str(panel_path))
    
    # 2. Rebuild Artifacts if Needed
    if args.train_if_needed and not args.no_train:
        result = run_full_pipeline(
            df_panel,
            artifacts_dir=str(artifacts_dir),
            models_root=str(similarity_root),
            state_path=str(artifacts_dir / "state.json"),
        )
        if any(result.values()):
            print(f"      Artifacts updated: {result}")
        else:
            print("      Artifacts up to date.")

    # 3. Run Production Forecast
    print(f"[2/3] Running Production Forecast ({args.mode} + {args.calibration_version})...")
    
    # Config
    config = ProductionConfig(
        horizon=args.horizon,
        early_weeks=args.early_weeks,
        top_k=args.top_k,
        n_sims=args.n_sims,
        calibration_version=args.calibration_version,
    )
    
    # Determine input path
    input_path = args.new_innovation_path
    if input_path is None:
        # Try default locations
        defaults = [Path("./Dataset/New_Innovations.csv")]
        for p in defaults:
            if p.exists():
                input_path = str(p)
                break
    
    if input_path is None:
        print("ERROR: No New_Innovations file found. Please provide --new_innovation_path")
        sys.exit(1)

    try:
        forecast = run_production_forecast(
            new_innovation_path=input_path,
            panel_path=panel_path,
            artifacts_dir=artifacts_dir,
            models_dir=models_dir,
            output_dir=outputs_dir,
            mode=args.mode,
            apply_calibration=args.calibrate,
            config=config
        )
        print("\n[3/3] Forecast Complete.")
        
    except Exception as e:
        print(f"\nERROR during forecasting: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()