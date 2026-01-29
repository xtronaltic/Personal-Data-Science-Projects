
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import json

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from retail_forecast.foundation_models import OPTIMIZED_3WAY_WEIGHTS
from retail_forecast.calibration_v5 import train_v5_calibration, V5CalibrationConfig

def main():
    print("Updating V5 Calibration Parameters...")
    
    # 1. Load Data
    df = pd.read_csv('outputs/hybrid_production_results.csv')
    
    # 2. Compute 3-Way Ensemble
    df['ens_3way_p50'] = 0.0
    for h, weights in OPTIMIZED_3WAY_WEIGHTS.items():
        mask = df['horizon_step'] == h
        w_ana, w_chr, w_tfm = weights
        df.loc[mask, 'ens_3way_p50'] = (
            w_ana * df.loc[mask, 'analog_p50'] +
            w_chr * df.loc[mask, 'chronos_p50'] +
            w_tfm * df.loc[mask, 'timesfm_p50']
        )
    
    # 3. Train V5 Calibration
    config = V5CalibrationConfig(target_coverage=0.80)
    v5_result = train_v5_calibration(df, config, p50_col='ens_3way_p50')
    
    # 4. Print Parameters for Copy-Paste
    print("\n# NEW V5 PARAMETERS (Generated from Optimal 3-Way Backtest)\n")
    
    print("V5_BIAS_FACTORS = {")
    for h in sorted(v5_result.bias_factors.keys()):
        print(f"    {h}: {v5_result.bias_factors[h]:.4f},")
    print("}")
    
    print("\nV5_CQR_BOUNDS = {")
    for h in sorted(v5_result.cqr_bounds.keys()):
        b = v5_result.cqr_bounds[h]
        print(f"    {h}: {{'q_lo': {b['q_lo']:.3f}, 'q_hi': {b['q_hi']:.3f}}},")
    print("}")
    
    # 5. Save to JSON for safety
    v5_result.save('outputs/v5_calibration_params_new.json')
    print("\nSaved to outputs/v5_calibration_params_new.json")

if __name__ == "__main__":
    main()
