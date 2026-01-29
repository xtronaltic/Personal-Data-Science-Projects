
import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from retail_forecast.calibration_v6 import apply_v6_calibration

def main():
    print("Finalizing Meta-Learner Results for Production (V6)...")
    
    # Load CV results
    df = pd.read_csv("outputs/meta_learner_cv_results.csv")
    
    # Rename
    df['hybrid_p50'] = df['meta_p50']
    
    # Apply V6 Calibration (Uncertainty-Aware)
    # Note: V6 requires component columns (analog_p50, etc.) which are in the CSV
    try:
        df_cal = apply_v6_calibration(df, p50_col='hybrid_p50')
    except Exception as e:
        print(f"V6 Calibration Failed: {e}")
        # Fallback to V5 if V6 fails (e.g. models missing)
        print("Falling back to V5...")
        from retail_forecast.calibration_v5 import apply_v5_calibration, get_v5_production_bounds
        bias, bounds = get_v5_production_bounds()
        df_cal = apply_v5_calibration(df, bias_factors=bias, cqr_bounds=bounds, p50_col='hybrid_p50')
        df_cal['v6_p50'] = df_cal['p50_v5']
        df_cal['v6_p10'] = df_cal['p10_v5']
        df_cal['v6_p90'] = df_cal['p90_v5']
    
    # Rename to production columns
    df_cal['prod_p50'] = df_cal['v6_p50']
    df_cal['prod_p10'] = df_cal['v6_p10']
    df_cal['prod_p90'] = df_cal['v6_p90']
    
    # Verify metrics
    y = df_cal['y_true']
    p50 = df_cal['prod_p50']
    p10 = df_cal['prod_p10']
    p90 = df_cal['prod_p90']
    
    wmape = np.sum(np.abs(y - p50)) / np.sum(np.abs(y)) * 100
    cov = ((y >= p10) & (y <= p90)).mean() * 100
    width = np.median((p90 - p10) / np.maximum(p50, 1)) * 100
    
    print("\nFINAL PRODUCTION METRICS (Meta-Learner + V6):")
    print(f"WMAPE: {wmape:.2f}%")
    print(f"Coverage: {cov:.2f}%")
    print(f"Rel Width: {width:.0f}%")
    
    # Save
    # Ensure key exists
    if 'key' not in df_cal.columns:
        # Reconstruct key if missing (meta_learner_cv_results might not have it if not passed through)
        # But wait, run_full_comparison_scratch saved it to hybrid_production_results.csv
        # eval_meta_learner.py read hybrid_production_results.csv and saved meta_learner_cv_results.csv
        # Check if key was preserved in eval_meta_learner.py
        pass

    # Actually, let's just reload hybrid_production_results.csv to get the keys and merge
    # Because meta_learner_cv_results might be a subset or missing columns
    original_df = pd.read_csv("outputs/hybrid_production_results.csv")
    
    # We need to align rows. Assuming index matches or use merge keys?
    # eval_meta_learner.py filtered dropna.
    # It's safer to run this on hybrid_production_results directly if we trust the meta_p50 column there?
    # Wait, run_full_comparison_scratch ALREADY computed meta_p50 and saved it to hybrid_production_results.csv
    # So we don't need meta_learner_cv_results.csv if run_full_comparison_scratch ran successfully.
    # Let's just use hybrid_production_results.csv
    
    # Reload from hybrid_production_results.csv which has the production meta forecasts
    df_prod = pd.read_csv("outputs/hybrid_production_results.csv")
    
    # Apply V6
    df_prod_cal = apply_v6_calibration(df_prod, p50_col='hybrid_p50')
    df_prod_cal['prod_p10'] = df_prod_cal['v6_p10']
    df_prod_cal['prod_p90'] = df_prod_cal['v6_p90']
    
    df_prod_cal.to_csv("outputs/hybrid_production_results.csv", index=False)
    print("\nSaved updated V6 results to outputs/hybrid_production_results.csv")

if __name__ == "__main__":
    main()
