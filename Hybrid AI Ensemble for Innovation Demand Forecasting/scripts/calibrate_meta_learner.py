import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from retail_forecast.calibration_v5 import train_v5_calibration, V5CalibrationConfig

def main():
    print("Calibrating Meta-Learner...")
    df = pd.read_csv("outputs/meta_learner_cv_results.csv")
    
    config = V5CalibrationConfig(target_coverage=0.80)
    # Using 'meta_p50' as the forecast column
    result = train_v5_calibration(df, config, p50_col='meta_p50')
    
    # Print params
    print("\nMETA_V5_BIAS_FACTORS = {")
    for h, v in result.bias_factors.items():
        print(f"    {h}: {v:.4f},")
    print("}")
    
    print("\nMETA_V5_CQR_BOUNDS = {")
    for h, v in result.cqr_bounds.items():
        print(f"    {h}: {v},")
    print("}")
    
    # Evaluate
    # Apply calibration manually to check
    df['cal_p50'] = df['meta_p50'] * df['horizon_step'].map(result.bias_factors)
    # Bounds
    q_lo = df['horizon_step'].map(lambda h: result.cqr_bounds[h]['q_lo'])
    q_hi = df['horizon_step'].map(lambda h: result.cqr_bounds[h]['q_hi'])
    
    df['cal_p10'] = df['cal_p50'] * (1 + q_lo)
    df['cal_p90'] = df['cal_p50'] * (1 + q_hi)
    
    cov = ((df['y_true'] >= df['cal_p10']) & (df['y_true'] <= df['cal_p90'])).mean()
    width = ((df['cal_p90'] - df['cal_p10']) / df['cal_p50']).median()
    
    print(f"\nResults with Meta-Learner + V5:")
    print(f"Coverage: {cov*100:.2f}%")
    print(f"Median Rel Width: {width*100:.0f}%")

if __name__ == "__main__":
    main()
