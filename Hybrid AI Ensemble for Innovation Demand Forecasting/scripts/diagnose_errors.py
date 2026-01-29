
import pandas as pd
import numpy as np

def main():
    print("Loading hybrid_production_results.csv...")
    df = pd.read_csv("outputs/hybrid_production_results.csv")
    
    # Calculate errors
    df['abs_error'] = np.abs(df['y_true'] - df['hybrid_p50'])
    df['ape'] = df['abs_error'] / df['y_true'].replace(0, np.nan)
    df['wmape_contribution'] = df['abs_error'] / df['y_true'].sum()
    
    print(f"Overall WMAPE: {df['abs_error'].sum() / df['y_true'].sum() * 100:.2f}%")
    
    # 1. Error by Scale (Volume)
    # Bin by y_true
    df['vol_bin'] = pd.qcut(df['y_true'], q=5, labels=['Very Low', 'Low', 'Med', 'High', 'Very High'])
    
    print("\nWMAPE by Volume Bin:")
    print(df.groupby('vol_bin')['abs_error'].sum() / df.groupby('vol_bin')['y_true'].sum() * 100)
    
    # 2. Error by Horizon
    print("\nWMAPE by Horizon:")
    print(df.groupby('horizon_step')['abs_error'].sum() / df.groupby('horizon_step')['y_true'].sum() * 100)
    
    # 3. Component Correlation
    # See which model correlates best with y_true in different segments
    # Columns: analog_p50, timesfm_p50, chronos_p50
    
    components = ['analog_p50', 'timesfm_p50', 'chronos_p50']
    available = [c for c in components if c in df.columns]
    
    if available:
        print("\nComponent Correlations with y_true (by Horizon):")
        corrs = df.groupby('horizon_step')[available + ['y_true']].corr()['y_true'].unstack()[available]
        print(corrs)
        
        # Check if one model dominates in specific horizons
        print("\nBest model per horizon (Pearson correlation):")
        print(corrs.idxmax(axis=1))
        
        # Check MAE per horizon
        print("\nMAE per model per horizon:")
        mae_data = []
        for h in range(1, 15):
            mask = df['horizon_step'] == h
            row = {}
            for c in available:
                row[c] = np.mean(np.abs(df.loc[mask, 'y_true'] - df.loc[mask, c]))
            mae_data.append(row)
        print(pd.DataFrame(mae_data, index=range(1, 15)))

if __name__ == "__main__":
    main()
