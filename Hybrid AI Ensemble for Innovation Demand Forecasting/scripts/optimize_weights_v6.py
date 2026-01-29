
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import GradientBoostingRegressor
from tqdm import tqdm

def main():
    print("Loading data...")
    df = pd.read_csv("outputs/hybrid_production_results.csv")
    
    # Filter valid rows
    cols = ['y_true', 'analog_p50', 'timesfm_p50', 'chronos_p50', 'horizon_step']
    df = df.dropna(subset=cols)
    
    # Metric function
    def wmape(y_true, y_pred):
        return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100

    print(f"Current Hybrid WMAPE: {wmape(df['y_true'], df['hybrid_p50']):.2f}%")
    
    # 1. Static Weight Optimization per Horizon
    print("\n=== Static Weight Optimization ===")
    
    best_weights = {}
    
    # Grid search for 3 components sum=1
    # step 0.05
    steps = np.arange(0, 1.05, 0.05)
    
    print(f"{'H':<3} | {'Analog':<6} {'Chronos':<7} {'TimesFM':<7} | {'New WMAPE':<9} | {'Old WMAPE'}")
    print("-" * 60)
    
    total_ae_new = 0
    total_y = 0
    
    # Current hardcoded weights for comparison
    # COPY from foundation_models.py
    OLD_WEIGHTS = {
        1: (0.00, 0.35, 0.65),
        2: (0.00, 0.00, 1.00),
        3: (0.00, 0.00, 1.00),
        4: (0.10, 0.10, 0.80),
        5: (0.45, 0.20, 0.35),
        6: (0.75, 0.25, 0.00),
        7: (0.80, 0.20, 0.00),
        8: (0.75, 0.20, 0.05),
        9: (0.65, 0.15, 0.20),
        10: (0.85, 0.15, 0.00),
        11: (0.90, 0.10, 0.00),
        12: (0.90, 0.10, 0.00),
        13: (0.90, 0.10, 0.00),
        14: (0.85, 0.15, 0.00),
    }

    for h in range(1, 15):
        mask = df['horizon_step'] == h
        if not mask.any(): continue
        
        y = df.loc[mask, 'y_true'].values
        ana = df.loc[mask, 'analog_p50'].values
        chr_ = df.loc[mask, 'chronos_p50'].values
        tfm = df.loc[mask, 'timesfm_p50'].values
        
        best_h_wmape = float('inf')
        best_w = (0, 0, 0)
        
        # Grid search
        for w_a in steps:
            for w_c in steps:
                w_t = 1.0 - w_a - w_c
                if w_t < -0.001: continue
                # Fix float precision
                if abs(w_t) < 0.001: w_t = 0.0
                
                pred = w_a * ana + w_c * chr_ + w_t * tfm
                err = np.sum(np.abs(y - pred))
                if err < best_h_wmape:
                    best_h_wmape = err
                    best_w = (w_a, w_c, w_t)
        
        # Calculate old WMAPE
        old_w = OLD_WEIGHTS.get(h, (0.33, 0.33, 0.33))
        old_pred = old_w[0]*ana + old_w[1]*chr_ + old_w[2]*tfm
        old_h_wmape_val = np.sum(np.abs(y - old_pred)) / np.sum(np.abs(y)) * 100
        
        # New WMAPE pct
        new_h_wmape_val = best_h_wmape / np.sum(np.abs(y)) * 100
        
        print(f"{h:<3} | {best_w[0]:.2f}   {best_w[1]:.2f}    {best_w[2]:.2f}    | {new_h_wmape_val:.2f}%     | {old_h_wmape_val:.2f}%")
        
        best_weights[h] = best_w
        total_ae_new += best_h_wmape
        total_y += np.sum(np.abs(y))

    overall_new_wmape = total_ae_new / total_y * 100
    print(f"\nOverall Static Optimized WMAPE: {overall_new_wmape:.2f}%")
    
    # 2. Meta-Learner (Stacking) Analysis
    print("\n=== Meta-Learner Analysis (XGBoost) ===")
    
    # Features: Horizon, and maybe Volume (log scale)?
    # Simple stacking: just use the predictions + horizon as feature
    
    # Prepare data
    # X = [analog, chronos, timesfm, horizon]
    # To make it scale-invariant, we might train on relative errors or just raw values?
    # Usually standard stacking uses raw values if scale is consistent, but here scale varies wildy.
    # Better: Train to predict weights? Or train separate models per scale bin?
    
    # Let's try simple per-horizon Linear Regression (OLS) first (constrained coefficients?)
    # OLS minimizes MSE, not MAE. We want MAE.
    # QuantileRegressor(quantile=0.5) minimizes MAE.
    
    from sklearn.linear_model import QuantileRegressor
    # QuantileRegressor is slow on large data. Use GradientBoostingRegressor(loss='absolute_error')
    
    X = df[['analog_p50', 'chronos_p50', 'timesfm_p50', 'horizon_step']].copy()
    y = df['y_true']
    
    # Train/Test split (random or time? Random is fine for backtest evaluation)
    # Actually LOBO is best but expensive here. Let's do a simple 5-fold CV.
    from sklearn.model_selection import KFold
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    stack_preds = np.zeros(len(df))
    
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # GBR
        model = GradientBoostingRegressor(loss='absolute_error', n_estimators=100, max_depth=3)
        model.fit(X_train, y_train)
        stack_preds[test_idx] = model.predict(X_test)
        
    stack_wmape = wmape(y, stack_preds)
    print(f"XGBoost Stacking WMAPE: {stack_wmape:.2f}%")
    
    if stack_wmape < overall_new_wmape:
        print("Stacking beats Static Weights!")
    else:
        print("Static Weights beat Stacking (or overfitting occurred).")

    # Print python dict for weights
    print("\nNEW_OPTIMIZED_WEIGHTS = {")
    for h in sorted(best_weights.keys()):
        w = best_weights[h]
        print(f"    {h}: ({w[0]:.2f}, {w[1]:.2f}, {w[2]:.2f}),")
    print("}")

if __name__ == "__main__":
    main()
