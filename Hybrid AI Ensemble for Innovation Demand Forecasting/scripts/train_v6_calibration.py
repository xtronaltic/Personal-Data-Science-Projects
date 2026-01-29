import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold
from pathlib import Path
import json

def main():
    print("Training V6 Calibration (Uncertainty-Aware CQR)...")
    
    # 1. Load Data
    # Use CV results to ensure calibration learns from out-of-sample errors
    df = pd.read_csv("outputs/meta_learner_cv_results.csv")
    
    # Rename for consistency if needed, or just use meta_p50
    if 'hybrid_p50' not in df.columns:
        df['hybrid_p50'] = df['meta_p50']
        
    cols = ['y_true', 'analog_p50', 'timesfm_p50', 'chronos_p50', 'horizon_step', 'hybrid_p50']
    df = df.dropna(subset=cols)
    
    # 2. Feature Engineering (Uncertainty Signals)
    print("Generating uncertainty features...")
    
    # Component predictions matrix
    comps = df[['analog_p50', 'timesfm_p50', 'chronos_p50']].values
    
    # Disagreement features
    df['ens_std'] = np.std(comps, axis=1)
    df['ens_mean'] = np.mean(comps, axis=1)
    df['ens_cv'] = df['ens_std'] / (df['ens_mean'] + 1) # Coeff of Variation
    df['ens_range'] = np.max(comps, axis=1) - np.min(comps, axis=1)
    
    # Horizon is categorical/ordinal
    df['horizon_step'] = df['horizon_step'].astype(int)
    
    # Features for Uncertainty Model
    features = ['horizon_step', 'ens_std', 'ens_mean', 'ens_cv', 'ens_range', 'hybrid_p50']
    
    # Target: We want to predict the ERROR limits
    # We will use CQR (Conformalized Quantile Regression)
    # We train two models: one for q_low (0.1) and one for q_high (0.9) of the RESIDUAL
    # Residual = y_true - y_pred
    df['residual'] = df['y_true'] - df['hybrid_p50']
    
    X = df[features]
    y_res = df['residual']
    
    # 3. Train Quantile Regressors (CV)
    print("Training Quantile XGBoost models (10th & 90th percentiles of error)...")
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Store predictions
    df['pred_err_low'] = 0.0
    df['pred_err_high'] = 0.0
    
    # Params for Quantile Regression
    # XGBoost supports quantile error objective
    
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y_res.iloc[train_idx], y_res.iloc[test_idx]
        
        # Model for 10th percentile (Lower Bound)
        # Note: XGBoost < 2.0 uses 'reg:quantileerror' with 'quantile_alpha'
        # Check XGBoost version or use standard objective
        # Assuming modern XGBoost installed by previous agent
        
        model_lo = xgb.XGBRegressor(
            objective='reg:quantileerror',
            quantile_alpha=0.1,
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            n_jobs=-1
        )
        model_lo.fit(X_train, y_train)
        df.loc[df.index[test_idx], 'pred_err_low'] = model_lo.predict(X_test)
        
        # Model for 90th percentile (Upper Bound)
        model_hi = xgb.XGBRegressor(
            objective='reg:quantileerror',
            quantile_alpha=0.9,
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            n_jobs=-1
        )
        model_hi.fit(X_train, y_train)
        df.loc[df.index[test_idx], 'pred_err_high'] = model_hi.predict(X_test)

    # 4. Conformalize (Apply CQR)
    # P10_raw = y_pred + pred_err_low
    # P90_raw = y_pred + pred_err_high
    # We compute coverage on calibration set and adjust bounds
    
    df['raw_p10'] = df['hybrid_p50'] + df['pred_err_low']
    df['raw_p90'] = df['hybrid_p50'] + df['pred_err_high']
    
    # Calculate non-conformity scores
    # We want y_true in [P10 - q, P90 + q]
    # Score_low = P10 - y_true
    # Score_high = y_true - P90
    # Score = max(P10 - y, y - P90)
    # If Score < 0, then y is inside. We want 80% coverage -> 90th percentile of scores < q?
    # Standard CQR: Coverage at 1-alpha.
    # Score_i = max(q_lo_i - y_i, y_i - q_hi_i)
    # Q = Quantile(scores, 1-alpha) * (n+1)/n ? No, Quantile(scores, 1-alpha)
    
    scores = np.maximum(
        df['raw_p10'] - df['y_true'],
        df['y_true'] - df['raw_p90']
    )
    
    target_coverage = 0.80
    alpha = 1 - target_coverage
    # We need the (1-alpha)*(1+1/n) quantile of scores? 
    # Usually we take the (1-alpha) quantile.
    # If 80% coverage, we allow 20% errors.
    # We want 80% of points to have Score < Q.
    # So Q = 80th percentile of scores.
    
    q_adj = np.quantile(scores, target_coverage)
    
    print(f"\nCQR Adjustment (Q): {q_adj:.2f}")
    
    df['v6_p10'] = df['raw_p10'] - q_adj
    df['v6_p90'] = df['raw_p90'] + q_adj
    
    # Clip
    df['v6_p10'] = df['v6_p10'].clip(lower=0)
    
    # 5. Evaluate V6 vs V5
    
    # Metrics
    def get_metrics(name, p10, p90, y, p50):
        cov = ((y >= p10) & (y <= p90)).mean() * 100
        width = np.mean(p90 - p10)
        rel_width = np.median((p90 - p10) / np.maximum(p50, 1)) * 100
        # Interval score
        alpha = 0.2
        s = ((p90 - p10) + (2/alpha)*(p10-y)*(y<p10) + (2/alpha)*(y-p90)*(y>p90)).mean()
        return [name, cov, width, rel_width, s]
    
    # V5 (Existing)
    # Re-calculate V5 metrics on this subset
    m_v5 = get_metrics("V5 (Horizon-Static)", df['prod_p10'], df['prod_p90'], df['y_true'], df['hybrid_p50'])
    
    # V6 (Dynamic)
    m_v6 = get_metrics("V6 (Dynamic CQR)", df['v6_p10'], df['v6_p90'], df['y_true'], df['hybrid_p50'])
    
    print("\nComparison Results:")
    print(f"{ 'Method':<25} {'Coverage':<10} {'Mean Width':<15} {'Rel Width':<10} {'Score (Lower=Better)'}")
    print("-" * 80)
    print(f"{m_v5[0]:<25} {m_v5[1]:<10.1f} ${m_v5[2]:<14,.0f} {m_v5[3]:<10.0f} {m_v5[4]:,.0f}")
    print(f"{m_v6[0]:<25} {m_v6[1]:<10.1f} ${m_v6[2]:<14,.0f} {m_v6[3]:<10.0f} {m_v6[4]:,.0f}")
    
    # Analyze correlation of width with error
    # Does V6 produce wider intervals when error is high?
    df['abs_err'] = np.abs(df['residual'])
    df['v6_width'] = df['v6_p90'] - df['v6_p10']
    corr = df['v6_width'].corr(df['abs_err'])
    print(f"\nCorrelation between V6 Interval Width and Actual Error: {corr:.2f}")
    
    # Train Final Models on Full Data
    print("\nTraining Final V6 Models on full dataset...")
    model_lo = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=0.1, n_estimators=200, max_depth=4)
    model_hi = xgb.XGBRegressor(objective='reg:quantileerror', quantile_alpha=0.9, n_estimators=200, max_depth=4)
    
    model_lo.fit(X, y_res)
    model_hi.fit(X, y_res)
    
    # Save models
    output_dir = Path("models/calibration_v6")
    output_dir.mkdir(parents=True, exist_ok=True)
    model_lo.save_model(output_dir / "q_low.json")
    model_hi.save_model(output_dir / "q_high.json")
    
    # Save CQR adjustment
    with open(output_dir / "cqr_params.json", "w") as f:
        json.dump({"q_adj": float(q_adj)}, f)
        
    print(f"V6 Artifacts saved to {output_dir}")

if __name__ == "__main__":
    main()
