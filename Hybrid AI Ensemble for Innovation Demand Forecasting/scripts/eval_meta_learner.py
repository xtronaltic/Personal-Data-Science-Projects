
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold

def main():
    print("Evaluating Meta-Learner (K-Fold CV)...")
    
    df = pd.read_csv("outputs/hybrid_production_results.csv")
    cols = ['y_true', 'analog_p50', 'timesfm_p50', 'chronos_p50', 'horizon_step']
    df = df.dropna(subset=cols)
    
    X = df[['analog_p50', 'timesfm_p50', 'chronos_p50', 'horizon_step']]
    y = df['y_true']
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    preds = np.zeros(len(y))
    
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        
        model = xgb.XGBRegressor(
            objective='reg:absoluteerror',
            n_estimators=200,
            max_depth=4,
            learning_rate=0.1,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        preds[test_idx] = model.predict(X_test)
    
    # Calc metrics
    wmape = np.sum(np.abs(y - preds)) / np.sum(np.abs(y)) * 100
    
    print(f"Meta-Learner CV WMAPE: {wmape:.2f}%")
    print(f"Baseline (Static 3-way) WMAPE: {33.27}%")
    
    # Save predictions for calibration analysis
    df['meta_p50'] = preds
    df.to_csv("outputs/meta_learner_cv_results.csv", index=False)

if __name__ == "__main__":
    main()
