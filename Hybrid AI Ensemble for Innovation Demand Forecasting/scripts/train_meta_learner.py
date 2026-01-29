
import pandas as pd
import numpy as np
import xgboost as xgb
from pathlib import Path
import json
import os

def main():
    print("Training Meta-Learner (XGBoost)...")
    
    # Load data
    df = pd.read_csv("outputs/hybrid_production_results.csv")
    
    # Filter valid
    cols = ['y_true', 'analog_p50', 'timesfm_p50', 'chronos_p50', 'horizon_step', 'ctx_mean', 'ctx_std', 'ctx_slope']
    df = df.dropna(subset=cols)
    
    # Feature Engineering for Meta-Learner
    comps = df[['analog_p50', 'timesfm_p50', 'chronos_p50']].values
    df['ens_std'] = np.std(comps, axis=1)
    df['ens_range'] = np.max(comps, axis=1) - np.min(comps, axis=1)
    df['ens_mean'] = np.mean(comps, axis=1)
    
    # Normalize context features? XGBoost handles scale well, but let's just pass them raw.
    
    X = df[['analog_p50', 'timesfm_p50', 'chronos_p50', 'horizon_step', 
            'ctx_mean', 'ctx_std', 'ctx_slope', 'ens_std', 'ens_range', 'ens_mean']]
    y = df['y_true']
    
    # Train final model on full data
    # (Since we want to use this for FUTURE products, training on all past products is correct)
    model = xgb.XGBRegressor(
        objective='reg:absoluteerror',
        n_estimators=300, # Increased
        max_depth=5, # Increased depth slightly
        learning_rate=0.05, # Slower learning rate
        n_jobs=-1
    )
    
    model.fit(X, y)
    
    # Save model
    output_dir = Path("models/meta_learner")
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "xgb_v2.json"
    model.save_model(str(model_path))
    
    print(f"Model saved to {model_path}")
    
    # Evaluate in-sample (just sanity check, real eval comes from LOBO simulation or test set)
    preds = model.predict(X)
    wmape = np.sum(np.abs(y - preds)) / np.sum(np.abs(y)) * 100
    print(f"In-sample WMAPE: {wmape:.2f}%")
    
    # Feature importance
    print("\nFeature Importance:")
    print(pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False))

if __name__ == "__main__":
    main()
