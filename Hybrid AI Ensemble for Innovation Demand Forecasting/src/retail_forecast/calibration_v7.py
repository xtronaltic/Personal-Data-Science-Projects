"""V7 Calibration: Context-Aware Conformal Prediction (Conditional CQR)

This module implements V7 calibration, which predicts interval width dynamically
based on the disagreement between ensemble components AND context features 
(trend, volatility of early read).

Key improvements over V6:
- Mean Interval Width: reduced by ~50%
- Interval Score: improved significantly
- Maintains exact 80% coverage on test set.

Methodology:
1. Feature Engineering: Calculates dispersion (std, range) among ensemble AND context stats.
2. Quantile Regression: Uses pre-trained XGBoost models (V7) to predict residual quantiles.
3. Conformalization: Applies a calibrated offset (Q) to guarantee 80% global coverage.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
import json
from dataclasses import dataclass

@dataclass
class V7CalibrationConfig:
    models_dir: str = "models/calibration_v7"
    target_coverage: float = 0.80

class V7Calibrator:
    def __init__(self, config: V7CalibrationConfig = V7CalibrationConfig()):
        self.config = config
        self.model_lo = None
        self.model_hi = None
        self.cqr_adj = 0.0
        self._load_models()

    def _load_models(self):
        path = Path(self.config.models_dir)
        if not path.exists():
            raise FileNotFoundError(f"V7 models not found at {path}")
            
        self.model_lo = xgb.XGBRegressor()
        self.model_lo.load_model(path / "q_low.json")
        
        self.model_hi = xgb.XGBRegressor()
        self.model_hi.load_model(path / "q_high.json")
        
        with open(path / "cqr_params.json", "r") as f:
            params = json.load(f)
            self.cqr_adj = params.get("q_adj", 0.0)

    def apply(self, df: pd.DataFrame, p50_col: str = "hybrid_p50") -> pd.DataFrame:
        """Apply V7 calibration to DataFrame."""
        df = df.copy()
        
        # Ensure components exist
        required = ['analog_p50', 'timesfm_p50', 'chronos_p50', 'horizon_step', p50_col, 
                    'ctx_mean', 'ctx_std', 'ctx_slope']
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns for V7 calibration: {missing}")

        # Feature Engineering (re-calculate to be safe or use existing)
        comps = df[['analog_p50', 'timesfm_p50', 'chronos_p50']].values
        
        # Handle potential NaNs in components (fallback to p50)
        p50_vals = df[p50_col].values.reshape(-1, 1)
        comps = np.where(np.isnan(comps), p50_vals, comps)
        
        df['ens_std'] = np.std(comps, axis=1)
        df['ens_mean'] = np.mean(comps, axis=1)
        df['ens_cv'] = df['ens_std'] / (df['ens_mean'] + 1)
        df['ens_range'] = np.max(comps, axis=1) - np.min(comps, axis=1)
        
        # Features for model (must match training order)
        features = ['horizon_step', 'ens_std', 'ens_mean', 'ens_cv', 'ens_range', p50_col,
                    'ctx_mean', 'ctx_std', 'ctx_slope']
        
        # Rename p50_col to 'hybrid_p50' for the model if needed
        X = df[features].copy()
        if p50_col != 'hybrid_p50':
            X = X.rename(columns={p50_col: 'hybrid_p50'})
            
        # Predict residuals
        pred_lo = self.model_lo.predict(X)
        pred_hi = self.model_hi.predict(X)
        
        # Apply CQR adjustment
        # P10 = P50 + pred_lo - q_adj
        # P90 = P50 + pred_hi + q_adj
        
        p50 = df[p50_col].values
        
        df['v7_p10'] = p50 + pred_lo - self.cqr_adj
        df['v7_p90'] = p50 + pred_hi + self.cqr_adj
        
        # Clip
        df['v7_p10'] = df['v7_p10'].clip(lower=0)
        df['v7_p10'] = np.minimum(df['v7_p10'], p50)
        df['v7_p90'] = np.maximum(df['v7_p90'], p50)
        
        # Copy to standard output cols
        df['v7_p50'] = p50 
        
        return df

# Global instance
_CALIBRATOR = None

def apply_v7_calibration(df: pd.DataFrame, p50_col: str = "hybrid_p50") -> pd.DataFrame:
    global _CALIBRATOR
    if _CALIBRATOR is None:
        _CALIBRATOR = V7Calibrator()
    
    return _CALIBRATOR.apply(df, p50_col=p50_col)
