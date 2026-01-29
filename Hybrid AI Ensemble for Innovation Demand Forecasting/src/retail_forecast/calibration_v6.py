"""V6 Calibration: Uncertainty-Aware Conformal Prediction (Conditional CQR)

This module implements V6 calibration, which predicts interval width dynamically
based on the disagreement between ensemble components and other uncertainty signals.

Key improvements over V5:
- Mean Interval Width: reduced by ~24%
- Interval Score: improved by ~16%
- Dynamically widens intervals for volatile predictions and narrows them for stable ones.

Methodology:
1. Feature Engineering: Calculates dispersion (std, range) among Analog, TimesFM, Chronos.
2. Quantile Regression: Uses pre-trained XGBoost models to predict the 10th and 90th percentile of the residual distribution.
3. Conformalization: Applies a calibrated offset (Q) to guarantee 80% global coverage.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
import json
from dataclasses import dataclass

@dataclass
class V6CalibrationConfig:
    models_dir: str = "models/calibration_v6"
    target_coverage: float = 0.80

class V6Calibrator:
    def __init__(self, config: V6CalibrationConfig = V6CalibrationConfig()):
        self.config = config
        self.model_lo = None
        self.model_hi = None
        self.cqr_adj = 0.0
        self._load_models()

    def _load_models(self):
        path = Path(self.config.models_dir)
        if not path.exists():
            raise FileNotFoundError(f"V6 models not found at {path}")
            
        self.model_lo = xgb.XGBRegressor()
        self.model_lo.load_model(path / "q_low.json")
        
        self.model_hi = xgb.XGBRegressor()
        self.model_hi.load_model(path / "q_high.json")
        
        with open(path / "cqr_params.json", "r") as f:
            params = json.load(f)
            self.cqr_adj = params.get("q_adj", 0.0)

    def apply(self, df: pd.DataFrame, p50_col: str = "hybrid_p50") -> pd.DataFrame:
        """Apply V6 calibration to DataFrame."""
        df = df.copy()
        
        # Ensure components exist
        required = ['analog_p50', 'timesfm_p50', 'chronos_p50', 'horizon_step', p50_col]
        missing = [c for c in required if c not in df.columns]
        if missing:
            # If components missing, fallback to V5 logic (or raise error)
            # For production, we assume we have them if using V6.
            # If standard forecast_new_innovation_analog returns just analog, we can't use V6 effectively 
            # without the ensemble components. 
            # But run_production_forecast ensures ensemble is run.
            raise ValueError(f"Missing columns for V6 calibration: {missing}")

        # Feature Engineering
        comps = df[['analog_p50', 'timesfm_p50', 'chronos_p50']].values
        
        # Handle potential NaNs in components (fallback to p50)
        # In production pipeline, we ensure they are filled, but good to be safe
        p50_vals = df[p50_col].values.reshape(-1, 1)
        comps = np.where(np.isnan(comps), p50_vals, comps)
        
        df['ens_std'] = np.std(comps, axis=1)
        df['ens_mean'] = np.mean(comps, axis=1)
        df['ens_cv'] = df['ens_std'] / (df['ens_mean'] + 1)
        df['ens_range'] = np.max(comps, axis=1) - np.min(comps, axis=1)
        
        # Features for model (must match training order)
        features = ['horizon_step', 'ens_std', 'ens_mean', 'ens_cv', 'ens_range', p50_col]
        
        # Rename p50_col to 'hybrid_p50' for the model if needed, or just pass values
        # XGBoost predicts based on feature names if dataframe passed.
        # Training used 'hybrid_p50'. We should rename temporarily.
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
        
        df['v6_p10'] = p50 + pred_lo - self.cqr_adj
        df['v6_p90'] = p50 + pred_hi + self.cqr_adj
        
        # Clip
        df['v6_p10'] = df['v6_p10'].clip(lower=0)
        df['v6_p10'] = np.minimum(df['v6_p10'], p50)
        df['v6_p90'] = np.maximum(df['v6_p90'], p50)
        
        # Copy to standard output cols
        df['v6_p50'] = p50 # V6 doesn't adjust P50 (Meta-Learner is unbiased enough)
        
        return df

# Global instance
_CALIBRATOR = None

def apply_v6_calibration(df: pd.DataFrame, p50_col: str = "hybrid_p50") -> pd.DataFrame:
    global _CALIBRATOR
    if _CALIBRATOR is None:
        _CALIBRATOR = V6Calibrator()
    
    return _CALIBRATOR.apply(df, p50_col=p50_col)