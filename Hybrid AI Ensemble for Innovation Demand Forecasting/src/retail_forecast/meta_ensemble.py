
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
from dataclasses import dataclass

@dataclass
class MetaEnsembleConfig:
    model_path: str = "models/meta_learner/xgb_v2.json"

class MetaEnsembleForecaster:
    """Meta-Learner Ensemble using XGBoost to combine base forecasts."""
    
    def __init__(self, config: MetaEnsembleConfig = MetaEnsembleConfig()):
        self.config = config
        self._model = None
        self._load_model()
        
    def _load_model(self):
        if self._model is None:
            self._model = xgb.XGBRegressor()
            self._model.load_model(self.config.model_path)
            
    def forecast(self, 
                 analog_p50: np.ndarray, 
                 timesfm_p50: np.ndarray, 
                 chronos_p50: np.ndarray,
                 ctx_mean: float = 0.0,
                 ctx_std: float = 0.0,
                 ctx_slope: float = 0.0) -> np.ndarray:
        """
        Generate meta-ensemble forecast.
        
        Args:
            analog_p50: Array of shape (horizon,)
            timesfm_p50: Array of shape (horizon,)
            chronos_p50: Array of shape (horizon,)
            ctx_mean: Mean of context window
            ctx_std: Std of context window
            ctx_slope: Slope of context window
            
        Returns:
            Array of shape (horizon,) with stacked predictions
        """
        horizon_len = len(analog_p50)
        horizon_steps = np.arange(1, horizon_len + 1)
        
        # Compute ensemble features
        comps = np.column_stack([analog_p50, timesfm_p50, chronos_p50])
        ens_std = np.std(comps, axis=1)
        ens_range = np.max(comps, axis=1) - np.min(comps, axis=1)
        ens_mean = np.mean(comps, axis=1)
        
        # Prepare input DataFrame (column order must match training)
        X = pd.DataFrame({
            'analog_p50': analog_p50,
            'timesfm_p50': timesfm_p50,
            'chronos_p50': chronos_p50,
            'horizon_step': horizon_steps,
            'ctx_mean': np.full(horizon_len, ctx_mean),
            'ctx_std': np.full(horizon_len, ctx_std),
            'ctx_slope': np.full(horizon_len, ctx_slope),
            'ens_std': ens_std,
            'ens_range': ens_range,
            'ens_mean': ens_mean
        })
        
        # Predict
        return self._model.predict(X)
