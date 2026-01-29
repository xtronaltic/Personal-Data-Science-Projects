"""Foundation Model Forecasters using state-of-the-art pretrained models.

This module implements forecasting using cutting-edge foundation models:
1. Amazon Chronos-Bolt (2025): Fast, accurate zero-shot forecasting
2. Google TimesFM (2024): Decoder-only foundation model for time series

These models are pretrained on billions of time points and provide excellent
zero-shot performance without task-specific training.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch


@dataclass
class FoundationModelConfig:
    """Configuration for foundation model forecasters."""
    
    # Model selection
    model_type: Literal["chronos-bolt", "timesfm", "ensemble"] = "chronos-bolt"
    
    # Chronos settings
    chronos_model: str = "amazon/chronos-bolt-base"  # Options: tiny, mini, small, base
    
    # TimesFM settings  
    timesfm_model: str = "google/timesfm-1.0-200m-pytorch"
    
    # Inference settings
    prediction_length: int = 26
    num_samples: int = 100  # For probabilistic forecasting
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# Optimized TimesFM weights by horizon (from backtest optimization)
# Format: {horizon_week: timesfm_weight} (analog_weight = 1 - timesfm_weight)
# NOTE: Superseded by OPTIMIZED_3WAY_WEIGHTS for production
OPTIMIZED_HYBRID_WEIGHTS = {
    1: 0.9,   # TimesFM dominates short-term
    2: 1.0,
    3: 1.0,
    4: 0.9,
    5: 0.5,   # Transition zone
    6: 0.3,
    7: 0.3,
    8: 0.3,
    9: 0.5,
    10: 0.2,  # Analog dominates long-term
    11: 0.1,
    12: 0.1,
    13: 0.1,
    14: 0.1,
    # Extended horizon (weeks 15-26): Analog dominates for long-term forecasts
    15: 0.1,
    16: 0.1,
    17: 0.1,
    18: 0.1,
    19: 0.05,
    20: 0.05,
    21: 0.05,
    22: 0.05,
    23: 0.05,
    24: 0.05,
    25: 0.05,
    26: 0.05,
}

# V2 Optimized 3-way ensemble weights (Analog + Chronos + TimesFM)
# From fine-grid search optimization on backtest data
# Achieves 34.5% WMAPE vs 35.8% for 2-way ensemble (3.6% relative improvement)
# Format: {horizon: (analog_weight, chronos_weight, timesfm_weight)}
OPTIMIZED_3WAY_WEIGHTS = {
    1: (0.00, 0.35, 0.65),   # TimesFM dominates, Chronos adds value
    2: (0.00, 0.00, 1.00),   # TimesFM only
    3: (0.00, 0.00, 1.00),   # TimesFM only
    4: (0.10, 0.10, 0.80),   # TimesFM with small Analog/Chronos
    5: (0.45, 0.20, 0.35),   # Transition zone - all contribute
    6: (0.75, 0.25, 0.00),   # Analog + Chronos
    7: (0.80, 0.20, 0.00),   # Analog + Chronos
    8: (0.75, 0.20, 0.05),   # Analog dominant
    9: (0.65, 0.15, 0.20),   # Analog dominant
    10: (0.85, 0.15, 0.00),  # Analog dominant
    11: (0.90, 0.10, 0.00),  # Analog dominant with Chronos stabilization
    12: (0.90, 0.10, 0.00),  # Analog dominant with Chronos stabilization
    13: (0.90, 0.10, 0.00),  # Analog dominant with Chronos stabilization
    14: (0.85, 0.15, 0.00),  # Analog dominant with Chronos stabilization
    # Extended horizon (weeks 15-26): Analog strongly dominates with Chronos stabilization
    # These are conservative defaults until retraining is performed
    15: (0.85, 0.15, 0.00),
    16: (0.85, 0.15, 0.00),
    17: (0.85, 0.15, 0.00),
    18: (0.85, 0.15, 0.00),
    19: (0.90, 0.10, 0.00),
    20: (0.90, 0.10, 0.00),
    21: (0.90, 0.10, 0.00),
    22: (0.90, 0.10, 0.00),
    23: (0.90, 0.10, 0.00),
    24: (0.90, 0.10, 0.00),
    25: (0.90, 0.10, 0.00),
    26: (0.90, 0.10, 0.00),
}


class ChronosBoltForecaster:
    """Amazon Chronos-Bolt foundation model forecaster.
    
    Chronos-Bolt is a patch-based variant that is up to 250x faster than
    original Chronos while maintaining high accuracy.
    """
    
    def __init__(self, config: FoundationModelConfig = FoundationModelConfig()):
        self.config = config
        self._pipeline = None
    
    def _load_model(self) -> None:
        """Lazy-load the Chronos pipeline."""
        if self._pipeline is not None:
            return
        
        from chronos import ChronosBoltPipeline
        
        self._pipeline = ChronosBoltPipeline.from_pretrained(
            self.config.chronos_model,
            device_map=self.config.device,
            dtype=torch.float32,
        )
    
    def forecast(
        self,
        context: np.ndarray | pd.Series,
        prediction_length: int | None = None,
    ) -> dict[str, np.ndarray]:
        """Generate probabilistic forecast.
        
        Args:
            context: Historical time series values
            prediction_length: Number of steps to forecast
        
        Returns:
            Dictionary with p10, p50, p90 forecasts
        """
        self._load_model()
        
        if prediction_length is None:
            prediction_length = self.config.prediction_length
        
        # Convert to tensor
        if isinstance(context, pd.Series):
            context = context.values
        
        context_tensor = torch.tensor(context, dtype=torch.float32).unsqueeze(0)
        
        # Generate quantile forecasts
        # Returns tuple: (quantiles [batch, horizon, num_quantiles], median [batch, horizon])
        quantile_forecasts, median_forecast = self._pipeline.predict_quantiles(
            context_tensor,
            prediction_length=prediction_length,
            quantile_levels=list(self.config.quantiles),
        )
        
        # Extract quantiles (shape: [batch, horizon, num_quantiles])
        forecasts = quantile_forecasts.numpy()[0]  # Remove batch dim
        
        return {
            "p10": forecasts[:, 0],
            "p50": forecasts[:, 1],  # Use quantile median for consistency
            "p90": forecasts[:, 2],
        }
    
    def forecast_batch(
        self,
        contexts: list[np.ndarray],
        prediction_length: int | None = None,
    ) -> list[dict[str, np.ndarray]]:
        """Batch forecast for multiple time series."""
        self._load_model()
        
        if prediction_length is None:
            prediction_length = self.config.prediction_length
        
        # Pad/truncate to same length for batching
        max_len = max(len(c) for c in contexts)
        padded = []
        for c in contexts:
            if len(c) < max_len:
                padded_c = np.pad(c, (max_len - len(c), 0), mode='edge')
            else:
                padded_c = c[-max_len:]
            padded.append(padded_c)
        
        batch_tensor = torch.tensor(np.array(padded), dtype=torch.float32)
        
        # Returns tuple: (quantiles [batch, horizon, num_quantiles], median [batch, horizon])
        quantile_forecasts, _ = self._pipeline.predict_quantiles(
            batch_tensor,
            prediction_length=prediction_length,
            quantile_levels=list(self.config.quantiles),
        )
        
        forecasts_np = quantile_forecasts.numpy()
        
        results = []
        for i in range(len(contexts)):
            results.append({
                "p10": forecasts_np[i, :, 0],
                "p50": forecasts_np[i, :, 1],
                "p90": forecasts_np[i, :, 2],
            })
        
        return results


class TimesFMForecaster:
    """Google TimesFM foundation model forecaster.
    
    TimesFM is a decoder-only foundation model pretrained on 100B real-world
    time points, demonstrating impressive zero-shot performance.
    """
    
    def __init__(self, config: FoundationModelConfig = FoundationModelConfig()):
        self.config = config
        self._model = None
    
    def _load_model(self) -> None:
        """Lazy-load the TimesFM model."""
        if self._model is not None:
            return
        
        import timesfm
        
        # Initialize TimesFM model with PyTorch backend
        self._model = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(
                backend="gpu" if self.config.device == "cuda" else "cpu",
                per_core_batch_size=32,
                horizon_len=self.config.prediction_length,
            ),
            checkpoint=timesfm.TimesFmCheckpoint(
                version="torch",  # Use PyTorch checkpoint
                huggingface_repo_id=self.config.timesfm_model,
            ),
        )
    
    def forecast(
        self,
        context: np.ndarray | pd.Series,
        prediction_length: int | None = None,
    ) -> dict[str, np.ndarray]:
        """Generate probabilistic forecast."""
        self._load_model()
        
        if prediction_length is None:
            prediction_length = self.config.prediction_length
        
        if isinstance(context, pd.Series):
            context = context.values
        
        # TimesFM expects list of context sequences
        context_list = [context.astype(np.float32)]
        
        # Generate forecast 
        # Returns: (mean [batch, horizon], full [batch, horizon, 1 + num_quantiles])
        point_forecast, full_forecast = self._model.forecast(
            context_list,
            freq=[0],  # High frequency (weekly data)
        )
        
        # Extract forecasts - truncate to prediction_length
        point = point_forecast[0][:prediction_length]  # Shape: [horizon]
        
        # full_forecast shape: [batch, horizon, 1 + num_quantiles]
        # Column 0 = mean, columns 1-9 = quantiles (0.1, 0.2, ..., 0.9)
        q_all = full_forecast[0][:prediction_length, :]  # Shape: [horizon, 1+num_quantiles]
        
        return {
            "p10": q_all[:, 1],   # 10th percentile (index 1)
            "p50": point,         # Median/point forecast
            "p90": q_all[:, -1],  # 90th percentile (index 9)
        }


class FoundationModelEnsemble:
    """Ensemble of foundation models for maximum accuracy.
    
    Combines predictions from multiple foundation models:
    - Chronos-Bolt (Amazon)
    - TimesFM (Google)
    - Original Analog Model (domain-specific)
    """
    
    def __init__(
        self,
        config: FoundationModelConfig = FoundationModelConfig(),
        include_analog: bool = True,
    ):
        self.config = config
        self.include_analog = include_analog
        
        # Initialize foundation models
        self.chronos = ChronosBoltForecaster(config)
        self.timesfm = TimesFMForecaster(config)
        
        # Weights for ensemble (can be tuned based on backtesting)
        self.weights = {
            "chronos": 0.4,
            "timesfm": 0.3,
            "analog": 0.3,
        }
    
    def forecast(
        self,
        context: np.ndarray | pd.Series,
        analog_forecast: dict[str, np.ndarray] | None = None,
        prediction_length: int | None = None,
    ) -> dict[str, np.ndarray]:
        """Generate ensemble forecast combining multiple models.
        
        Args:
            context: Historical time series values
            analog_forecast: Optional forecast from analog model
            prediction_length: Number of steps to forecast
        
        Returns:
            Weighted ensemble forecast with p10, p50, p90
        """
        if prediction_length is None:
            prediction_length = self.config.prediction_length
        
        forecasts = []
        weights = []
        
        # Chronos-Bolt forecast
        try:
            chronos_fc = self.chronos.forecast(context, prediction_length)
            forecasts.append(chronos_fc)
            weights.append(self.weights["chronos"])
        except Exception as e:
            print(f"Chronos forecast failed: {e}")
        
        # TimesFM forecast
        try:
            timesfm_fc = self.timesfm.forecast(context, prediction_length)
            forecasts.append(timesfm_fc)
            weights.append(self.weights["timesfm"])
        except Exception as e:
            print(f"TimesFM forecast failed: {e}")
        
        # Analog forecast (if provided)
        if self.include_analog and analog_forecast is not None:
            forecasts.append(analog_forecast)
            weights.append(self.weights["analog"])
        
        if not forecasts:
            raise ValueError("All forecasting models failed")
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        # Weighted average ensemble
        ensemble = {
            "p10": np.zeros(prediction_length),
            "p50": np.zeros(prediction_length),
            "p90": np.zeros(prediction_length),
        }
        
        for fc, w in zip(forecasts, weights):
            for q in ["p10", "p50", "p90"]:
                fc_arr = fc[q]
                # Handle length mismatch
                if len(fc_arr) < prediction_length:
                    fc_arr = np.pad(fc_arr, (0, prediction_length - len(fc_arr)), mode='edge')
                elif len(fc_arr) > prediction_length:
                    fc_arr = fc_arr[:prediction_length]
                ensemble[q] += w * fc_arr
        
        return ensemble


class HybridEnsembleForecaster:
    """Hybrid ensemble combining foundation models with analog forecasts.
    
    Uses horizon-aware weighting with optimized 3-way ensemble (V2):
    - Short-term (weeks 1-4): TimesFM dominates with Chronos contribution
    - Mid-term (weeks 5-9): All three models contribute
    - Long-term (weeks 10-14): Analog dominates with Chronos stabilization
    
    This combines the pattern recognition of foundation models with
    the domain knowledge of analog-based forecasting.
    
    V2 achieves 34.5% WMAPE vs 35.8% for 2-way ensemble (3.6% relative improvement).
    """
    
    def __init__(
        self,
        config: FoundationModelConfig = FoundationModelConfig(),
        use_3way_ensemble: bool = True,  # V2: Use Analog + Chronos + TimesFM
    ):
        self.config = config
        self.use_3way_ensemble = use_3way_ensemble
        self.chronos = ChronosBoltForecaster(config)
        self.timesfm = TimesFMForecaster(config)
    
    def forecast_with_analog(
        self,
        context: np.ndarray | pd.Series,
        analog_forecast: dict[str, np.ndarray],
        prediction_length: int | None = None,
    ) -> dict[str, np.ndarray]:
        """Generate hybrid forecast combining foundation models + analog.
        
        Args:
            context: Historical time series values
            analog_forecast: Forecast from analog model (p10, p50, p90)
            prediction_length: Number of steps to forecast
        
        Returns:
            Hybrid ensemble forecast
        """
        if prediction_length is None:
            prediction_length = self.config.prediction_length
        
        if isinstance(context, pd.Series):
            context = context.values
        
        # Get TimesFM forecast
        timesfm_forecast = None
        try:
            timesfm_forecast = self.timesfm.forecast(context, prediction_length)
        except Exception:
            pass
        
        # Get Chronos forecast for 3-way ensemble
        chronos_forecast = None
        if self.use_3way_ensemble:
            try:
                chronos_forecast = self.chronos.forecast(context, prediction_length)
            except Exception:
                pass
        
        # Use optimized weights from backtest analysis
        ensemble = {
            "p10": np.zeros(prediction_length),
            "p50": np.zeros(prediction_length),
            "p90": np.zeros(prediction_length),
        }
        
        for h in range(prediction_length):
            horizon_week = h + 1  # 1-indexed
            
            if self.use_3way_ensemble and chronos_forecast is not None:
                # V2: Optimized 3-way ensemble (Analog + Chronos + TimesFM)
                weights = OPTIMIZED_3WAY_WEIGHTS.get(horizon_week, (0.5, 0.2, 0.3))
                analog_weight, chronos_weight, timesfm_weight = weights
            else:
                # V1: Original 2-way ensemble (Analog + TimesFM)
                timesfm_weight = OPTIMIZED_HYBRID_WEIGHTS.get(horizon_week, 0.3)
                analog_weight = 1.0 - timesfm_weight
                chronos_weight = 0.0
            
            for q in ["p10", "p50", "p90"]:
                weighted_sum = 0.0
                total_weight = 0.0
                
                # Analog contribution
                if len(analog_forecast[q]) > h:
                    weighted_sum += analog_forecast[q][h] * analog_weight
                    total_weight += analog_weight
                
                # Chronos contribution (V2)
                if chronos_forecast is not None and len(chronos_forecast[q]) > h:
                    weighted_sum += chronos_forecast[q][h] * chronos_weight
                    total_weight += chronos_weight
                
                # TimesFM contribution
                if timesfm_forecast is not None and len(timesfm_forecast[q]) > h:
                    weighted_sum += timesfm_forecast[q][h] * timesfm_weight
                    total_weight += timesfm_weight
                
                # Normalize weights if needed
                if total_weight > 0:
                    ensemble[q][h] = weighted_sum / total_weight
            
        return ensemble



    def forecast_with_meta_learner(
        self,
        context: np.ndarray | pd.Series,
        analog_forecast: dict[str, np.ndarray],
        prediction_length: int | None = None,
        ctx_mean: float = 0.0,
        ctx_std: float = 0.0,
        ctx_slope: float = 0.0,
    ) -> dict[str, np.ndarray]:
        """Generate forecast using Meta-Learner (Stacking).
        
        Args:
            context: Historical time series values
            analog_forecast: Forecast from analog model (p10, p50, p90)
            prediction_length: Number of steps to forecast
            ctx_mean: Mean of context window
            ctx_std: Std of context window
            ctx_slope: Slope of context window
        
        Returns:
            Meta-ensemble forecast
        """
        from .meta_ensemble import MetaEnsembleForecaster
        
        if prediction_length is None:
            prediction_length = self.config.prediction_length
        
        if isinstance(context, pd.Series):
            context = context.values
            
        # Get TimesFM forecast
        timesfm_p50 = np.zeros(prediction_length)
        try:
            tfm_res = self.timesfm.forecast(context, prediction_length)
            timesfm_p50 = tfm_res["p50"]
        except Exception:
            pass
            
        # Get Chronos forecast
        chronos_p50 = np.zeros(prediction_length)
        try:
            chr_res = self.chronos.forecast(context, prediction_length)
            chronos_p50 = chr_res["p50"]
        except Exception:
            pass
            
        # Load meta learner
        meta_model = MetaEnsembleForecaster()
        
        # Analog p50 (pad if needed)
        analog_p50 = analog_forecast["p50"]
        if len(analog_p50) < prediction_length:
            analog_p50 = np.pad(analog_p50, (0, prediction_length - len(analog_p50)), mode='edge')
        elif len(analog_p50) > prediction_length:
            analog_p50 = analog_p50[:prediction_length]
            
        # Predict P50
        p50 = meta_model.forecast(
            analog_p50, timesfm_p50, chronos_p50,
            ctx_mean=ctx_mean, ctx_std=ctx_std, ctx_slope=ctx_slope
        )
        
        # P10/P90 will be handled by calibration
        return {
            "p10": p50, # Placeholder
            "p50": p50,
            "p90": p50,  # Placeholder
            "timesfm_p50": timesfm_p50,
            "chronos_p50": chronos_p50
        }

def run_foundation_model_backtest(
    panel_path: str = "./Dataset/Historical_Data.csv",
    model_type: Literal["chronos-bolt", "timesfm", "ensemble"] = "chronos-bolt",
    early_weeks: int = 4,
    horizon: int = 26,
) -> pd.DataFrame:
    """Run backtest using foundation models.
    
    Args:
        panel_path: Path to panel data
        model_type: Which foundation model to use
        early_weeks: Number of weeks of observed data
        horizon: Forecast horizon
    
    Returns:
        DataFrame with backtest results
    """
    from .io import load_panel
    
    df = load_panel(panel_path)
    
    config = FoundationModelConfig(
        model_type=model_type,
        prediction_length=horizon,
    )
    
    if model_type == "chronos-bolt":
        forecaster = ChronosBoltForecaster(config)
    elif model_type == "timesfm":
        forecaster = TimesFMForecaster(config)
    else:
        forecaster = FoundationModelEnsemble(config, include_analog=False)
    
    results = []
    
    # Group by brand/market
    for (tm, br, mkt), g in df.groupby(
        ["trademark", "brand", "markets"], dropna=False
    ):
        g = g.sort_values("date")
        
        if len(g) < early_weeks + horizon:
            continue
        
        # Use first early_weeks as context, next horizon as test
        for metric in ["dollars", "units", "eq"]:
            if metric not in g.columns:
                continue
            
            values = pd.to_numeric(g[metric], errors="coerce").dropna().values
            
            if len(values) < early_weeks + horizon:
                continue
            
            context = values[:early_weeks]
            actuals = values[early_weeks:early_weeks + horizon]
            
            try:
                forecast = forecaster.forecast(context, prediction_length=len(actuals))
                
                for h in range(len(actuals)):
                    results.append({
                        "model_type": model_type,
                        "trademark": tm,
                        "brand": br,
                        "markets": mkt,
                        "metric": metric,
                        "horizon_step": h + 1,
                        "y_true": actuals[h],
                        "p10": forecast["p10"][h],
                        "p50": forecast["p50"][h],
                        "p90": forecast["p90"][h],
                    })
            except Exception as e:
                print(f"Forecast failed for {tm}/{br}/{mkt}: {e}")
                continue
    
    return pd.DataFrame(results)
