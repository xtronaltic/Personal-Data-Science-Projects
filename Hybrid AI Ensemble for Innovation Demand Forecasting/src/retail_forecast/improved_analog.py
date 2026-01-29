"""Improved analog forecaster with enhanced matching and weighting.

Key improvements over base analog_forecaster:
1. Dynamic Time Warping (DTW) distance for curve shape matching
2. Exponential decay weighting (more weight to similar analogs)
3. Scale-invariant matching (normalize curves before comparison)
4. Multi-factor weighting: distance + growth rate similarity + level similarity
5. Bootstrap resampling for uncertainty quantification
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from .constants import HIERARCHY_COLS
from .decomposition import add_decomposition_columns
from .io import derive_date_from_periods


@dataclass(frozen=True)
class ImprovedAnalogConfig:
    """Configuration for improved analog forecaster."""
    
    # Similarity
    top_k: int = 30  # Number of analogs to retrieve
    min_analogs: int = 5  # Minimum required analogs
    
    # Weighting
    distance_decay: float = 2.0  # Exponential decay factor for distance weighting
    growth_weight: float = 0.3  # Weight for growth rate similarity
    level_weight: float = 0.3  # Weight for level similarity
    shape_weight: float = 0.4  # Weight for curve shape similarity
    
    # Uncertainty
    n_bootstrap: int = 1000  # Number of bootstrap samples
    quantiles: tuple[float, ...] = (0.10, 0.50, 0.90)
    
    # Curve normalization
    normalize_curves: bool = True  # Scale-invariant matching
    
    # Horizon
    horizon: int = 26
    early_weeks: int = 4


def _normalize_curve(curve: np.ndarray) -> np.ndarray:
    """Normalize curve to [0, 1] range for scale-invariant comparison."""
    if curve.max() == curve.min():
        return np.zeros_like(curve)
    return (curve - curve.min()) / (curve.max() - curve.min() + 1e-8)


def _dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Dynamic Time Warping distance between two curves.
    
    DTW captures shape similarity even if curves are slightly misaligned in time.
    """
    n, m = len(a), len(b)
    
    # DTW matrix
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(a[i-1] - b[j-1])
            dtw[i, j] = cost + min(dtw[i-1, j], dtw[i, j-1], dtw[i-1, j-1])
    
    return dtw[n, m] / (n + m)  # Normalize by path length


def _euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance between curves."""
    return np.sqrt(np.sum((a - b) ** 2))


def _growth_rate(curve: np.ndarray) -> float:
    """Calculate average growth rate of a curve."""
    if len(curve) < 2:
        return 0.0
    # Use log returns for better numerical properties
    safe_curve = np.maximum(curve, 1e-6)
    log_returns = np.diff(np.log(safe_curve))
    return np.mean(log_returns)


def _compute_similarity_score(
    query_curve: np.ndarray,
    analog_curve: np.ndarray,
    config: ImprovedAnalogConfig,
) -> float:
    """Compute multi-factor similarity score between query and analog curves."""
    
    # Normalize for shape comparison
    if config.normalize_curves:
        query_norm = _normalize_curve(query_curve)
        analog_norm = _normalize_curve(analog_curve)
    else:
        query_norm = query_curve
        analog_norm = analog_curve
    
    # Shape similarity (DTW for robustness)
    shape_dist = _dtw_distance(query_norm, analog_norm)
    shape_sim = np.exp(-shape_dist * config.distance_decay)
    
    # Growth rate similarity
    query_growth = _growth_rate(query_curve)
    analog_growth = _growth_rate(analog_curve)
    growth_diff = abs(query_growth - analog_growth)
    growth_sim = np.exp(-growth_diff * 10)  # Sensitive to growth differences
    
    # Level similarity (ratio of means)
    query_mean = np.mean(query_curve) + 1e-8
    analog_mean = np.mean(analog_curve) + 1e-8
    level_ratio = min(query_mean, analog_mean) / max(query_mean, analog_mean)
    level_sim = level_ratio  # Already in [0, 1]
    
    # Weighted combination
    score = (
        config.shape_weight * shape_sim +
        config.growth_weight * growth_sim +
        config.level_weight * level_sim
    )
    
    return score


def _compute_weights(
    similarity_scores: np.ndarray,
    distances: np.ndarray,
    config: ImprovedAnalogConfig,
) -> np.ndarray:
    """Compute analog weights from similarity scores and distances."""
    
    # Exponential decay based on distance
    dist_weights = np.exp(-distances * config.distance_decay / (distances.mean() + 1e-8))
    
    # Combine with similarity scores
    combined = similarity_scores * dist_weights
    
    # Normalize to sum to 1
    total = combined.sum()
    if total > 0:
        return combined / total
    else:
        return np.ones_like(combined) / len(combined)


def _bootstrap_forecast(
    analog_futures: np.ndarray,  # (n_analogs, horizon)
    weights: np.ndarray,  # (n_analogs,)
    n_bootstrap: int = 1000,
    quantiles: tuple[float, ...] = (0.10, 0.50, 0.90),
) -> dict[float, np.ndarray]:
    """Generate forecast quantiles via weighted bootstrap resampling."""
    
    n_analogs, horizon = analog_futures.shape
    
    # Sample analogs with replacement, weighted by similarity
    rng = np.random.default_rng(42)
    bootstrap_samples = np.zeros((n_bootstrap, horizon))
    
    for i in range(n_bootstrap):
        # Sample analog indices weighted by weights
        sampled_indices = rng.choice(n_analogs, size=n_analogs, replace=True, p=weights)
        
        # Average the sampled analog curves
        sampled_curves = analog_futures[sampled_indices]
        
        # Add noise based on historical variance
        noise_scale = np.std(sampled_curves, axis=0) * 0.1
        noise = rng.normal(0, noise_scale)
        
        bootstrap_samples[i] = np.mean(sampled_curves, axis=0) + noise
    
    # Compute quantiles
    results = {}
    for q in quantiles:
        results[q] = np.percentile(bootstrap_samples, q * 100, axis=0)
    
    return results


def _scale_forecast_to_observed(
    forecast: np.ndarray,
    observed_values: np.ndarray,
) -> np.ndarray:
    """Scale forecast to match observed level.
    
    Key insight: If the new brand is selling 2x the analog average in weeks 1-4,
    we should expect it to continue selling ~2x in future weeks.
    """
    if len(observed_values) < 1 or len(forecast) < 1:
        return forecast
    
    # Use mean of observed vs mean of first few forecast points
    obs_mean = np.mean(observed_values)
    forecast_early_mean = np.mean(forecast[:min(4, len(forecast))])
    
    if forecast_early_mean < 1e-8 or obs_mean < 1e-8:
        return forecast
    
    # Calculate scale factor
    scale = obs_mean / forecast_early_mean
    
    # Apply gentler scaling - don't scale too aggressively
    # Use a dampened scale that moves toward 1.0
    dampened_scale = 0.5 * scale + 0.5  # Average with 1.0
    dampened_scale = np.clip(dampened_scale, 0.3, 3.0)
    
    return forecast * dampened_scale


class ImprovedAnalogForecaster:
    """Improved analog forecaster with enhanced curve matching."""
    
    def __init__(
        self,
        vel_library: pd.DataFrame,
        dist_library: pd.DataFrame,
        scaler_df: pd.DataFrame,
        config: ImprovedAnalogConfig = ImprovedAnalogConfig(),
    ):
        self.vel_library = vel_library.copy()
        self.dist_library = dist_library.copy()
        self.scaler_df = scaler_df.copy()
        self.config = config
        
        # Precompute velocity curves from JSON
        self._precompute_curves()
    
    def _precompute_curves(self) -> None:
        """Precompute analog curves for fast lookup."""
        self.vel_curves: dict[str, dict[str, np.ndarray]] = {}
        self.dist_curves: dict[str, np.ndarray] = {}
        
        for _, row in self.vel_library.iterrows():
            series_id = row["series_id"]
            self.vel_curves[series_id] = {
                "dollars": self._parse_json_curve(row.get("future_vel_dollars_json", "[]")),
                "units": self._parse_json_curve(row.get("future_vel_units_json", "[]")),
                "eq": self._parse_json_curve(row.get("future_vel_eq_json", "[]")),
            }
        
        for _, row in self.dist_library.iterrows():
            series_id = row["series_id"]
            self.dist_curves[series_id] = self._parse_json_curve(
                row.get("future_dist_acv_json", "[]")
            )
    
    def _parse_json_curve(self, s: Any) -> np.ndarray:
        """Parse JSON curve to numpy array."""
        if s is None or (isinstance(s, float) and np.isnan(s)):
            return np.array([])
        if isinstance(s, (list, tuple)):
            return np.array([0.0 if v is None else float(v) for v in s])
        try:
            parsed = json.loads(str(s))
            return np.array([0.0 if v is None else float(v) for v in parsed])
        except json.JSONDecodeError:
            return np.array([])
    
    def _get_analog_curves(
        self,
        trademark: str,
        metric: str,
        exclude_brand: str | None = None,
        category: str | None = None,
    ) -> tuple[list[str], np.ndarray]:
        """Get analog curves for a given trademark and metric.
        
        Prefers analogs from the same category first, then falls back.
        """
        
        # Start with category+trademark filter if possible
        mask = self.vel_library["trademark"] == trademark
        if category and "category" in self.vel_library.columns:
            mask &= self.vel_library["category"] == category
        if exclude_brand:
            mask &= self.vel_library["brand"] != exclude_brand
        
        candidates = self.vel_library[mask]
        
        if len(candidates) == 0:
            # Fallback: just trademark
            mask = self.vel_library["trademark"] == trademark
            if exclude_brand:
                mask &= self.vel_library["brand"] != exclude_brand
            candidates = self.vel_library[mask]
        
        if len(candidates) == 0:
            # Fallback to all
            candidates = self.vel_library
            if exclude_brand:
                candidates = candidates[candidates["brand"] != exclude_brand]
        
        series_ids = []
        curves = []
        
        for _, row in candidates.iterrows():
            series_id = row["series_id"]
            if series_id in self.vel_curves:
                curve = self.vel_curves[series_id].get(metric, np.array([]))
                if len(curve) >= self.config.horizon:
                    series_ids.append(series_id)
                    curves.append(curve[:self.config.horizon])
        
        if len(curves) == 0:
            return [], np.array([])
        
        return series_ids, np.vstack(curves)
    
    def _get_early_curve(
        self,
        df_new: pd.DataFrame,
        metric: str,
    ) -> np.ndarray:
        """Extract early weeks curve from new innovation data."""
        
        df = df_new.sort_values("date").head(self.config.early_weeks)
        
        # Map metric to the correct velocity column
        col_map = {
            "dollars": "vel_dollars",
            "units": "vel_units",
            "eq": "vel_eq",
            "dist_acv": "dist_acv",
        }
        
        col = col_map.get(metric, metric)
        
        if col in df.columns:
            values = df[col].values
        else:
            # Fallback to raw columns if velocity not computed
            raw_map = {
                "vel_dollars": "dollars_per_mm_acv",
                "vel_units": "units_per_mm_acv",
                "vel_eq": "eq_per_mm_acv",
                "dist_acv": "acv_pct",
            }
            fallback_col = raw_map.get(col, col)
            if fallback_col in df.columns:
                values = df[fallback_col].values
            else:
                values = np.zeros(len(df))
        
        return np.array(values, dtype=float)
    
    def forecast(
        self,
        df_new: pd.DataFrame,
        exclude_brand: str | None = None,
    ) -> pd.DataFrame:
        """Generate probabilistic forecast for new innovation."""
        
        # Ensure required columns
        if "date" not in df_new.columns and "periods" in df_new.columns:
            df_new = df_new.copy()
            df_new["date"] = derive_date_from_periods(df_new["periods"])
        
        if "vel_dollars" not in df_new.columns:
            df_new = add_decomposition_columns(df_new)
        
        results = []
        
        # Process each hierarchy group
        group_cols_present = [c for c in HIERARCHY_COLS if c in df_new.columns]
        if not group_cols_present:
            group_cols_present = ["markets", "trademark", "brand"]
        
        for group_values, group_df in df_new.groupby(group_cols_present, dropna=False, sort=False):
            if not isinstance(group_values, tuple):
                group_values = (group_values,)
            
            # Extract hierarchy values
            market = str(group_df["markets"].iloc[0]) if "markets" in group_df.columns else ""
            manufacturer = str(group_df["manufacturer"].iloc[0]) if "manufacturer" in group_df.columns else ""
            category = str(group_df["category"].iloc[0]) if "category" in group_df.columns else ""
            trademark = str(group_df["trademark"].iloc[0]) if "trademark" in group_df.columns else ""
            brand = str(group_df["brand"].iloc[0]) if "brand" in group_df.columns else ""
            
            # Exclude self from analogs if specified
            if exclude_brand is None:
                exclude_brand = brand
            
            # Forecast each metric
            for metric in ["dollars", "units", "eq"]:
                # Get analog curves (now with category)
                series_ids, analog_curves = self._get_analog_curves(
                    trademark, metric, exclude_brand, category=category
                )
                
                if len(series_ids) < self.config.min_analogs:
                    # Fallback: use all available (no category filter)
                    series_ids, analog_curves = self._get_analog_curves(
                        trademark, metric, None, category=None
                    )
                
                if len(series_ids) == 0:
                    continue
                
                # Get query curve (early weeks)
                query_curve = self._get_early_curve(group_df, metric)
                
                if len(query_curve) == 0:
                    continue
                
                # Compute similarity scores for each analog
                similarity_scores = np.array([
                    _compute_similarity_score(
                        query_curve,
                        curve[:len(query_curve)],
                        self.config,
                    )
                    for curve in analog_curves
                ])
                
                # Compute distances (for compatibility)
                distances = 1 - similarity_scores  # Convert similarity to distance
                
                # Compute weights
                weights = _compute_weights(similarity_scores, distances, self.config)
                
                # Select top-K analogs
                top_indices = np.argsort(-weights)[:self.config.top_k]
                top_curves = analog_curves[top_indices]
                top_weights = weights[top_indices]
                top_weights /= top_weights.sum()  # Renormalize
                
                # Bootstrap forecast
                quantile_forecasts = _bootstrap_forecast(
                    top_curves,
                    top_weights,
                    self.config.n_bootstrap,
                    self.config.quantiles,
                )
                
                # Scale forecasts to match observed level
                for q in quantile_forecasts:
                    quantile_forecasts[q] = _scale_forecast_to_observed(
                        quantile_forecasts[q],
                        query_curve,
                    )
                
                # Build result rows
                last_date = pd.to_datetime(group_df["date"].max())
                for h in range(self.config.horizon):
                    forecast_date = last_date + pd.Timedelta(weeks=h + 1)
                    row = {
                        "markets": market,
                        "manufacturer": manufacturer,
                        "category": category,
                        "trademark": trademark,
                        "brand": brand,
                        "week_ending": forecast_date,
                        "horizon_step": h + 1,
                        "metric": metric,
                    }
                    for q in self.config.quantiles:
                        col_name = f"p{int(q * 100)}"
                        row[col_name] = max(0, quantile_forecasts[q][h])
                    
                    results.append(row)
        
        return pd.DataFrame(results)


def load_improved_analog_forecaster(
    artifacts_dir: str | Path = "./artifacts",
    config: ImprovedAnalogConfig = ImprovedAnalogConfig(),
) -> ImprovedAnalogForecaster:
    """Load improved analog forecaster from artifacts."""
    artifacts_dir = Path(artifacts_dir)
    
    vel_library = pd.read_parquet(artifacts_dir / "vel_curve_library.parquet")
    dist_library = pd.read_parquet(artifacts_dir / "dist_curve_library.parquet")
    scaler_df = pd.read_parquet(artifacts_dir / "market_scalers.parquet")
    
    return ImprovedAnalogForecaster(vel_library, dist_library, scaler_df, config)
