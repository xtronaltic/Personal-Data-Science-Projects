"""LightGBM-based forecaster with M5-competition-style feature engineering.

This module implements best practices from the M5 Forecasting competition:
1. Rich lag and rolling features
2. Calendar/temporal features
3. Hierarchical modeling (by trademark)
4. Quantile regression for probabilistic forecasts
5. Recursive and non-recursive prediction modes
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

from .decomposition import add_decomposition_columns


@dataclass(frozen=True)
class LGBConfig:
    """Configuration for LightGBM forecaster."""
    
    # Feature engineering
    lags: tuple[int, ...] = (1, 2, 3, 4, 7, 14, 21, 28)
    rolling_windows: tuple[int, ...] = (7, 14, 28)
    
    # Model parameters (M5 winning approach)
    n_estimators: int = 2000
    learning_rate: float = 0.02
    num_leaves: int = 63
    max_depth: int = -1
    min_child_samples: int = 30
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    
    # Quantiles for probabilistic forecasts
    quantiles: tuple[float, ...] = (0.10, 0.50, 0.90)
    
    # Training
    early_stopping_rounds: int = 100
    verbose: int = -1


def _add_calendar_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Add calendar features for seasonality capture."""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Week of year (capture seasonality)
    df["week_of_year"] = df[date_col].dt.isocalendar().week.astype(int)
    
    # Month
    df["month"] = df[date_col].dt.month
    
    # Quarter
    df["quarter"] = df[date_col].dt.quarter
    
    # Year (for trend)
    df["year"] = df[date_col].dt.year
    
    # Week number since start (for trend)
    min_date = df[date_col].min()
    df["weeks_since_start"] = ((df[date_col] - min_date).dt.days / 7).astype(int)
    
    # Sin/cos encoding for cyclical features
    df["week_sin"] = np.sin(2 * np.pi * df["week_of_year"] / 52)
    df["week_cos"] = np.cos(2 * np.pi * df["week_of_year"] / 52)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    
    return df


def _add_lag_features(
    df: pd.DataFrame,
    target_cols: list[str],
    group_cols: list[str],
    lags: tuple[int, ...],
) -> pd.DataFrame:
    """Add lagged features for each target column."""
    df = df.copy()
    df = df.sort_values(group_cols + ["date"])
    
    for target in target_cols:
        for lag in lags:
            col_name = f"{target}_lag_{lag}"
            df[col_name] = df.groupby(group_cols)[target].shift(lag)
    
    return df


def _add_rolling_features(
    df: pd.DataFrame,
    target_cols: list[str],
    group_cols: list[str],
    windows: tuple[int, ...],
    min_periods: int = 1,
) -> pd.DataFrame:
    """Add rolling mean and std features."""
    df = df.copy()
    df = df.sort_values(group_cols + ["date"])
    
    for target in target_cols:
        for window in windows:
            # Rolling mean
            mean_col = f"{target}_roll_mean_{window}"
            df[mean_col] = df.groupby(group_cols)[target].transform(
                lambda x: x.shift(1).rolling(window, min_periods=min_periods).mean()
            )
            
            # Rolling std
            std_col = f"{target}_roll_std_{window}"
            df[std_col] = df.groupby(group_cols)[target].transform(
                lambda x: x.shift(1).rolling(window, min_periods=min_periods).std()
            )
            
            # Rolling min/max
            min_col = f"{target}_roll_min_{window}"
            df[min_col] = df.groupby(group_cols)[target].transform(
                lambda x: x.shift(1).rolling(window, min_periods=min_periods).min()
            )
            
            max_col = f"{target}_roll_max_{window}"
            df[max_col] = df.groupby(group_cols)[target].transform(
                lambda x: x.shift(1).rolling(window, min_periods=min_periods).max()
            )
    
    return df


def _add_growth_features(
    df: pd.DataFrame,
    target_cols: list[str],
    group_cols: list[str],
) -> pd.DataFrame:
    """Add growth rate features."""
    df = df.copy()
    df = df.sort_values(group_cols + ["date"])
    
    for target in target_cols:
        # Week-over-week growth
        df[f"{target}_wow_growth"] = df.groupby(group_cols)[target].pct_change(1)
        
        # 4-week growth rate
        df[f"{target}_4w_growth"] = df.groupby(group_cols)[target].pct_change(4)
        
        # Momentum (acceleration)
        df[f"{target}_momentum"] = df[f"{target}_wow_growth"] - df.groupby(group_cols)[f"{target}_wow_growth"].shift(1)
    
    # Replace inf with nan
    for col in df.columns:
        if "growth" in col or "momentum" in col:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
    
    return df


def _add_trademark_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add trademark-level aggregate features."""
    df = df.copy()
    
    # Trademark average velocity
    tm_agg = df.groupby(["trademark", "date"]).agg({
        "vel_dollars": "mean",
        "vel_units": "mean",
        "dist_acv": "mean",
    }).reset_index()
    tm_agg.columns = ["trademark", "date", "tm_avg_vel_dollars", "tm_avg_vel_units", "tm_avg_dist_acv"]
    
    df = df.merge(tm_agg, on=["trademark", "date"], how="left")
    
    # Ratio to trademark average
    df["vel_dollars_vs_tm"] = df["vel_dollars"] / df["tm_avg_vel_dollars"].clip(lower=1e-6)
    df["dist_acv_vs_tm"] = df["dist_acv"] / df["tm_avg_dist_acv"].clip(lower=1e-6)
    
    return df


def _add_market_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add market-level aggregate features."""
    df = df.copy()
    
    # Market average velocity
    mkt_agg = df.groupby(["markets", "date"]).agg({
        "vel_dollars": "mean",
        "vel_units": "mean",
    }).reset_index()
    mkt_agg.columns = ["markets", "date", "mkt_avg_vel_dollars", "mkt_avg_vel_units"]
    
    df = df.merge(mkt_agg, on=["markets", "date"], how="left")
    
    return df


def _add_brand_age_feature(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Add brand age (weeks since launch) feature."""
    df = df.copy()
    df = df.sort_values(group_cols + ["date"])
    
    # Week number within each brand-market series
    df["brand_week_num"] = df.groupby(group_cols).cumcount() + 1
    
    # Normalized brand age (0 to 1 over first 26 weeks)
    df["brand_age_norm"] = (df["brand_week_num"] / 26).clip(upper=1.0)
    
    return df


def build_feature_df(
    df: pd.DataFrame,
    config: LGBConfig = LGBConfig(),
) -> pd.DataFrame:
    """Build feature-engineered dataframe for LightGBM training."""
    if not HAS_LGB:
        raise ImportError("lightgbm is required. Install with: pip install lightgbm")
    
    df = df.copy()
    
    # Ensure decomposition columns exist
    if "vel_dollars" not in df.columns:
        df = add_decomposition_columns(df)
    
    group_cols = ["trademark", "brand", "markets"]
    target_cols = ["vel_dollars", "vel_units", "vel_eq", "dist_acv"]
    
    # Add calendar features
    df = _add_calendar_features(df, date_col="date")
    
    # Add brand age
    df = _add_brand_age_feature(df, group_cols)
    
    # Add lag features
    df = _add_lag_features(df, target_cols, group_cols, config.lags)
    
    # Add rolling features
    df = _add_rolling_features(df, target_cols, group_cols, config.rolling_windows)
    
    # Add growth features
    df = _add_growth_features(df, target_cols, group_cols)
    
    # Add trademark-level features
    df = _add_trademark_features(df)
    
    # Add market-level features
    df = _add_market_features(df)
    
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Get list of feature columns for modeling."""
    exclude_cols = {
        # Identifiers
        "date", "periods", "markets", "manufacturer", "category",
        "trademark", "brand", "series_id",
        # Raw targets
        "dollars", "units", "eq", "vel_dollars", "vel_units", "vel_eq",
        "dist_acv", "dist_tdp", "acv_pct", "tdp",
        # Derived targets
        "recon_dollars", "recon_units", "recon_eq",
        # Other
        "avg_unit_price", "dollars_per_mm_acv", "units_per_mm_acv", "eq_per_mm_acv",
    }
    
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    return feature_cols


class LGBQuantileForecaster:
    """LightGBM-based quantile forecaster for probabilistic predictions."""
    
    def __init__(self, config: LGBConfig = LGBConfig()):
        if not HAS_LGB:
            raise ImportError("lightgbm is required. Install with: pip install lightgbm")
        
        self.config = config
        self.models: dict[str, dict[float, lgb.Booster]] = {}  # target -> quantile -> model
        self.feature_cols: list[str] = []
    
    def fit(
        self,
        df: pd.DataFrame,
        targets: list[str] = ["vel_dollars", "vel_units", "vel_eq", "dist_acv"],
        valid_df: pd.DataFrame | None = None,
    ) -> "LGBQuantileForecaster":
        """Train quantile regression models for each target."""
        
        # Build features
        df_feat = build_feature_df(df, self.config)
        self.feature_cols = get_feature_columns(df_feat)
        
        # Drop rows with missing features (first few weeks)
        min_lag = max(self.config.lags)
        df_train = df_feat[df_feat["brand_week_num"] > min_lag].copy()
        
        # Prepare validation set if provided
        valid_data = None
        if valid_df is not None:
            valid_feat = build_feature_df(valid_df, self.config)
            valid_data = valid_feat[valid_feat["brand_week_num"] > min_lag].copy()
        
        X_train = df_train[self.feature_cols].values
        
        for target in targets:
            self.models[target] = {}
            y_train = df_train[target].values
            
            for quantile in self.config.quantiles:
                print(f"Training {target} quantile={quantile}...")
                
                # LightGBM quantile regression
                params = {
                    "objective": "quantile",
                    "alpha": quantile,
                    "metric": "quantile",
                    "boosting_type": "gbdt",
                    "n_estimators": self.config.n_estimators,
                    "learning_rate": self.config.learning_rate,
                    "num_leaves": self.config.num_leaves,
                    "max_depth": self.config.max_depth,
                    "min_child_samples": self.config.min_child_samples,
                    "subsample": self.config.subsample,
                    "colsample_bytree": self.config.colsample_bytree,
                    "verbose": self.config.verbose,
                    "n_jobs": -1,
                    "random_state": 42,
                }
                
                model = lgb.LGBMRegressor(**params)
                
                # Validation set for early stopping
                if valid_data is not None:
                    X_val = valid_data[self.feature_cols].values
                    y_val = valid_data[target].values
                    model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        callbacks=[lgb.early_stopping(self.config.early_stopping_rounds, verbose=False)],
                    )
                else:
                    model.fit(X_train, y_train)
                
                self.models[target][quantile] = model
        
        return self
    
    def predict(
        self,
        df: pd.DataFrame,
        horizon: int = 26,
    ) -> pd.DataFrame:
        """Generate quantile predictions for future weeks."""
        
        # Build features for input data
        df_feat = build_feature_df(df, self.config)
        
        results = []
        group_cols = ["trademark", "brand", "markets"]
        
        for (tm, brand, market), group in df_feat.groupby(group_cols):
            group = group.sort_values("date").copy()
            last_date = pd.to_datetime(group["date"].max())
            
            # Recursive prediction
            current_data = group.copy()
            
            for h in range(1, horizon + 1):
                forecast_date = last_date + pd.Timedelta(weeks=h)
                
                # Create feature row for this horizon
                # Use last available features and update what we can
                last_row = current_data.iloc[-1:].copy()
                last_row["date"] = forecast_date
                last_row["brand_week_num"] = last_row["brand_week_num"].iloc[0] + h
                last_row["brand_age_norm"] = min(last_row["brand_week_num"].iloc[0] / 26, 1.0)
                
                # Update calendar features
                last_row["week_of_year"] = forecast_date.isocalendar().week
                last_row["month"] = forecast_date.month
                last_row["quarter"] = forecast_date.quarter
                last_row["year"] = forecast_date.year
                last_row["week_sin"] = np.sin(2 * np.pi * last_row["week_of_year"].iloc[0] / 52)
                last_row["week_cos"] = np.cos(2 * np.pi * last_row["week_of_year"].iloc[0] / 52)
                
                # Predict for each target and quantile
                row_result = {
                    "trademark": tm,
                    "brand": brand,
                    "markets": market,
                    "week_ending": forecast_date,
                    "horizon_step": h,
                }
                
                X_pred = last_row[self.feature_cols].values
                
                for target in self.models:
                    for quantile, model in self.models[target].items():
                        pred = model.predict(X_pred)[0]
                        pred = max(0, pred)  # Non-negative
                        col_name = f"{target}_q{int(quantile*100)}"
                        row_result[col_name] = pred
                
                results.append(row_result)
        
        return pd.DataFrame(results)
    
    def save(self, path: str | Path) -> None:
        """Save models to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save feature columns
        with open(path / "feature_cols.json", "w") as f:
            json.dump(self.feature_cols, f)
        
        # Save config
        with open(path / "config.json", "w") as f:
            json.dump({
                "lags": self.config.lags,
                "rolling_windows": self.config.rolling_windows,
                "n_estimators": self.config.n_estimators,
                "learning_rate": self.config.learning_rate,
                "quantiles": self.config.quantiles,
            }, f)
        
        # Save models
        for target, quantile_models in self.models.items():
            for quantile, model in quantile_models.items():
                model_path = path / f"{target}_q{int(quantile*100)}.txt"
                model.booster_.save_model(str(model_path))
    
    @classmethod
    def load(cls, path: str | Path) -> "LGBQuantileForecaster":
        """Load models from disk."""
        path = Path(path)
        
        # Load feature columns
        with open(path / "feature_cols.json", "r") as f:
            feature_cols = json.load(f)
        
        # Load config
        with open(path / "config.json", "r") as f:
            config_dict = json.load(f)
        
        config = LGBConfig(
            lags=tuple(config_dict["lags"]),
            rolling_windows=tuple(config_dict["rolling_windows"]),
            n_estimators=config_dict["n_estimators"],
            learning_rate=config_dict["learning_rate"],
            quantiles=tuple(config_dict["quantiles"]),
        )
        
        forecaster = cls(config)
        forecaster.feature_cols = feature_cols
        
        # Load models
        for model_file in path.glob("*.txt"):
            parts = model_file.stem.split("_q")
            target = parts[0]
            quantile = int(parts[1]) / 100
            
            if target not in forecaster.models:
                forecaster.models[target] = {}
            
            booster = lgb.Booster(model_file=str(model_file))
            # Wrap in LGBMRegressor-like interface
            forecaster.models[target][quantile] = _BoosterWrapper(booster)
        
        return forecaster


class _BoosterWrapper:
    """Wrapper to make Booster behave like trained LGBMRegressor."""
    
    def __init__(self, booster: lgb.Booster):
        self.booster_ = booster
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.booster_.predict(X)


def train_lgb_models(
    df: pd.DataFrame,
    output_dir: str | Path = "./models/lightgbm",
    config: LGBConfig = LGBConfig(),
) -> LGBQuantileForecaster:
    """Train and save LightGBM quantile forecaster."""
    
    output_dir = Path(output_dir)
    
    # Split into train/valid by time
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    max_date = df["date"].max()
    valid_cutoff = max_date - pd.Timedelta(weeks=14)
    
    train_df = df[df["date"] <= valid_cutoff].copy()
    valid_df = df[df["date"] > valid_cutoff].copy()
    
    print(f"Training set: {len(train_df)} rows, up to {valid_cutoff.date()}")
    print(f"Validation set: {len(valid_df)} rows, after {valid_cutoff.date()}")
    
    # Train models
    forecaster = LGBQuantileForecaster(config)
    forecaster.fit(train_df, valid_df=valid_df)
    
    # Save
    forecaster.save(output_dir)
    print(f"Models saved to {output_dir}")
    
    return forecaster
