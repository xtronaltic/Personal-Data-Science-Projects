"""Integration tests for the full forecast pipeline."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


class TestPipelineIntegration:
    """Integration tests for pipeline functions."""

    @pytest.fixture
    def sample_panel_data(self) -> pd.DataFrame:
        """Create synthetic RetailPanel-like data for testing."""
        np.random.seed(42)

        # Create 3 trademarks, 2 brands each, 3 markets, 20 weeks
        trademarks = ["TM_A", "TM_B", "TM_C"]
        brands_per_tm = {"TM_A": ["BRAND_A1", "BRAND_A2"], "TM_B": ["BRAND_B1", "BRAND_B2"], "TM_C": ["BRAND_C1", "BRAND_C2"]}
        markets = ["Market1", "Market2", "Market3"]
        weeks = pd.date_range("2024-01-06", periods=20, freq="W-SAT")

        rows = []
        for tm in trademarks:
            for brand in brands_per_tm[tm]:
                for market in markets:
                    for i, week in enumerate(weeks):
                        # Simulate launch ramp
                        ramp = min(1.0, (i + 1) / 8.0)
                        base_dollars = np.random.uniform(1000, 5000) * ramp
                        base_units = base_dollars / np.random.uniform(3, 8)

                        rows.append({
                            "markets": market,
                            "periods": f"1 w/e {week.strftime('%m/%d/%y')}",
                            "manufacturer": "TEST_MFG",
                            "category": "TTL SSD",
                            "trademark": tm,
                            "brand": brand,
                            "dollars": base_dollars,
                            "units": base_units,
                            "eq": base_units * 0.8,
                            "avg_unit_price": base_dollars / max(base_units, 1),
                            "acv_pct": np.random.uniform(5, 50) * ramp,
                            "tdp": np.random.uniform(10, 100) * ramp,
                            "dollars_per_mm_acv": np.random.uniform(50, 200),
                            "units_per_mm_acv": np.random.uniform(10, 50),
                            "eq_per_mm_acv": np.random.uniform(5, 30),
                        })

        df = pd.DataFrame(rows)
        return df

    def test_io_derive_date(self, sample_panel_data: pd.DataFrame) -> None:
        """Test date derivation from periods."""
        from retail_forecast.io import derive_date_from_periods

        dates = derive_date_from_periods(sample_panel_data["periods"])
        assert len(dates) == len(sample_panel_data)
        assert dates.notna().all()
        # All should be Saturdays
        assert (dates.dt.dayofweek == 5).all()

    def test_decomposition_adds_columns(self, sample_panel_data: pd.DataFrame) -> None:
        """Test decomposition column addition."""
        from retail_forecast.decomposition import add_decomposition_columns

        df = sample_panel_data.copy()
        result = add_decomposition_columns(df)

        assert "dist_acv" in result.columns
        assert "dist_tdp" in result.columns
        assert "vel_dollars" in result.columns
        assert "vel_units" in result.columns
        assert "vel_eq" in result.columns

    def test_market_scalers_estimation(self, sample_panel_data: pd.DataFrame) -> None:
        """Test market scaler estimation."""
        from retail_forecast.decomposition import add_decomposition_columns, estimate_market_scalers

        df = add_decomposition_columns(sample_panel_data)
        scalers = estimate_market_scalers(df)

        assert "markets" in scalers.columns
        assert "k_dollars" in scalers.columns
        assert "k_units" in scalers.columns
        assert "k_eq" in scalers.columns

        # Scalers should be finite
        assert np.isfinite(scalers["k_dollars"]).all()
        assert np.isfinite(scalers["k_units"]).all()
        assert np.isfinite(scalers["k_eq"]).all()

    def test_curve_library_build(self, sample_panel_data: pd.DataFrame) -> None:
        """Test curve library building."""
        from retail_forecast.decomposition import add_decomposition_columns
        from retail_forecast.curve_library import build_distribution_library, build_velocity_library
        from retail_forecast.io import derive_date_from_periods

        df = sample_panel_data.copy()
        df["date"] = derive_date_from_periods(df["periods"])
        df = add_decomposition_columns(df)

        dist_lib = build_distribution_library(df, min_weeks=8)
        vel_lib = build_velocity_library(df, min_weeks=8)

        # Should have some series
        assert len(dist_lib) > 0
        assert len(vel_lib) > 0

        # Should have early vectors
        assert "dist_acv_0" in dist_lib.columns
        assert "vel_dollars_0" in vel_lib.columns

    def test_similarity_fit_and_query(self, sample_panel_data: pd.DataFrame) -> None:
        """Test similarity index fitting and querying."""
        from retail_forecast.decomposition import add_decomposition_columns
        from retail_forecast.curve_library import build_velocity_library
        from retail_forecast.similarity import fit_indices_by_trademark, query_neighbors
        from retail_forecast.io import derive_date_from_periods

        df = sample_panel_data.copy()
        df["date"] = derive_date_from_periods(df["periods"])
        df = add_decomposition_columns(df)

        vel_lib = build_velocity_library(df, min_weeks=8)

        with tempfile.TemporaryDirectory() as tmpdir:
            models_root = Path(tmpdir) / "similarity"
            result = fit_indices_by_trademark(vel_lib, kind="vel", models_root=models_root)

            assert "fingerprint" in result
            assert result["saved_indices"] > 0

            # Query neighbors
            query_features = {f"vel_dollars_{i}": 100.0 for i in range(4)}
            query_features.update({f"vel_units_{i}": 25.0 for i in range(4)})
            query_features.update({f"vel_eq_{i}": 15.0 for i in range(4)})
            query_features["early_slope_vel_dollars"] = 10.0
            query_features["early_slope_vel_units"] = 2.0
            query_features["early_slope_vel_eq"] = 1.0
            query_features["early_mean_avg_unit_price"] = 4.0
            query_features["early_mean_acv_pct"] = 30.0
            query_features["early_mean_tdp"] = 50.0

            neighbors = query_neighbors(
                models_root=models_root,
                fingerprint=result["fingerprint"],
                kind="vel",
                trademark="TM_A",
                target="dollars",
                query_features=query_features,
                top_k=10,
                allow_global_fallback=True,
            )

            assert len(neighbors) > 0
            assert "distance" in neighbors.columns
            assert "series_id" in neighbors.columns

    def test_fingerprint_stability(self, sample_panel_data: pd.DataFrame) -> None:
        """Test that fingerprinting is deterministic."""
        from retail_forecast.io import fingerprint_df

        fp1 = fingerprint_df(sample_panel_data)
        fp2 = fingerprint_df(sample_panel_data)

        assert fp1 == fp2

        # Different data should give different fingerprint
        df_modified = sample_panel_data.copy()
        df_modified.loc[0, "dollars"] = 999999
        fp3 = fingerprint_df(df_modified)
        assert fp1 != fp3


class TestConformalCalibration:
    """Tests for conformal calibration."""

    def test_fit_conformal_deltas(self) -> None:
        """Test conformal delta fitting."""
        from retail_forecast.conformal import fit_conformal_deltas, ConformalConfig

        # Create mock backtest data
        np.random.seed(42)
        n = 200
        backtest_df = pd.DataFrame({
            "model_type": ["analog"] * n,
            "metric": ["dollars"] * n,
            "horizon_step": np.tile(np.arange(1, 15), n // 14 + 1)[:n],
            "y_true": np.random.uniform(1000, 5000, n),
            "p10": np.random.uniform(800, 3000, n),
            "p50": np.random.uniform(1500, 4000, n),
            "p90": np.random.uniform(2000, 6000, n),
        })

        config = ConformalConfig(min_samples=10)
        deltas = fit_conformal_deltas(backtest_df, config=config)

        assert len(deltas) > 0
        assert "delta_p10" in deltas.columns
        assert "delta_p90" in deltas.columns

    def test_apply_conformal_calibration(self) -> None:
        """Test conformal calibration application."""
        from retail_forecast.conformal import apply_conformal_calibration

        # Create mock forecast
        forecast_df = pd.DataFrame({
            "markets": ["M1", "M1", "M1"],
            "trademark": ["TM", "TM", "TM"],
            "brand": ["BR", "BR", "BR"],
            "metric": ["dollars", "dollars", "dollars"],
            "week_ending": ["2024-01-06", "2024-01-13", "2024-01-20"],
            "p10": [1000, 1100, 1200],
            "p50": [1500, 1600, 1700],
            "p90": [2000, 2100, 2200],
        })

        deltas_df = pd.DataFrame({
            "model_type": ["analog", "analog", "analog"],
            "metric": ["dollars", "dollars", "dollars"],
            "horizon_step": [1, 2, 3],
            "trademark": ["__GLOBAL__", "__GLOBAL__", "__GLOBAL__"],
            "delta_p10": [100, 100, 100],
            "delta_p90": [200, 200, 200],
        })

        result = apply_conformal_calibration(forecast_df, deltas_df, model_type="analog")

        # p10 should decrease, p90 should increase
        assert result["p10"].iloc[0] < forecast_df["p10"].iloc[0]
        assert result["p90"].iloc[0] > forecast_df["p90"].iloc[0]
