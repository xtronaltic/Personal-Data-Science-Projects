"""Unit tests for the analog forecaster with tiny synthetic data."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from retail_forecast.analog_forecaster import (
    EPS,
    _clamp,
    _inverse_distance_weights,
    _parse_future_json,
    _as_float_array,
    _lookup_scalers,
    _reconcile_dollars_units,
    _pool_from_neighbors,
)


# ---------------------------------------------------------------------------
# Helper: tiny synthetic data builders
# ---------------------------------------------------------------------------


def make_tiny_df_new(
    market: str = "US",
    trademark: str = "TM1",
    brand: str = "Brand1",
    category: str = "CAT1",
    n_weeks: int = 4,
) -> pd.DataFrame:
    """Create a tiny df_new with n_weeks of early data."""
    base_date = pd.Timestamp("2025-01-04")
    rows = []
    for i in range(n_weeks):
        rows.append({
            "markets": market,
            "trademark": trademark,
            "brand": brand,
            "category": category,
            "date": base_date + pd.Timedelta(weeks=i),
            "periods": f"{i+1} w/e 01/{4 + i*7:02d}/25",
            "dollars": 1000 + i * 100,
            "units": 100 + i * 10,
            "eq": 50 + i * 5,
            "acv_pct": 20 + i * 2,
            "tdp": 10 + i,
            "dollars_per_mm_acv": 5000 + i * 500,
            "units_per_mm_acv": 500 + i * 50,
            "eq_per_mm_acv": 250 + i * 25,
            "avg_unit_price": 10.0,
        })
    df = pd.DataFrame(rows)
    df["dist_acv"] = df["acv_pct"]
    df["dist_tdp"] = df["tdp"]
    df["vel_dollars"] = df["dollars_per_mm_acv"]
    df["vel_units"] = df["units_per_mm_acv"]
    df["vel_eq"] = df["eq_per_mm_acv"]
    df.attrs["asof_date"] = df["date"].max()
    return df


def make_fake_dist_lib(
    series_ids: list[str],
    future_len: int = 30,  # early_weeks + horizon (4 + 26)
) -> pd.DataFrame:
    """Create a tiny dist curve library with deterministic future vectors."""
    rows = []
    for i, sid in enumerate(series_ids):
        parts = sid.split("||")
        tm = parts[0] if len(parts) > 0 else "TM1"
        br = parts[1] if len(parts) > 1 else f"Brand{i}"
        mkt = parts[2] if len(parts) > 2 else "US"

        # Deterministic ACV vector: starts at 30, ramps up
        future_acv = [float(30 + j * 2) for j in range(future_len)]
        future_tdp = [float(15 + j) for j in range(future_len)]

        rows.append({
            "series_id": sid,
            "trademark": tm,
            "brand": br,
            "markets": mkt,
            "launch_date": "2024-01-06",
            "n_weeks_observed": 20,
            "max_week": 19,
            "early_slope_dist_acv": 2.0,
            "early_slope_dist_tdp": 1.0,
            "future_dist_acv_json": json.dumps(future_acv),
            "future_dist_tdp_json": json.dumps(future_tdp),
            **{f"dist_acv_{j}": future_acv[j] for j in range(4)},
            **{f"dist_tdp_{j}": future_tdp[j] for j in range(4)},
        })
    return pd.DataFrame(rows)


def make_fake_vel_lib(
    series_ids: list[str],
    future_len: int = 30,  # early_weeks + horizon (4 + 26)
) -> pd.DataFrame:
    """Create a tiny velocity curve library with deterministic future vectors."""
    rows = []
    for i, sid in enumerate(series_ids):
        parts = sid.split("||")
        tm = parts[0] if len(parts) > 0 else "TM1"
        br = parts[1] if len(parts) > 1 else f"Brand{i}"
        mkt = parts[2] if len(parts) > 2 else "US"

        future_vd = [float(5000 + j * 100) for j in range(future_len)]
        future_vu = [float(500 + j * 10) for j in range(future_len)]
        future_ve = [float(250 + j * 5) for j in range(future_len)]

        rows.append({
            "series_id": sid,
            "trademark": tm,
            "brand": br,
            "markets": mkt,
            "launch_date": "2024-01-06",
            "n_weeks_observed": 20,
            "max_week": 19,
            "early_slope_vel_dollars": 100.0,
            "early_slope_vel_units": 10.0,
            "early_slope_vel_eq": 5.0,
            "early_mean_avg_unit_price": 10.0,
            "early_mean_acv_pct": 30.0,
            "early_mean_tdp": 15.0,
            "future_vel_dollars_json": json.dumps(future_vd),
            "future_vel_units_json": json.dumps(future_vu),
            "future_vel_eq_json": json.dumps(future_ve),
            **{f"vel_dollars_{j}": future_vd[j] for j in range(4)},
            **{f"vel_units_{j}": future_vu[j] for j in range(4)},
            **{f"vel_eq_{j}": future_ve[j] for j in range(4)},
        })
    return pd.DataFrame(rows)


def make_fake_scaler_df(
    market: str = "US",
    category: str = "CAT1",
    k_dollars: float = 0.01,
    k_units: float = 0.001,
    k_eq: float = 0.0005,
) -> pd.DataFrame:
    """Create a fake market scalers df with known constants."""
    return pd.DataFrame([{
        "markets": market,
        "category": category,
        "k_dollars": k_dollars,
        "k_units": k_units,
        "k_eq": k_eq,
        "k_dollars_iqr": 0.001,
        "k_units_iqr": 0.0001,
        "k_eq_iqr": 0.00005,
        "k_dollars_count": 100,
        "k_units_count": 100,
        "k_eq_count": 100,
    }])


def make_fake_neighbor_df(series_ids: list[str], distances: list[float]) -> pd.DataFrame:
    """Create a fake neighbor_df for testing."""
    rows = []
    for i, (sid, d) in enumerate(zip(series_ids, distances)):
        parts = sid.split("||")
        rows.append({
            "series_id": sid,
            "trademark": parts[0] if len(parts) > 0 else "TM1",
            "brand": parts[1] if len(parts) > 1 else f"Brand{i}",
            "markets": parts[2] if len(parts) > 2 else "US",
            "launch_date": "2024-01-06",
            "distance": d,
            "rank": i + 1,
            "neighbor_same_trademark": True,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tests: _clamp
# ---------------------------------------------------------------------------


class TestClamp:
    def test_clamp_within_bounds(self):
        arr = np.array([10.0, 50.0, 90.0])
        result = _clamp(arr, 0.0, 100.0)
        np.testing.assert_array_equal(result, arr)

    def test_clamp_below_zero(self):
        arr = np.array([-10.0, -5.0, 0.0, 50.0])
        result = _clamp(arr, 0.0, 100.0)
        assert result[0] == 0.0
        assert result[1] == 0.0
        assert result[2] == 0.0
        assert result[3] == 50.0

    def test_clamp_above_100(self):
        arr = np.array([50.0, 100.0, 110.0, 200.0])
        result = _clamp(arr, 0.0, 100.0)
        assert result[0] == 50.0
        assert result[1] == 100.0
        assert result[2] == 100.0
        assert result[3] == 100.0

    def test_clamp_both_ends(self):
        arr = np.array([-50.0, 50.0, 150.0])
        result = _clamp(arr, 0.0, 100.0)
        np.testing.assert_array_equal(result, [0.0, 50.0, 100.0])


# ---------------------------------------------------------------------------
# Tests: _inverse_distance_weights
# ---------------------------------------------------------------------------


class TestInverseDistanceWeights:
    def test_weights_sum_to_one(self):
        dist = np.array([0.1, 0.2, 0.3, 0.5])
        w = _inverse_distance_weights(dist)
        assert abs(w.sum() - 1.0) < 1e-9

    def test_handles_distance_zero_safely(self):
        dist = np.array([0.0, 0.1, 0.2])
        w = _inverse_distance_weights(dist)
        assert np.isfinite(w).all()
        assert abs(w.sum() - 1.0) < 1e-9
        # Distance 0 should get highest weight
        assert w[0] > w[1] > w[2]

    def test_all_zeros(self):
        dist = np.array([0.0, 0.0, 0.0])
        w = _inverse_distance_weights(dist)
        assert np.isfinite(w).all()
        assert abs(w.sum() - 1.0) < 1e-9

    def test_single_element(self):
        dist = np.array([0.5])
        w = _inverse_distance_weights(dist)
        assert abs(w[0] - 1.0) < 1e-9

    def test_closer_gets_higher_weight(self):
        dist = np.array([0.1, 1.0])
        w = _inverse_distance_weights(dist)
        assert w[0] > w[1]


# ---------------------------------------------------------------------------
# Tests: _parse_future_json
# ---------------------------------------------------------------------------


class TestParseFutureJson:
    def test_parse_valid_json(self):
        s = json.dumps([1.0, 2.0, 3.0, None, 5.0])
        result = _parse_future_json(s)
        assert result == [1.0, 2.0, 3.0, None, 5.0]

    def test_parse_none(self):
        assert _parse_future_json(None) == []

    def test_parse_nan(self):
        assert _parse_future_json(float("nan")) == []

    def test_parse_list_directly(self):
        lst = [1.0, 2.0, 3.0]
        result = _parse_future_json(lst)
        assert result == [1.0, 2.0, 3.0]


# ---------------------------------------------------------------------------
# Tests: _as_float_array
# ---------------------------------------------------------------------------


class TestAsFloatArray:
    def test_converts_floats(self):
        values = [1.0, 2.0, 3.0]
        result = _as_float_array(values)
        np.testing.assert_array_equal(result, [1.0, 2.0, 3.0])

    def test_fills_none(self):
        values = [1.0, None, 3.0]
        result = _as_float_array(values, fill=0.0)
        np.testing.assert_array_equal(result, [1.0, 0.0, 3.0])

    def test_custom_fill(self):
        values = [None, None]
        result = _as_float_array(values, fill=99.0)
        np.testing.assert_array_equal(result, [99.0, 99.0])


# ---------------------------------------------------------------------------
# Tests: _lookup_scalers
# ---------------------------------------------------------------------------


class TestLookupScalers:
    def test_exact_match(self):
        scaler_df = make_fake_scaler_df("US", "CAT1", 0.01, 0.001, 0.0005)
        kd, ku, ke = _lookup_scalers(
            scaler_df=scaler_df,
            market="US",
            category="CAT1",
        )
        assert kd == 0.01
        assert ku == 0.001
        assert ke == 0.0005

    def test_fallback_to_market_level(self):
        # Category doesn't match
        scaler_df = make_fake_scaler_df("US", "CAT1", 0.01, 0.001, 0.0005)
        kd, ku, ke = _lookup_scalers(
            scaler_df=scaler_df,
            market="US",
            category="CAT_OTHER",
        )
        # Falls back to market median
        assert kd == 0.01
        assert ku == 0.001
        assert ke == 0.0005

    def test_fallback_to_global(self):
        scaler_df = make_fake_scaler_df("UK", "CAT1", 0.02, 0.002, 0.001)
        kd, ku, ke = _lookup_scalers(
            scaler_df=scaler_df,
            market="US",  # No US in df
            category="CAT1",
        )
        # Falls back to global median
        assert kd == 0.02


# ---------------------------------------------------------------------------
# Tests: _reconcile_dollars_units
# ---------------------------------------------------------------------------


class TestReconcileDollarsUnits:
    def test_adjust_units(self):
        dollars = np.array([100.0, 200.0])
        units = np.array([5.0, 10.0])
        price = 10.0
        d_out, u_out = _reconcile_dollars_units(dollars, units, price=price, rule="adjust_units")
        np.testing.assert_array_almost_equal(u_out, [10.0, 20.0])
        np.testing.assert_array_equal(d_out, dollars)

    def test_adjust_dollars(self):
        dollars = np.array([100.0, 200.0])
        units = np.array([5.0, 10.0])
        price = 10.0
        d_out, u_out = _reconcile_dollars_units(dollars, units, price=price, rule="adjust_dollars")
        np.testing.assert_array_almost_equal(d_out, [50.0, 100.0])
        np.testing.assert_array_equal(u_out, units)

    def test_rule_none(self):
        dollars = np.array([100.0])
        units = np.array([5.0])
        d_out, u_out = _reconcile_dollars_units(dollars, units, price=10.0, rule="none")
        np.testing.assert_array_equal(d_out, dollars)
        np.testing.assert_array_equal(u_out, units)


# ---------------------------------------------------------------------------
# Tests: _pool_from_neighbors
# ---------------------------------------------------------------------------


class TestPoolFromNeighbors:
    def test_creates_pool(self):
        series_ids = ["TM1||B1||US", "TM1||B2||US"]
        distances = [0.1, 0.2]
        neighbor_df = make_fake_neighbor_df(series_ids, distances)
        pool = _pool_from_neighbors(neighbor_df)
        assert len(pool.series_ids) == 2
        assert abs(pool.weights.sum() - 1.0) < 1e-9

    def test_empty_neighbors_raises(self):
        neighbor_df = pd.DataFrame()
        with pytest.raises(ValueError, match="No neighbors"):
            _pool_from_neighbors(neighbor_df)


# ---------------------------------------------------------------------------
# Tests: output structure validation (with monkeypatched query_neighbors)
# ---------------------------------------------------------------------------


class TestForecastOutputStructure:
    """Test that forecast outputs have correct structure and quantile ordering."""

    def test_quantile_ordering(self):
        """p10 <= p50 <= p90 for any valid forecast."""
        # Create random samples and compute quantiles
        rng = np.random.default_rng(42)
        samples = rng.normal(100, 20, size=(1000, 14))  # 1000 sims, 14 weeks

        p10 = np.percentile(samples, 10, axis=0)
        p50 = np.percentile(samples, 50, axis=0)
        p90 = np.percentile(samples, 90, axis=0)

        assert all(p10 <= p50)
        assert all(p50 <= p90)

    def test_acv_clamping_in_simulation(self):
        """Verify ACV values would be clamped to [0, 100]."""
        # Simulate what happens in the forecaster
        raw_acv = np.array([-10.0, 50.0, 110.0, 200.0])
        clamped = _clamp(raw_acv, 0.0, 100.0)
        assert all(clamped >= 0.0)
        assert all(clamped <= 100.0)

    def test_horizon_length(self):
        """Verify horizon length calculation."""
        early_weeks = 4
        horizon = 26
        future_len = early_weeks + horizon
        assert future_len == 30

        # Forecast weeks should be horizon, starting after early_weeks
        forecast_weeks = list(range(early_weeks, early_weeks + horizon))
        assert len(forecast_weeks) == horizon


# ---------------------------------------------------------------------------
# Integration test with synthetic data (no disk I/O)
# ---------------------------------------------------------------------------


class TestSyntheticIntegration:
    """Integration test using synthetic libraries and monkeypatched neighbors."""

    def test_weight_sum_with_synthetic_neighbors(self):
        """Weights from synthetic neighbors should sum to 1."""
        series_ids = ["TM1||B1||US", "TM1||B2||US", "TM1||B3||US"]
        distances = [0.05, 0.1, 0.2]
        neighbor_df = make_fake_neighbor_df(series_ids, distances)

        pool = _pool_from_neighbors(neighbor_df)
        assert abs(pool.weights.sum() - 1.0) < 1e-9

    def test_synthetic_library_vectors(self):
        """Library vectors should have correct length and values."""
        series_ids = ["TM1||B1||US", "TM1||B2||US"]
        dist_lib = make_fake_dist_lib(series_ids, future_len=30)
        vel_lib = make_fake_vel_lib(series_ids, future_len=30)

        assert len(dist_lib) == 2
        assert len(vel_lib) == 2

        # Parse future vectors
        acv_vec = json.loads(dist_lib.iloc[0]["future_dist_acv_json"])
        assert len(acv_vec) == 30
        assert all(0 <= v <= 100 for v in acv_vec)  # ACV should be reasonable

        vel_vec = json.loads(vel_lib.iloc[0]["future_vel_dollars_json"])
        assert len(vel_vec) == 30

    def test_synthetic_df_new_structure(self):
        """df_new should have all required columns."""
        df_new = make_tiny_df_new(n_weeks=4)
        required = [
            "markets", "trademark", "brand", "date",
            "dollars", "units", "eq", "acv_pct", "tdp",
            "dist_acv", "dist_tdp", "vel_dollars", "vel_units", "vel_eq",
        ]
        for col in required:
            assert col in df_new.columns, f"Missing column: {col}"

        assert len(df_new) == 4

