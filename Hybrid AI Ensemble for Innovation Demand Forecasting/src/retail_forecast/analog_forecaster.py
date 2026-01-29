from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .constants import HIERARCHY_COLS, HIERARCHY_SEPARATOR
from .decomposition import add_decomposition_columns
from .io import derive_date_from_periods, fingerprint_df
from .similarity import Kind, query_neighbors, summarize_neighbors
from . import similarity as similarity_mod


EPS = 1e-6
MODEL_VERSION = "v1"


def _load_fingerprints_from_latest(models_root: Path) -> tuple[str, str]:
    """Load dist/vel fingerprints from models/similarity/latest.json.

    This ensures we use the same fingerprints as the trained similarity indices.
    """
    latest_path = models_root / "latest.json"
    if not latest_path.exists():
        raise FileNotFoundError(
            f"latest.json not found at {latest_path}. "
            "Run pipeline.run_full_pipeline() or ensure_similarity_indices() first."
        )
    data = json.loads(latest_path.read_text(encoding="utf-8"))
    dist_fp = str(data["dist_fingerprint"])
    vel_fp = str(data["vel_fingerprint"])
    return dist_fp, vel_fp


def _clamp(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.minimum(np.maximum(x, lo), hi)


def _as_float_array(values: list[float | None], *, fill: float = 0.0) -> np.ndarray:
    out = np.asarray([fill if v is None else float(v) for v in values], dtype=float)
    return out


def _safe_mean(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    return float(np.mean(values))


def _slope_early(vec: np.ndarray) -> float:
    if vec.size <= 1:
        return 0.0
    return float((vec[-1] - vec[0]) / max(int(vec.size) - 1, 1))


def _inverse_distance_weights(dist: np.ndarray) -> np.ndarray:
    w = 1.0 / (dist.astype(float) + EPS)
    s = float(w.sum())
    if not np.isfinite(s) or s <= 0:
        return np.ones_like(w) / max(int(w.size), 1)
    return w / s


def _parse_future_json(s: Any) -> list[float | None]:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return []
    if isinstance(s, (list, tuple)):
        return [None if v is None else float(v) for v in s]
    return json.loads(str(s))


def _market_level_fallback_scalers(scaler_df: pd.DataFrame) -> pd.DataFrame:
    needed = ["markets", "k_dollars", "k_units", "k_eq"]
    missing = [c for c in needed if c not in scaler_df.columns]
    if missing:
        raise ValueError(f"market_scalers: missing columns: {missing}")

    out = (
        scaler_df.groupby(["markets"], dropna=False, sort=False)[["k_dollars", "k_units", "k_eq"]]
        .median(numeric_only=True)
        .reset_index()
    )
    return out


def _lookup_scalers(
    *,
    scaler_df: pd.DataFrame,
    market: str,
    category: str | None,
) -> tuple[float, float, float]:
    # Exact match when category is present
    if category is not None and "category" in scaler_df.columns:
        hit = scaler_df[
            (scaler_df["markets"].astype("string") == str(market))
            & (scaler_df["category"].astype("string") == str(category))
        ]
        if not hit.empty:
            row = hit.iloc[0]
            return float(row["k_dollars"]), float(row["k_units"]), float(row["k_eq"])

    # Market-level fallback
    fb = _market_level_fallback_scalers(scaler_df)
    hit2 = fb[fb["markets"].astype("string") == str(market)]
    if hit2.empty:
        # Grand fallback
        kd = float(pd.to_numeric(scaler_df["k_dollars"], errors="coerce").median())
        ku = float(pd.to_numeric(scaler_df["k_units"], errors="coerce").median())
        ke = float(pd.to_numeric(scaler_df["k_eq"], errors="coerce").median())
        return kd, ku, ke

    row = hit2.iloc[0]
    return float(row["k_dollars"]), float(row["k_units"]), float(row["k_eq"])


def _ensure_new_has_timeline(df_new: pd.DataFrame) -> pd.DataFrame:
    out = df_new.copy()

    if "date" not in out.columns:
        if "periods" not in out.columns:
            raise ValueError("df_new: expected 'date' or 'periods'")
        out["date"] = derive_date_from_periods(out["periods"])
    else:
        out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()

    req = HIERARCHY_COLS + ["date"]
    missing = [c for c in req if c not in out.columns]
    if missing:
        raise ValueError(f"df_new: missing required columns: {missing}")

    out = out.sort_values(HIERARCHY_COLS + ["date"], kind="mergesort")

    launch = (
        out.groupby(HIERARCHY_COLS, dropna=False, sort=False)["date"]
        .min()
        .rename("launch_date")
        .reset_index()
    )

    out = out.merge(launch, on=HIERARCHY_COLS, how="left", validate="m:1")
    out["weeks_since_launch"] = ((out["date"] - out["launch_date"]).dt.days // 7).astype("int64")

    return out


def _build_query_features_dist(g: pd.DataFrame, early_weeks: int) -> dict[str, float]:
    g2 = g.copy()
    g2["weeks_since_launch"] = pd.to_numeric(g2["weeks_since_launch"], errors="coerce")

    def vec(col: str) -> np.ndarray:
        s = (
            g2.set_index("weeks_since_launch")[col]
            .reindex(range(4))
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=float)
        )
        if int(s.size) != 4:
            s = np.pad(s, (0, max(0, 4 - int(s.size))), constant_values=0.0)[:4]
        return s

    acv = vec("dist_acv")
    tdp = vec("dist_tdp")

    feats: dict[str, float] = {}
    for i in range(4):
        feats[f"dist_acv_{i}"] = float(acv[i])
        feats[f"dist_tdp_{i}"] = float(tdp[i])

    feats["early_slope_dist_acv"] = _slope_early(acv[: min(int(early_weeks), 4)])
    feats["early_slope_dist_tdp"] = _slope_early(tdp[: min(int(early_weeks), 4)])
    return feats


def _build_query_features_vel(g: pd.DataFrame, early_weeks: int) -> dict[str, float]:
    g2 = g.copy()
    g2["weeks_since_launch"] = pd.to_numeric(g2["weeks_since_launch"], errors="coerce")

    def vec(col: str) -> np.ndarray:
        s = (
            g2.set_index("weeks_since_launch")[col]
            .reindex(range(4))
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=float)
        )
        if int(s.size) != 4:
            s = np.pad(s, (0, max(0, 4 - int(s.size))), constant_values=0.0)[:4]
        return s

    vd = vec("vel_dollars")
    vu = vec("vel_units")
    ve = vec("vel_eq")

    feats: dict[str, float] = {}
    for i in range(4):
        feats[f"vel_dollars_{i}"] = float(vd[i])
        feats[f"vel_units_{i}"] = float(vu[i])
        feats[f"vel_eq_{i}"] = float(ve[i])

    feats["early_slope_vel_dollars"] = _slope_early(vd[: min(int(early_weeks), 4)])
    feats["early_slope_vel_units"] = _slope_early(vu[: min(int(early_weeks), 4)])
    feats["early_slope_vel_eq"] = _slope_early(ve[: min(int(early_weeks), 4)])

    early_mask = g2["weeks_since_launch"].between(0, min(int(early_weeks), 4) - 1)
    feats["early_mean_avg_unit_price"] = float(pd.to_numeric(g2.loc[early_mask, "avg_unit_price"], errors="coerce").mean())
    feats["early_mean_acv_pct"] = float(pd.to_numeric(g2.loc[early_mask, "acv_pct"], errors="coerce").mean())
    feats["early_mean_tdp"] = float(pd.to_numeric(g2.loc[early_mask, "tdp"], errors="coerce").mean())

    for k in ["early_mean_avg_unit_price", "early_mean_acv_pct", "early_mean_tdp"]:
        if not np.isfinite(feats[k]):
            feats[k] = 0.0

    return feats


def _extract_library_future(
    *,
    df_lib: pd.DataFrame,
    series_ids: list[str],
    future_col: str,
    take_len: int,
) -> np.ndarray:
    rows = df_lib.set_index("series_id").loc[series_ids]
    futures: list[np.ndarray] = []
    for _, r in rows.iterrows():
        vec = _parse_future_json(r.get(future_col))
        arr = _as_float_array(vec, fill=0.0)
        if int(arr.size) < int(take_len):
            last = float(arr[-1]) if int(arr.size) > 0 else 0.0
            arr = np.pad(arr, (0, int(take_len) - int(arr.size)), constant_values=last)
        else:
            arr = arr[:take_len]
        futures.append(arr)
    if not futures:
        return np.zeros((0, int(take_len)), dtype=float)
    return np.vstack(futures)


def _choose_reconcile_rule(artifacts_dir: str | Path) -> str:
    cfg_path = Path(artifacts_dir) / "reconcile_config.json"
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            rule = str(cfg.get("rule", "adjust_units"))
            if rule in {"adjust_units", "adjust_dollars", "none"}:
                return rule
        except Exception:
            return "adjust_units"
    return "adjust_units"


def _reconcile_dollars_units(
    dollars: np.ndarray,
    units: np.ndarray,
    *,
    price: float,
    rule: str,
) -> tuple[np.ndarray, np.ndarray]:
    if rule == "none":
        return dollars, units

    price = float(price) if np.isfinite(price) and price > 0 else 0.0
    if price <= 0:
        return dollars, units

    if rule == "adjust_units":
        units_adj = dollars / max(price, EPS)
        return dollars, units_adj

    if rule == "adjust_dollars":
        dollars_adj = units * price
        return dollars_adj, units

    return dollars, units


@dataclass(frozen=True)
class _NeighborPool:
    neighbor_df: pd.DataFrame
    weights: np.ndarray
    series_ids: list[str]


@dataclass(frozen=True)
class AnalogComponentSims:
    markets: str
    manufacturer: str
    category: str
    trademark: str
    brand: str
    launch_date: pd.Timestamp
    start_week: int
    horizon: int
    week_endings: list[pd.Timestamp]

    # Component samples: shape = (n_sims, horizon)
    dist_acv: np.ndarray
    dist_tdp: np.ndarray
    vel_dollars: np.ndarray
    vel_units: np.ndarray
    vel_eq: np.ndarray

    # Context
    last_avg_unit_price: float
    k_dollars: float
    k_units: float
    k_eq: float


def _pool_from_neighbors(neighbor_df: pd.DataFrame) -> _NeighborPool:
    if neighbor_df.empty:
        raise ValueError("No neighbors returned")

    dist = pd.to_numeric(neighbor_df["distance"], errors="coerce").fillna(1.0).to_numpy(dtype=float)
    w = _inverse_distance_weights(dist)
    sids = neighbor_df["series_id"].astype("string").tolist()
    return _NeighborPool(neighbor_df=neighbor_df.reset_index(drop=True), weights=w, series_ids=sids)


def forecast_new_innovation_analog(
    df_new: pd.DataFrame,
    *,
    horizon: int = 26,
    early_weeks: int = 4,
    top_k: int = 50,
    n_sims: int = 5000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Forecast a NEW innovation using trademark-strict analog simulation.

    Refactored to use simulate_new_innovation_analog_components for core logic.

    Returns:
    - forecast_df: CSV-ready long form with quantiles per metric/week
    - explain_df: top neighbors per target with weights and scaling factors
    """
    # 1. Run simulation to get components
    sims, explain_df = simulate_new_innovation_analog_components(
        df_new,
        horizon=horizon,
        early_weeks=early_weeks,
        top_k=top_k,
        n_sims=n_sims,
    )

    # 2. Recombine components into final metrics (dollars, units, eq)
    artifacts_dir = Path(str(df_new.attrs.get("artifacts_dir", "./artifacts")))
    reconcile_rule = _choose_reconcile_rule(artifacts_dir)
    
    run_date = pd.Timestamp(date.today()).normalize()
    asof_date_used = df_new.attrs.get("asof_date_used")
    asof_date_used = pd.to_datetime(asof_date_used, errors="coerce").normalize() if asof_date_used is not None else pd.NaT
    
    # Optional: attach a fingerprint for traceability
    fp_new = df_new.attrs.get("fingerprint")
    if fp_new is None:
        try:
            # We need to reconstruct df for fingerprinting or just use empty if not available
            # In the original code, df was constructed from df_new.
            # We'll skip precise fingerprint regeneration here to keep it simple, 
            # or rely on what's in df_new.attrs if set by caller.
            fp_new = ""
        except Exception:
            fp_new = ""

    forecasts: list[dict[str, Any]] = []

    for sim in sims:
        # Recombine to sales
        # sim.dist_acv shape: (n_sims, horizon)
        # sim.vel_dollars shape: (n_sims, horizon)
        
        dollars = float(sim.k_dollars) * sim.vel_dollars * sim.dist_acv
        units = float(sim.k_units) * sim.vel_units * sim.dist_acv
        eq = float(sim.k_eq) * sim.vel_eq * sim.dist_acv

        # Reconcile dollars vs units*price
        # Check median implied price vs last observed price
        implied_price = np.median(dollars / np.maximum(units, EPS), axis=0)
        
        # Apply reconciliation if price deviates significantly
        if sim.last_avg_unit_price > 0 and (
            float(np.nanmedian(implied_price)) > 4.0 * sim.last_avg_unit_price
            or float(np.nanmedian(implied_price)) < 0.25 * sim.last_avg_unit_price
        ):
            # Apply per-simulation reconciliation
            for s in range(dollars.shape[0]):
                d_adj, u_adj = _reconcile_dollars_units(
                    dollars[s, :],
                    units[s, :],
                    price=float(sim.last_avg_unit_price),
                    rule=reconcile_rule,
                )
                dollars[s, :] = d_adj
                units[s, :] = u_adj

        # Compute quantiles
        def get_quantiles(data):
            return (
                np.quantile(data, 0.10, axis=0),
                np.quantile(data, 0.50, axis=0),
                np.quantile(data, 0.90, axis=0)
            )

        d10, d50, d90 = get_quantiles(dollars)
        u10, u50, u90 = get_quantiles(units)
        e10, e50, e90 = get_quantiles(eq)

        # Format output
        for i, we in enumerate(sim.week_endings):
            week_ending = pd.Timestamp(we).normalize()
            
            # Dollars
            forecasts.append({
                "run_date": str(run_date.date()),
                "asof_date_used": "" if asof_date_used is pd.NaT else str(asof_date_used.date()),
                "markets": sim.markets,
                "manufacturer": sim.manufacturer,
                "category": sim.category,
                "trademark": sim.trademark,
                "brand": sim.brand,
                "week_ending": str(week_ending.date()),
                "metric": "dollars",
                "p10": float(d10[i]),
                "p50": float(d50[i]),
                "p90": float(d90[i]),
                "method": "analog_decomposition",
                "model_version": MODEL_VERSION,
                "fingerprint": str(fp_new),
            })
            
            # Units
            forecasts.append({
                "run_date": str(run_date.date()),
                "asof_date_used": "" if asof_date_used is pd.NaT else str(asof_date_used.date()),
                "markets": sim.markets,
                "manufacturer": sim.manufacturer,
                "category": sim.category,
                "trademark": sim.trademark,
                "brand": sim.brand,
                "week_ending": str(week_ending.date()),
                "metric": "units",
                "p10": float(u10[i]),
                "p50": float(u50[i]),
                "p90": float(u90[i]),
                "method": "analog_decomposition",
                "model_version": MODEL_VERSION,
                "fingerprint": str(fp_new),
            })
            
            # Eq
            forecasts.append({
                "run_date": str(run_date.date()),
                "asof_date_used": "" if asof_date_used is pd.NaT else str(asof_date_used.date()),
                "markets": sim.markets,
                "manufacturer": sim.manufacturer,
                "category": sim.category,
                "trademark": sim.trademark,
                "brand": sim.brand,
                "week_ending": str(week_ending.date()),
                "metric": "eq",
                "p10": float(e10[i]),
                "p50": float(e50[i]),
                "p90": float(e90[i]),
                "method": "analog_decomposition",
                "model_version": MODEL_VERSION,
                "fingerprint": str(fp_new),
            })

    return pd.DataFrame(forecasts), explain_df


def simulate_new_innovation_analog_components(
    df_new: pd.DataFrame,
    *,
    horizon: int = 26,
    early_weeks: int = 4,
    top_k: int = 50,
    n_sims: int = 5000,
) -> tuple[list[AnalogComponentSims], pd.DataFrame]:
    """Run the analog simulator and return component-level sample arrays.

    Returns:
    - sims: list of AnalogComponentSims (one per Market/trademark/brand)
    - explain_df: neighbor/weight/scaling explanation table
    """

    if int(early_weeks) != 4:
        raise ValueError("early_weeks must be 4 (current similarity indices use 4-week features)")

    artifacts_dir = Path(str(df_new.attrs.get("artifacts_dir", "./artifacts")))
    models_root = Path(str(df_new.attrs.get("models_root", "./models/similarity")))

    dist_lib_path = artifacts_dir / "dist_curve_library.parquet"
    vel_lib_path = artifacts_dir / "vel_curve_library.parquet"
    scaler_path = artifacts_dir / "market_scalers.parquet"

    if not dist_lib_path.exists():
        raise FileNotFoundError(dist_lib_path)
    if not vel_lib_path.exists():
        raise FileNotFoundError(vel_lib_path)
    if not scaler_path.exists():
        raise FileNotFoundError(scaler_path)

    dist_lib = pd.read_parquet(dist_lib_path)
    vel_lib = pd.read_parquet(vel_lib_path)
    scaler_df = pd.read_parquet(scaler_path)

    # Use fingerprints from latest.json (must match trained indices)
    dist_fp, vel_fp = _load_fingerprints_from_latest(models_root)

    df = _ensure_new_has_timeline(df_new)
    df = add_decomposition_columns(df)

    rng = np.random.default_rng(0)

    explains: list[dict[str, Any]] = []
    sims_out: list[AnalogComponentSims] = []

    group_cols = HIERARCHY_COLS
    for group_values, g in df.groupby(group_cols, dropna=False, sort=False):
        # Unpack hierarchy values
        market_s = str(group_values[0])
        manufacturer_s = str(group_values[1]) if len(group_values) > 1 else ""
        category_s = str(group_values[2]) if len(group_values) > 2 else ""
        trademark_s = str(group_values[3]) if len(group_values) > 3 else ""
        brand_s = str(group_values[4]) if len(group_values) > 4 else ""

        g = g[g["weeks_since_launch"] >= 0].copy()
        g = g.sort_values("weeks_since_launch", kind="mergesort")

        launch_date = pd.to_datetime(g["launch_date"].iloc[0], errors="coerce").normalize()
        max_obs_week = int(pd.to_numeric(g["weeks_since_launch"], errors="coerce").max())
        start_week = max_obs_week + 1
        end_week = max_obs_week + int(horizon)
        sim_weeks = np.arange(start_week, end_week + 1, dtype=int)

        q_dist = _build_query_features_dist(g, early_weeks=early_weeks)
        q_vel = _build_query_features_vel(g, early_weeks=early_weeks)

        dist_neighbors_acv = query_neighbors(
            models_root=models_root,
            fingerprint=dist_fp,
            kind="dist",
            trademark=trademark_s,
            category=category_s,
            target="acv",
            query_features=q_dist,
            top_k=int(top_k),
        )
        dist_neighbors_tdp = query_neighbors(
            models_root=models_root,
            fingerprint=dist_fp,
            kind="dist",
            trademark=trademark_s,
            category=category_s,
            target="tdp",
            query_features=q_dist,
            top_k=int(top_k),
        )

        vel_neighbors_d = query_neighbors(
            models_root=models_root,
            fingerprint=vel_fp,
            kind="vel",
            trademark=trademark_s,
            target="dollars",
            query_features=q_vel,
            top_k=int(top_k),
            category=category_s,
        )
        vel_neighbors_u = query_neighbors(
            models_root=models_root,
            fingerprint=vel_fp,
            kind="vel",
            trademark=trademark_s,
            target="units",
            query_features=q_vel,
            top_k=int(top_k),
            category=category_s,
        )
        vel_neighbors_e = query_neighbors(
            models_root=models_root,
            fingerprint=vel_fp,
            kind="vel",
            trademark=trademark_s,
            target="eq",
            query_features=q_vel,
            top_k=int(top_k),
            category=category_s,
        )

        # Explain: top 10 per target with weights
        for target, ndf in [
            ("dist_acv", dist_neighbors_acv),
            ("dist_tdp", dist_neighbors_tdp),
            ("vel_dollars", vel_neighbors_d),
            ("vel_units", vel_neighbors_u),
            ("vel_eq", vel_neighbors_e),
        ]:
            top = summarize_neighbors(ndf)
            top_dist = pd.to_numeric(top["distance"], errors="coerce").fillna(1.0).to_numpy(dtype=float)
            top_w = _inverse_distance_weights(top_dist)
            for i in range(int(top.shape[0])):
                explains.append(
                    {
                        "markets": market_s,
                        "manufacturer": manufacturer_s,
                        "category": category_s,
                        "trademark": trademark_s,
                        "brand": brand_s,
                        "target": target,
                        "neighbor_series_id": str(top.loc[i, "series_id"]),
                        "neighbor_market": str(top.loc[i, "markets"]) if "markets" in top.columns else "",
                        "neighbor_manufacturer": str(top.loc[i, "manufacturer"]) if "manufacturer" in top.columns else "",
                        "neighbor_category": str(top.loc[i, "category"]) if "category" in top.columns else "",
                        "neighbor_trademark": str(top.loc[i, "trademark"]) if "trademark" in top.columns else "",
                        "neighbor_brand": str(top.loc[i, "brand"]) if "brand" in top.columns else "",
                        "launch_date": str(top.loc[i, "launch_date"]),
                        "distance": float(top.loc[i, "distance"]),
                        "weight": float(top_w[i]),
                        "neighbor_same_trademark": bool(top.loc[i, "neighbor_same_trademark"]),
                        "neighbor_same_category": bool(top.loc[i, "neighbor_same_category"]),
                    }
                )

        pool_dist_acv = _pool_from_neighbors(dist_neighbors_acv)
        pool_dist_tdp = _pool_from_neighbors(dist_neighbors_tdp)
        pool_vd = _pool_from_neighbors(vel_neighbors_d)
        pool_vu = _pool_from_neighbors(vel_neighbors_u)
        pool_ve = _pool_from_neighbors(vel_neighbors_e)

        take_len = int(end_week + 1)
        dist_acv_futures = _extract_library_future(
            df_lib=dist_lib,
            series_ids=pool_dist_acv.series_ids,
            future_col="future_dist_acv_json",
            take_len=take_len,
        )
        dist_tdp_futures = _extract_library_future(
            df_lib=dist_lib,
            series_ids=pool_dist_tdp.series_ids,
            future_col="future_dist_tdp_json",
            take_len=take_len,
        )

        vel_d_futures = _extract_library_future(
            df_lib=vel_lib,
            series_ids=pool_vd.series_ids,
            future_col="future_vel_dollars_json",
            take_len=take_len,
        )
        vel_u_futures = _extract_library_future(
            df_lib=vel_lib,
            series_ids=pool_vu.series_ids,
            future_col="future_vel_units_json",
            take_len=take_len,
        )
        vel_e_futures = _extract_library_future(
            df_lib=vel_lib,
            series_ids=pool_ve.series_ids,
            future_col="future_vel_eq_json",
            take_len=take_len,
        )

        new_dist_acv_early = (
            g.set_index("weeks_since_launch")["dist_acv"].reindex(range(4)).apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
        )
        new_dist_tdp_early = (
            g.set_index("weeks_since_launch")["dist_tdp"].reindex(range(4)).apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
        )
        new_vd_early = (
            g.set_index("weeks_since_launch")["vel_dollars"].reindex(range(4)).apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
        )
        new_vu_early = (
            g.set_index("weeks_since_launch")["vel_units"].reindex(range(4)).apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
        )
        new_ve_early = (
            g.set_index("weeks_since_launch")["vel_eq"].reindex(range(4)).apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
        )

        price_series = pd.to_numeric(g["avg_unit_price"], errors="coerce").dropna()
        last_price = float(price_series.iloc[-1]) if not price_series.empty else 0.0

        category = None
        if "category" in g.columns and g["category"].notna().any():
            category = str(g["category"].dropna().iloc[0])
        k_dollars, k_units, k_eq = _lookup_scalers(
            scaler_df=scaler_df,
            market=market_s,
            category=category,
        )

        idx_dist_acv = rng.choice(dist_acv_futures.shape[0], size=int(n_sims), replace=True, p=pool_dist_acv.weights)
        idx_dist_tdp = rng.choice(dist_tdp_futures.shape[0], size=int(n_sims), replace=True, p=pool_dist_tdp.weights)
        idx_vd = rng.choice(vel_d_futures.shape[0], size=int(n_sims), replace=True, p=pool_vd.weights)
        idx_vu = rng.choice(vel_u_futures.shape[0], size=int(n_sims), replace=True, p=pool_vu.weights)
        idx_ve = rng.choice(vel_e_futures.shape[0], size=int(n_sims), replace=True, p=pool_ve.weights)

        neigh_dist_acv_early_means = np.mean(dist_acv_futures[:, :4], axis=1)
        neigh_dist_tdp_early_means = np.mean(dist_tdp_futures[:, :4], axis=1)
        neigh_vd_early_means = np.mean(vel_d_futures[:, :4], axis=1)
        neigh_vu_early_means = np.mean(vel_u_futures[:, :4], axis=1)
        neigh_ve_early_means = np.mean(vel_e_futures[:, :4], axis=1)

        new_dist_acv_mean = _safe_mean(new_dist_acv_early)
        new_dist_tdp_mean = _safe_mean(new_dist_tdp_early)
        new_vd_mean = _safe_mean(new_vd_early)
        new_vu_mean = _safe_mean(new_vu_early)
        new_ve_mean = _safe_mean(new_ve_early)

        sim_dist_acv = np.zeros((int(n_sims), int(horizon)), dtype=float)
        sim_dist_tdp = np.zeros((int(n_sims), int(horizon)), dtype=float)
        sim_vd = np.zeros((int(n_sims), int(horizon)), dtype=float)
        sim_vu = np.zeros((int(n_sims), int(horizon)), dtype=float)
        sim_ve = np.zeros((int(n_sims), int(horizon)), dtype=float)

        for s in range(int(n_sims)):
            n_i = int(idx_dist_acv[s])
            neigh_future_acv = dist_acv_futures[n_i]
            delta = float(new_dist_acv_mean - float(neigh_dist_acv_early_means[n_i]))
            dist_acv_future = _clamp(neigh_future_acv + delta, 0.0, 100.0)

            n2_i = int(idx_dist_tdp[s])
            neigh_future_tdp = dist_tdp_futures[n2_i]
            scale_tdp = float((new_dist_tdp_mean + EPS) / (float(neigh_dist_tdp_early_means[n2_i]) + EPS))
            dist_tdp_future = np.maximum(neigh_future_tdp * scale_tdp, 0.0)

            vd_i = int(idx_vd[s])
            vu_i = int(idx_vu[s])
            ve_i = int(idx_ve[s])

            scale_vd = float((new_vd_mean + EPS) / (float(neigh_vd_early_means[vd_i]) + EPS))
            scale_vu = float((new_vu_mean + EPS) / (float(neigh_vu_early_means[vu_i]) + EPS))
            scale_ve = float((new_ve_mean + EPS) / (float(neigh_ve_early_means[ve_i]) + EPS))

            vel_d_future = np.maximum(vel_d_futures[vd_i] * scale_vd, 0.0)
            vel_u_future = np.maximum(vel_u_futures[vu_i] * scale_vu, 0.0)
            vel_e_future = np.maximum(vel_e_futures[ve_i] * scale_ve, 0.0)

            sim_dist_acv[s, :] = dist_acv_future[sim_weeks]
            sim_dist_tdp[s, :] = dist_tdp_future[sim_weeks]
            sim_vd[s, :] = vel_d_future[sim_weeks]
            sim_vu[s, :] = vel_u_future[sim_weeks]
            sim_ve[s, :] = vel_e_future[sim_weeks]

        week_endings = [
            (launch_date + pd.to_timedelta(int(w) * 7, unit="D")).normalize() for w in sim_weeks
        ]

        sims_out.append(
            AnalogComponentSims(
                markets=market_s,
                manufacturer=manufacturer_s,
                category=category_s,
                trademark=trademark_s,
                brand=brand_s,
                launch_date=launch_date,
                start_week=int(start_week),
                horizon=int(horizon),
                week_endings=week_endings,
                dist_acv=sim_dist_acv,
                dist_tdp=sim_dist_tdp,
                vel_dollars=sim_vd,
                vel_units=sim_vu,
                vel_eq=sim_ve,
                last_avg_unit_price=float(last_price),
                k_dollars=float(k_dollars),
                k_units=float(k_units),
                k_eq=float(k_eq),
            )
        )

    return sims_out, pd.DataFrame(explains)