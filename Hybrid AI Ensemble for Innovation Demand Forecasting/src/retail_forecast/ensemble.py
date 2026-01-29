from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .analog_forecaster import (
    EPS,
    AnalogComponentSims,
    _reconcile_dollars_units,
    simulate_new_innovation_analog_components,
)
from .io import derive_date_from_periods, fingerprint_df
from .neuralforecast_models import load_bundle, predict_components_for_new_innovation


MODEL_VERSION = "v1"


def _triangular_samples(rng: np.random.Generator, q10: float, q50: float, q90: float, n: int) -> np.ndarray:
    a = float(q10)
    c = float(q50)
    b = float(q90)

    # Ensure order for triangular
    lo = min(a, b, c)
    hi = max(a, b, c)
    mode = float(np.clip(c, lo, hi))

    if not np.isfinite(lo) or not np.isfinite(hi):
        return np.zeros(int(n), dtype=float)

    if hi <= lo:
        return np.full(int(n), float(lo), dtype=float)

    return rng.triangular(left=float(lo), mode=float(mode), right=float(hi), size=int(n)).astype(float)


def _default_weight_analog(observed_weeks: int) -> float:
    # Spec: 0.7 early weeks, decays toward 0.5 as more history available.
    if observed_weeks <= 4:
        return 0.7

    # decay 0.05 per extra observed week until 0.5
    w = 0.7 - 0.05 * float(observed_weeks - 4)
    return float(np.clip(w, 0.5, 0.7))


def _load_learned_weights(artifacts_dir: str | Path) -> pd.DataFrame | None:
    p = Path(artifacts_dir) / "ensemble_weights.csv"
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p)
        return df
    except Exception:
        return None


def _pick_weight(
    *,
    learned: pd.DataFrame | None,
    markets: str,
    trademark: str,
    component: str,
    observed_weeks: int,
) -> tuple[float, str]:
    if learned is not None and not learned.empty:
        need = {"markets", "trademark", "component", "weight_analog"}
        if need.issubset(set(learned.columns)):
            hit = learned[
                (learned["markets"].astype("string") == str(markets))
                & (learned["trademark"].astype("string") == str(trademark))
                & (learned["component"].astype("string") == str(component))
            ]
            if not hit.empty:
                w = float(pd.to_numeric(hit["weight_analog"].iloc[0], errors="coerce"))
                if np.isfinite(w):
                    return float(np.clip(w, 0.0, 1.0)), "learned"

    return _default_weight_analog(int(observed_weeks)), "default"


def _ensure_new_has_date(df_new: pd.DataFrame) -> pd.DataFrame:
    out = df_new.copy()
    if "date" not in out.columns:
        if "periods" not in out.columns:
            raise ValueError("df_new: expected 'date' or 'periods'")
        out["date"] = derive_date_from_periods(out["periods"])
    else:
        out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    return out


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


@dataclass(frozen=True)
class EnsembleConfig:
    horizon: int = 26
    n_sims: int = 5000
    top_k: int = 50
    early_weeks: int = 4
    nf_models_dir: str = "./models/neuralforecast/h26_in26"
    artifacts_dir: str = "./artifacts"
    models_root_similarity: str = "./models/similarity"
    # Use TimesFM foundation model instead of NeuralForecast
    use_timesfm: bool = True
    # Horizon-aware TimesFM weights (from optimized backtest)
    timesfm_weights: dict | None = None


def forecast_new_innovation_hybrid_ensemble(
    df_new: pd.DataFrame,
    *,
    config: EnsembleConfig = EnsembleConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hybrid ensemble forecast.

    - Gets analog component samples (dist_acv, vel_*)
    - Gets NeuralForecast component quantiles
    - Converts NF quantiles -> pseudo-samples
    - Mixes samples at component level using dynamic weights
    - Recombines to dollars/units/eq using analog scalers (k_*)
    """

    df_new = _ensure_new_has_date(df_new)

    # Attach paths for analog simulator
    df_new.attrs.setdefault("artifacts_dir", config.artifacts_dir)
    df_new.attrs.setdefault("models_root", config.models_root_similarity)

    # Analog component sims
    sims, explain_analog = simulate_new_innovation_analog_components(
        df_new,
        horizon=int(config.horizon),
        early_weeks=int(config.early_weeks),
        top_k=int(config.top_k),
        n_sims=int(config.n_sims),
    )

    # NeuralForecast models (pretrained)
    models_dir = Path(config.nf_models_dir)
    bundles = {
        "dist_acv": load_bundle("dist_acv", models_dir),
        "vel_dollars": load_bundle("vel_dollars", models_dir),
        "vel_units": load_bundle("vel_units", models_dir),
        "vel_eq": load_bundle("vel_eq", models_dir),
    }

    nf_forecasts: dict[str, pd.DataFrame] = {}
    nf_forecasts["dist_acv"] = predict_components_for_new_innovation(
        bundles["dist_acv"],
        df_new,
        horizon=int(config.horizon),
        include_velocity_exog=False,
    )
    for comp in ["vel_dollars", "vel_units", "vel_eq"]:
        nf_forecasts[comp] = predict_components_for_new_innovation(
            bundles[comp],
            df_new,
            horizon=int(config.horizon),
            include_velocity_exog=True,
        )

    # Build lookup: (markets,manufacturer,category,trademark,brand,week_ending) -> (p10,p50,p90)
    def to_map(df: pd.DataFrame) -> dict[tuple[str, str, str, str, str, str], tuple[float, float, float]]:
        out: dict[tuple[str, str, str, str, str, str], tuple[float, float, float]] = {}
        for _, r in df.iterrows():
            # Handle both old column names (trademark, brand) and new ones (trademark, brand)
            tm = str(r.get("trademark", r.get("trademark", "")))
            br = str(r.get("brand", r.get("brand", "")))
            mfr = str(r.get("manufacturer", ""))
            cat = str(r.get("category", ""))
            k = (str(r["markets"]), mfr, cat, tm, br, str(r["week_ending"]))
            out[k] = (
                float(pd.to_numeric(r["p10"], errors="coerce")) if pd.notna(r["p10"]) else 0.0,
                float(pd.to_numeric(r["p50"], errors="coerce")) if pd.notna(r["p50"]) else 0.0,
                float(pd.to_numeric(r["p90"], errors="coerce")) if pd.notna(r["p90"]) else 0.0,
            )
        return out

    nf_maps = {comp: to_map(df) for comp, df in nf_forecasts.items()}

    learned = _load_learned_weights(config.artifacts_dir)
    reconcile_rule = _choose_reconcile_rule(config.artifacts_dir)

    rng = np.random.default_rng(1)

    run_date = pd.Timestamp(date.today()).normalize()
    fp_new = df_new.attrs.get("fingerprint")
    if fp_new is None:
        try:
            fp_new = fingerprint_df(df_new)
        except Exception:
            fp_new = ""

    forecasts: list[dict[str, Any]] = []
    explains: list[dict[str, Any]] = []

    # For each market-level simulation group
    for sim in sims:
        markets = sim.markets
        manufacturer = sim.manufacturer
        category = sim.category
        trademark = sim.trademark
        brand = sim.brand

        observed_weeks = int(sim.start_week)

        # Convert NF quantiles -> pseudo samples aligned to sim.week_endings
        def nf_samples(component: str) -> np.ndarray:
            arr = np.zeros((int(config.n_sims), int(config.horizon)), dtype=float)
            for j, we in enumerate(sim.week_endings):
                key = (markets, manufacturer, category, trademark, brand, str(we.date()))
                q10, q50, q90 = nf_maps[component].get(key, (0.0, 0.0, 0.0))
                arr[:, j] = _triangular_samples(rng, q10, q50, q90, int(config.n_sims))
            return arr

        nf_dist_acv = nf_samples("dist_acv")
        nf_vd = nf_samples("vel_dollars")
        nf_vu = nf_samples("vel_units")
        nf_ve = nf_samples("vel_eq")

        # Weights per component
        weights: dict[str, tuple[float, str]] = {}
        for comp in ["dist_acv", "vel_dollars", "vel_units", "vel_eq"]:
            weights[comp] = _pick_weight(
                learned=learned,
                markets=markets,
                trademark=trademark,
                component=comp,
                observed_weeks=observed_weeks,
            )

        # Mix samples (convex combination)
        def mix(a: np.ndarray, b: np.ndarray, w: float, *, clamp_0_100: bool = False) -> np.ndarray:
            out = w * a + (1.0 - w) * b
            if clamp_0_100:
                out = np.clip(out, 0.0, 100.0)
            else:
                out = np.maximum(out, 0.0)
            return out

        w_dist, w_src_dist = weights["dist_acv"]
        w_vd, w_src_vd = weights["vel_dollars"]
        w_vu, w_src_vu = weights["vel_units"]
        w_ve, w_src_ve = weights["vel_eq"]

        dist_acv_mix = mix(sim.dist_acv, nf_dist_acv, w_dist, clamp_0_100=True)
        vel_d_mix = mix(sim.vel_dollars, nf_vd, w_vd)
        vel_u_mix = mix(sim.vel_units, nf_vu, w_vu)
        vel_e_mix = mix(sim.vel_eq, nf_ve, w_ve)

        # Recombine to sales using scalers
        dollars = float(sim.k_dollars) * vel_d_mix * dist_acv_mix
        units = float(sim.k_units) * vel_u_mix * dist_acv_mix
        eq = float(sim.k_eq) * vel_e_mix * dist_acv_mix

        # Reconcile dollars vs units*price
        implied_price = np.median(dollars / np.maximum(units, EPS), axis=0)
        if sim.last_avg_unit_price > 0 and (
            float(np.nanmedian(implied_price)) > 4.0 * sim.last_avg_unit_price
            or float(np.nanmedian(implied_price)) < 0.25 * sim.last_avg_unit_price
        ):
            for s in range(int(config.n_sims)):
                d_adj, u_adj = _reconcile_dollars_units(
                    dollars[s, :],
                    units[s, :],
                    price=float(sim.last_avg_unit_price),
                    rule=reconcile_rule,
                )
                dollars[s, :] = d_adj
                units[s, :] = u_adj

        def quantiles(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            return (
                np.quantile(x, 0.10, axis=0),
                np.quantile(x, 0.50, axis=0),
                np.quantile(x, 0.90, axis=0),
            )

        d10, d50, d90 = quantiles(dollars)
        u10, u50, u90 = quantiles(units)
        e10, e50, e90 = quantiles(eq)

        for j, we in enumerate(sim.week_endings):
            for metric, p10, p50, p90 in [
                ("dollars", d10[j], d50[j], d90[j]),
                ("units", u10[j], u50[j], u90[j]),
                ("eq", e10[j], e50[j], e90[j]),
            ]:
                forecasts.append(
                    {
                        "run_date": str(run_date.date()),
                        "markets": markets,
                        "manufacturer": manufacturer,
                        "category": category,
                        "trademark": trademark,
                        "brand": brand,
                        "week_ending": str(we.date()),
                        "metric": metric,
                        "p10": float(p10),
                        "p50": float(p50),
                        "p90": float(p90),
                        "method": "hybrid_ensemble",
                        "model_version": MODEL_VERSION,
                        "fingerprint": str(fp_new),
                    }
                )

        # Explain weights
        for comp, (w, src) in weights.items():
            explains.append(
                {
                    "markets": markets,
                    "manufacturer": manufacturer,
                    "category": category,
                    "trademark": trademark,
                    "brand": brand,
                    "component": comp,
                    "weight_analog": float(w),
                    "weight_neuralforecast": float(1.0 - w),
                    "weight_source": src,
                    "observed_weeks": int(observed_weeks),
                    "nf_models_dir": str(models_dir),
                }
            )

    forecast_df = pd.DataFrame(forecasts)

    # Combine explain_analog + explain weights
    explain_df = pd.concat(
        [
            pd.DataFrame(explains),
            explain_analog.assign(section="analog_neighbors"),
        ],
        ignore_index=True,
        sort=False,
    )

    return forecast_df, explain_df
