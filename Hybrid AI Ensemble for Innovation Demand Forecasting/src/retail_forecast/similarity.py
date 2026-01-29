from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from .constants import HIERARCHY_COLS, HIERARCHY_SEPARATOR


Kind = Literal["dist", "vel"]


def _feature_cols(kind: Kind) -> list[str]:
    if kind == "dist":
        cols = [
            *[f"dist_acv_{i}" for i in range(4)],
            *[f"dist_tdp_{i}" for i in range(4)],
            "early_slope_dist_acv",
            "early_slope_dist_tdp",
        ]
        return cols

    # vel
    cols = [
        *[f"vel_dollars_{i}" for i in range(4)],
        *[f"vel_units_{i}" for i in range(4)],
        *[f"vel_eq_{i}" for i in range(4)],
        "early_slope_vel_dollars",
        "early_slope_vel_units",
        "early_slope_vel_eq",
        "early_mean_avg_unit_price",
        "early_mean_acv_pct",
        "early_mean_tdp",
    ]
    return cols


def _library_fingerprint(df_lib: pd.DataFrame, kind: Kind) -> str:
    """Stable fingerprint for model storage paths."""

    cols = [
        "series_id",
        *HIERARCHY_COLS,
        "launch_date",
        *_feature_cols(kind),
    ]
    present = [c for c in cols if c in df_lib.columns]
    base = df_lib[present].copy()

    # Normalize launch_date for stability
    if "launch_date" in base.columns:
        base["launch_date"] = pd.to_datetime(base["launch_date"], errors="coerce").dt.strftime("%Y-%m-%d")

    # Normalize numeric columns
    for c in _feature_cols(kind):
        if c in base.columns:
            base[c] = pd.to_numeric(base[c], errors="coerce")

    h = pd.util.hash_pandas_object(base.fillna(""), index=False)
    digest = hashlib.sha256(h.values.tobytes()).hexdigest()

    max_launch = ""
    if "launch_date" in df_lib.columns:
        max_launch_ts = pd.to_datetime(df_lib["launch_date"], errors="coerce").max()
        if max_launch_ts is not pd.NaT:
            max_launch = str(max_launch_ts.date())

    final = hashlib.sha256(f"{kind}|{int(df_lib.shape[0])}|{max_launch}|{digest}".encode("utf-8")).hexdigest()
    return final


def featurize_library_rows(
    df_lib: pd.DataFrame,
    kind: Kind,
) -> tuple[np.ndarray, pd.DataFrame, StandardScaler]:
    """Featurize library rows and standardize them.

    Returns
    - X: standardized feature matrix
    - meta: identity columns for mapping neighbors back to series
    - scaler: fitted StandardScaler
    """

    cols = _feature_cols(kind)
    missing = [c for c in cols if c not in df_lib.columns]
    if missing:
        raise ValueError(f"featurize_library_rows({kind}): missing feature columns: {missing}")

    features = df_lib[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    scaler = StandardScaler()
    X = scaler.fit_transform(features.values)

    meta_cols = [c for c in ["series_id", *HIERARCHY_COLS, "launch_date"] if c in df_lib.columns]
    meta = df_lib[meta_cols].copy()

    return X, meta, scaler


def fit_indices_by_hierarchy(
    df_lib: pd.DataFrame,
    kind: Kind,
    k_neighbors: int = 50,
    models_root: str | Path = "./models/similarity",
) -> dict[str, int]:
    """Fit and persist NearestNeighbors indices per (category, trademark) hierarchy (+ a global index).

    Uses the full hierarchy: Markets > Manufacturer > Category > Trademark > Brand
    Indexes are built at the (category, trademark) level for optimal analog matching.
    
    Saves per (category, trademark) AND per target.
    Dist targets: acv, tdp
    Vel targets: dollars, units, eq
    """

    targets = ["acv", "tdp"] if kind == "dist" else ["dollars", "units", "eq"]

    fp = _library_fingerprint(df_lib, kind)
    root = Path(models_root) / fp / kind
    root.mkdir(parents=True, exist_ok=True)

    saved = 0

    def _fit_and_save(pool: pd.DataFrame, hierarchy_key: str) -> None:
        nonlocal saved

        if pool.empty:
            return

        X, meta, scaler = featurize_library_rows(pool, kind)

        nn = NearestNeighbors(metric="cosine")
        nn.fit(X)

        for target in targets:
            out_dir = root / hierarchy_key / target
            out_dir.mkdir(parents=True, exist_ok=True)

            joblib.dump(nn, out_dir / "index.joblib")
            joblib.dump(scaler, out_dir / "scaler.joblib")
            meta.to_parquet(out_dir / "meta.parquet", index=False)

            saved += 1

    # Per (category, trademark) indices - most specific matching
    group_cols = ["category", "trademark"]
    available_cols = [c for c in group_cols if c in df_lib.columns]
    
    if len(available_cols) == 2:
        for (category, trademark), g in df_lib.groupby(available_cols, dropna=False, sort=True):
            key = f"{category}||{trademark}"
            _fit_and_save(g.reset_index(drop=True), hierarchy_key=key)
    
    # Per-trademark indices (for backward compatibility and fallback)
    if "trademark" in df_lib.columns:
        for trademark, g in df_lib.groupby("trademark", dropna=False, sort=True):
            key = str(trademark)
            _fit_and_save(g.reset_index(drop=True), hierarchy_key=key)

    # Global index (for fallback)
    _fit_and_save(df_lib.reset_index(drop=True), hierarchy_key="__GLOBAL__")

    return {"fingerprint": fp, "saved_indices": saved}


# Backward compatibility alias
fit_indices_by_trademark = fit_indices_by_hierarchy


def _load_index_bundle(
    models_root: str | Path,
    fingerprint: str,
    kind: Kind,
    trademark: str,
    target: str,
) -> tuple[NearestNeighbors, StandardScaler, pd.DataFrame]:
    base = Path(models_root) / fingerprint / kind / trademark / target
    nn: NearestNeighbors = joblib.load(base / "index.joblib")
    scaler: StandardScaler = joblib.load(base / "scaler.joblib")
    meta = pd.read_parquet(base / "meta.parquet")
    return nn, scaler, meta


def query_neighbors(
    *,
    models_root: str | Path,
    fingerprint: str,
    kind: Kind,
    trademark: str,
    target: str,
    query_features: dict[str, float | int | None],
    category: str | None = None,
    top_k: int = 50,
    min_k: int = 15,
    allow_global_fallback: bool = True,
    other_trademark_penalty: float = 0.05,
    other_category_penalty: float = 0.10,
) -> pd.DataFrame:
    """Query neighbors with hierarchy-aware search and fallback.

    Search order:
    1. (category, trademark) - most specific match
    2. trademark only - fallback if category+trademark not found
    3. global pool - final fallback with penalties for different trademarks/categories
    """

    feature_cols = _feature_cols(kind)

    def build_q(scaler: StandardScaler) -> np.ndarray:
        row = [float(query_features.get(c, 0.0) or 0.0) for c in feature_cols]
        q = np.asarray(row, dtype=float).reshape(1, -1)
        return scaler.transform(q)

    # Try most specific: (category, trademark)
    primary: pd.DataFrame | None = None
    hierarchy_key = f"{category}||{trademark}" if category else None
    
    if hierarchy_key:
        try:
            nn, scaler, meta = _load_index_bundle(models_root, fingerprint, kind, hierarchy_key, target)
            qx = build_q(scaler)
            n_neighbors = min(int(top_k), int(meta.shape[0]))
            dist, idx = nn.kneighbors(qx, n_neighbors=n_neighbors, return_distance=True)
            primary = meta.iloc[idx.ravel()].copy()
            primary["distance"] = dist.ravel()
            primary["neighbor_same_trademark"] = True
            primary["neighbor_same_category"] = True
        except FileNotFoundError:
            pass
    
    # Fallback to trademark-only if category+trademark not found
    if primary is None:
        try:
            nn, scaler, meta = _load_index_bundle(models_root, fingerprint, kind, trademark, target)
            qx = build_q(scaler)
            n_neighbors = min(int(top_k), int(meta.shape[0]))
            dist, idx = nn.kneighbors(qx, n_neighbors=n_neighbors, return_distance=True)
            primary = meta.iloc[idx.ravel()].copy()
            primary["distance"] = dist.ravel()
            primary["neighbor_same_trademark"] = True
            # Mark same category if category column exists
            if category and "category" in primary.columns:
                primary["neighbor_same_category"] = primary["category"].astype("string") == str(category)
            else:
                primary["neighbor_same_category"] = False
        except FileNotFoundError:
            primary = None

    if primary is not None and int(primary.shape[0]) >= int(min_k):
        primary["rank"] = np.arange(1, primary.shape[0] + 1)
        primary["kind"] = kind
        primary["target"] = target
        return primary.reset_index(drop=True)

    if not allow_global_fallback:
        if primary is not None:
            primary["rank"] = np.arange(1, primary.shape[0] + 1)
            primary["kind"] = kind
            primary["target"] = target
            return primary.reset_index(drop=True)
        else:
            raise FileNotFoundError(
                f"No similarity index for trademark={trademark} and global fallback disabled."
            )

    # Fallback: global pool (penalize other trademarks and categories)
    g_nn, g_scaler, g_meta = _load_index_bundle(models_root, fingerprint, kind, "__GLOBAL__", target)
    g_qx = build_q(g_scaler)
    g_n = min(int(top_k * 3), int(g_meta.shape[0]))
    g_dist, g_idx = g_nn.kneighbors(g_qx, n_neighbors=g_n, return_distance=True)

    global_neighbors = g_meta.iloc[g_idx.ravel()].copy()
    global_neighbors["distance"] = g_dist.ravel()
    global_neighbors["neighbor_same_trademark"] = global_neighbors["trademark"].astype("string") == str(trademark)
    
    # Check same category
    if category and "category" in global_neighbors.columns:
        global_neighbors["neighbor_same_category"] = global_neighbors["category"].astype("string") == str(category)
    else:
        global_neighbors["neighbor_same_category"] = False

    # Add penalties for different trademarks and categories
    global_neighbors.loc[~global_neighbors["neighbor_same_trademark"], "distance"] += float(other_trademark_penalty)
    global_neighbors.loc[~global_neighbors["neighbor_same_category"], "distance"] += float(other_category_penalty)

    # Combine with primary results if they exist
    if primary is not None and not primary.empty:
        combined = pd.concat([primary, global_neighbors], ignore_index=True)
    else:
        combined = global_neighbors
    combined = combined.drop_duplicates(subset=["series_id"], keep="first")
    combined = combined.sort_values("distance", kind="mergesort").head(int(top_k)).copy()

    combined["rank"] = np.arange(1, combined.shape[0] + 1)
    combined["kind"] = kind
    combined["target"] = target

    return combined.reset_index(drop=True)


def summarize_neighbors(neighbor_df: pd.DataFrame) -> pd.DataFrame:
    """Return top 10 neighbors with distance and launch context."""

    cols = [
        c
        for c in [
            "rank",
            "series_id",
            *HIERARCHY_COLS,
            "launch_date",
            "distance",
            "neighbor_same_trademark",
            "neighbor_same_category",
        ]
        if c in neighbor_df.columns
    ]
    out = neighbor_df[cols].copy()
    return out.head(10).reset_index(drop=True)
