"""Unified pipeline: rebuild artifacts/models only when upstream fingerprints change."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .curve_library import build_distribution_library, build_velocity_library, save_curve_libraries
from .decomposition import add_decomposition_columns, estimate_market_scalers
from .io import fingerprint_df
from .similarity import fit_indices_by_trademark
from .state import load_state, save_state, update_state


def _df_fingerprint_lightweight(df: pd.DataFrame) -> str:
    """Lightweight fingerprint over a parquet-style artifact."""
    rowcount = int(df.shape[0])
    cols_hash = hashlib.sha256(",".join(sorted(df.columns)).encode()).hexdigest()[:16]
    sample = df.head(100).to_json(orient="split", date_format="iso")
    sample_hash = hashlib.sha256(sample.encode()).hexdigest()[:32]
    return hashlib.sha256(f"{rowcount}|{cols_hash}|{sample_hash}".encode()).hexdigest()


# ---------------------------------------------------------------------------
# ensure_market_scalers
# ---------------------------------------------------------------------------


def ensure_market_scalers(
    panel_df: pd.DataFrame,
    state: dict[str, Any],
    *,
    artifacts_dir: str | Path = "artifacts",
) -> tuple[pd.DataFrame, dict[str, Any], bool]:
    """Build or reuse market scalers.

    Returns (scalers_df, updated_state, rebuilt: bool).
    """
    artifacts_dir = Path(artifacts_dir)
    scaler_path = artifacts_dir / "market_scalers.parquet"

    panel_fp = fingerprint_df(panel_df)
    cached_fp = state.get("panel_file_sha256")

    # Reuse if panel fingerprint unchanged and file exists
    if scaler_path.exists() and cached_fp == panel_fp:
        scalers = pd.read_parquet(scaler_path)
        return scalers, state, False

    # Rebuild
    decomposed = add_decomposition_columns(panel_df)
    scalers = estimate_market_scalers(decomposed)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    scalers.to_parquet(scaler_path, index=False)

    scalers_fp = _df_fingerprint_lightweight(scalers)
    new_state = update_state(state, panel_file_sha256=panel_fp, scalers_fingerprint=scalers_fp)
    return scalers, new_state, True


# ---------------------------------------------------------------------------
# ensure_curve_libraries
# ---------------------------------------------------------------------------


def ensure_curve_libraries(
    panel_df: pd.DataFrame,
    state: dict[str, Any],
    *,
    artifacts_dir: str | Path = "artifacts",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], bool]:
    """Build or reuse distribution/velocity curve libraries.

    Returns (dist_lib, vel_lib, updated_state, rebuilt: bool).
    """
    artifacts_dir = Path(artifacts_dir)
    dist_path = artifacts_dir / "dist_curve_library.parquet"
    vel_path = artifacts_dir / "vel_curve_library.parquet"

    panel_fp = fingerprint_df(panel_df)
    cached_fp = state.get("panel_file_sha256")

    if dist_path.exists() and vel_path.exists() and cached_fp == panel_fp:
        dist_lib = pd.read_parquet(dist_path)
        vel_lib = pd.read_parquet(vel_path)
        return dist_lib, vel_lib, state, False

    decomposed = add_decomposition_columns(panel_df)
    dist_lib = build_distribution_library(decomposed)
    vel_lib = build_velocity_library(decomposed)
    save_curve_libraries(dist_lib, vel_lib, artifacts_dir=artifacts_dir)

    dist_fp = _df_fingerprint_lightweight(dist_lib)
    vel_fp = _df_fingerprint_lightweight(vel_lib)
    new_state = update_state(
        state,
        panel_file_sha256=panel_fp,
        dist_lib_fingerprint=dist_fp,
        vel_lib_fingerprint=vel_fp,
    )
    return dist_lib, vel_lib, new_state, True


# ---------------------------------------------------------------------------
# ensure_similarity_indices
# ---------------------------------------------------------------------------


def ensure_similarity_indices(
    dist_lib: pd.DataFrame,
    vel_lib: pd.DataFrame,
    state: dict[str, Any],
    *,
    models_root: str | Path = "models/similarity",
) -> tuple[dict[str, Any], bool]:
    """Build or reuse similarity indices.

    Writes models/similarity/latest.json with fingerprints.

    Returns (updated_state, rebuilt: bool).
    """
    from .similarity import _library_fingerprint

    models_root = Path(models_root)
    latest_json = models_root / "latest.json"

    # Use the same fingerprint function as fit_indices_by_trademark
    dist_fp = _library_fingerprint(dist_lib, "dist")
    vel_fp = _library_fingerprint(vel_lib, "vel")

    cached_dist = state.get("similarity_dist_fingerprint")
    cached_vel = state.get("similarity_vel_fingerprint")

    # Check latest.json consistency
    if latest_json.exists():
        try:
            latest = json.loads(latest_json.read_text(encoding="utf-8"))
            if latest.get("dist_fingerprint") == dist_fp and latest.get("vel_fingerprint") == vel_fp:
                if cached_dist == dist_fp and cached_vel == vel_fp:
                    return state, False
        except (json.JSONDecodeError, OSError):
            pass

    # Rebuild indices - they return the fingerprints used for directory paths
    dist_result = fit_indices_by_trademark(dist_lib, kind="dist", models_root=models_root)
    vel_result = fit_indices_by_trademark(vel_lib, kind="vel", models_root=models_root)

    # Use the fingerprints returned by fit_indices_by_trademark (same as _library_fingerprint)
    dist_fp = dist_result["fingerprint"]
    vel_fp = vel_result["fingerprint"]

    # Write latest.json with the ACTUAL fingerprints used for directory paths
    models_root.mkdir(parents=True, exist_ok=True)
    latest_json.write_text(
        json.dumps({"dist_fingerprint": dist_fp, "vel_fingerprint": vel_fp}, indent=2),
        encoding="utf-8",
    )

    new_state = update_state(
        state,
        similarity_dist_fingerprint=dist_fp,
        similarity_vel_fingerprint=vel_fp,
    )
    return new_state, True


# ---------------------------------------------------------------------------
# run_full_pipeline (convenience)
# ---------------------------------------------------------------------------


def run_full_pipeline(
    panel_df: pd.DataFrame,
    *,
    artifacts_dir: str | Path = "artifacts",
    models_root: str | Path = "models/similarity",
    state_path: str | Path = "artifacts/state.json",
) -> dict[str, Any]:
    """Run ensure_* pipeline and persist state. Returns rebuild summary."""
    state = load_state(state_path)

    scalers, state, sc_rebuilt = ensure_market_scalers(panel_df, state, artifacts_dir=artifacts_dir)
    dist_lib, vel_lib, state, lib_rebuilt = ensure_curve_libraries(panel_df, state, artifacts_dir=artifacts_dir)
    state, sim_rebuilt = ensure_similarity_indices(dist_lib, vel_lib, state, models_root=models_root)

    save_state(state, state_path)

    return {
        "scalers_rebuilt": sc_rebuilt,
        "curve_libs_rebuilt": lib_rebuilt,
        "similarity_rebuilt": sim_rebuilt,
        "state": state,
    }
