"""Unified state tracking for rebuild-only-if-changed logic."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_STATE_PATH = "artifacts/state.json"


def load_state(path: str | Path = DEFAULT_STATE_PATH) -> dict[str, Any]:
    """Load build state from disk. Returns empty dict if missing."""
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def save_state(state: dict[str, Any], path: str | Path = DEFAULT_STATE_PATH) -> None:
    """Persist build state atomically."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")
    tmp.replace(p)


def _now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def update_state(
    state: dict[str, Any],
    *,
    panel_file_sha256: str | None = None,
    panel_asof_date: str | None = None,
    scalers_fingerprint: str | None = None,
    dist_lib_fingerprint: str | None = None,
    vel_lib_fingerprint: str | None = None,
    similarity_dist_fingerprint: str | None = None,
    similarity_vel_fingerprint: str | None = None,
) -> dict[str, Any]:
    """Return updated state dict with provided fields (immutable)."""
    out = dict(state)
    ts = _now_iso()

    if panel_file_sha256 is not None:
        out["panel_file_sha256"] = panel_file_sha256
        out["panel_file_sha256_ts"] = ts
    if panel_asof_date is not None:
        out["panel_asof_date"] = panel_asof_date
    if scalers_fingerprint is not None:
        out["scalers_fingerprint"] = scalers_fingerprint
        out["scalers_built_ts"] = ts
    if dist_lib_fingerprint is not None:
        out["dist_lib_fingerprint"] = dist_lib_fingerprint
        out["dist_lib_built_ts"] = ts
    if vel_lib_fingerprint is not None:
        out["vel_lib_fingerprint"] = vel_lib_fingerprint
        out["vel_lib_built_ts"] = ts
    if similarity_dist_fingerprint is not None:
        out["similarity_dist_fingerprint"] = similarity_dist_fingerprint
        out["similarity_dist_built_ts"] = ts
    if similarity_vel_fingerprint is not None:
        out["similarity_vel_fingerprint"] = similarity_vel_fingerprint
        out["similarity_vel_built_ts"] = ts

    return out
