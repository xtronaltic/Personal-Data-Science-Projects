from __future__ import annotations

from pathlib import Path
import sys


def _ensure_src_on_path() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    return repo_root


repo_root = _ensure_src_on_path()

import pandas as pd  # noqa: E402

from retail_forecast.similarity import fit_indices_by_hierarchy  # noqa: E402


def main() -> None:
    artifacts = repo_root / "artifacts"
    dist_path = artifacts / "dist_curve_library.parquet"
    vel_path = artifacts / "vel_curve_library.parquet"

    if not dist_path.exists() or not vel_path.exists():
        raise SystemExit(
            "Missing curve libraries. Run scripts/build_curve_libraries.py first. "
            f"dist={dist_path.exists()} vel={vel_path.exists()}"
        )

    dist_lib = pd.read_parquet(dist_path)
    vel_lib = pd.read_parquet(vel_path)

    dist_info = fit_indices_by_hierarchy(dist_lib, kind="dist", models_root=repo_root / "models" / "similarity")
    vel_info = fit_indices_by_hierarchy(vel_lib, kind="vel", models_root=repo_root / "models" / "similarity")

    print(
        "saved_counts="
        + str(
            {
                "dist_fingerprint": dist_info.get("fingerprint"),
                "dist_saved_indices": dist_info.get("saved_indices"),
                "vel_fingerprint": vel_info.get("fingerprint"),
                "vel_saved_indices": vel_info.get("saved_indices"),
            }
        )
    )


if __name__ == "__main__":
    main()
