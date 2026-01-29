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

from retail_forecast.decomposition import (  # noqa: E402
    add_decomposition_columns,
    estimate_market_scalers,
    quality_report,
)
from retail_forecast.io import load_panel  # noqa: E402


def main() -> None:
    panel_df = load_panel(repo_root / "Dataset" / "Historical_Data.csv")

    panel_df = add_decomposition_columns(panel_df)
    scalers = estimate_market_scalers(panel_df)

    artifacts_dir = repo_root / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    out_path = artifacts_dir / "market_scalers.parquet"
    try:
        scalers.to_parquet(out_path, index=False)
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "Failed to write parquet. Ensure `pyarrow` is installed in your environment. "
            f"Error: {e}"
        )

    qc = quality_report(panel_df, scalers, outputs_dir=repo_root / "outputs")

    print("scalers head(5):")
    print(scalers.head(5).to_string(index=False))

    price_stats = qc.get("price_ratio_stats", {})
    print(
        "price_ratio_stats="
        + str({k: price_stats.get(k) for k in ["count", "median", "p10", "p90", "iqr"]})
    )
    print(f"quality_checks_csv={qc.get('qc_path')}")
    print(f"market_scalers_parquet={out_path}")


if __name__ == "__main__":
    main()
