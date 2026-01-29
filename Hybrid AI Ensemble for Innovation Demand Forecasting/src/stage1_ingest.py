from __future__ import annotations

from .config import get_paths
from .ingest_panel import ingest_if_needed


def main() -> None:
    paths = get_paths()
    rebuilt, meta = ingest_if_needed(
        panel_xlsx=paths.panel_xlsx,
        out_clean_csv=paths.panel_clean_csv,
        out_meta_json=paths.panel_meta_json,
        fingerprints_json=paths.fingerprint_json,
    )

    status = "rebuilt" if rebuilt else "up-to-date"
    # Intentionally minimal output (no DataFrame printing).
    print(f"Stage1 Panel ingest: {status}")
    print(f"Rows={meta.rows} Cols={meta.cols}")
    print(f"WeekEndingRange={meta.earliest_week_ending}..{meta.latest_week_ending}")


if __name__ == "__main__":
    main()
