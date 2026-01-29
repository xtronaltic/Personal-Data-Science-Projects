from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    root: Path

    @property
    def dataset_dir(self) -> Path:
        return self.root / "Dataset"

    @property
    def artifacts_dir(self) -> Path:
        return self.root / "artifacts"

    @property
    def outputs_dir(self) -> Path:
        return self.root / "outputs"

    @property
    def panel_xlsx(self) -> Path:
        return self.dataset_dir / "Historical_Data.csv"

    @property
    def panel_clean_csv(self) -> Path:
        return self.artifacts_dir / "panel_clean.csv"

    @property
    def panel_meta_json(self) -> Path:
        return self.artifacts_dir / "panel_meta.json"

    @property
    def fingerprint_json(self) -> Path:
        return self.artifacts_dir / "fingerprints.json"


def get_paths() -> Paths:
    # Assumes scripts are run from within the repo; resilient to cwd changes.
    root = Path(__file__).resolve().parents[1]
    return Paths(root=root)
