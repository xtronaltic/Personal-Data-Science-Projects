from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class FileFingerprint:
    path: str
    size_bytes: int
    mtime_ns: int
    sha256: str


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def fingerprint_file(path: Path) -> FileFingerprint:
    stat = path.stat()
    return FileFingerprint(
        path=str(path.as_posix()),
        size_bytes=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
        sha256=sha256_file(path),
    )


def load_fingerprints(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_fingerprints(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def fingerprint_changed(prev: dict[str, Any] | None, current: FileFingerprint) -> bool:
    if not prev:
        return True
    # Treat any sha change as authoritative; also detect path/size/mtime changes.
    return (
        prev.get("sha256") != current.sha256
        or prev.get("size_bytes") != current.size_bytes
        or prev.get("mtime_ns") != current.mtime_ns
        or prev.get("path") != current.path
    )


def as_json_dict(fp: FileFingerprint) -> dict[str, Any]:
    return asdict(fp)
