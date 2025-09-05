import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np

def _ensure_dir(p: str | Path):
    Path(p).mkdir(parents=True, exist_ok=True)

def _try_import_faiss():
    try:
        import faiss 

        return faiss
    except Exception:
        return None

@dataclass
class IndexArtifacts:
    index_type: str  
    dim: int
    size: int
    backend: str
    embed_model: str

class VectorIndex:
    def __init__(self, dim: int, use_faiss: bool = True):
        self.dim = dim
        self.faiss = _try_import_faiss() if use_faiss else None
        self.index = None

    def build(self, vecs: np.ndarray):
        assert vecs.ndim == 2 and vecs.shape[1] == self.dim
        if self.faiss is not None:
            idx = self.faiss.IndexFlatIP(self.dim)
            idx.add(vecs.astype(np.float32))
            self.index = ("faiss", idx)
        else:
            from sklearn.neighbors import NearestNeighbors

            nn = NearestNeighbors(n_neighbors=10, metric="cosine")
            nn.fit(vecs)
            self.index = ("sklearn", nn, vecs)

    def search(self, q: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        assert self.index is not None
        if self.index[0] == "faiss":
            _, idx = self.index
            D, I = idx.search(q.astype(np.float32), k)
            return D, I
        else:
            _, nn, vecs = self.index
            dists, inds = nn.kneighbors(q, n_neighbors=k, return_distance=True)
            sims = 1.0 - dists
            return sims.astype(np.float32), inds.astype(np.int32)

    def save(self, out_dir: str | Path):
        out_dir = Path(out_dir)
        _ensure_dir(out_dir)
        if self.index is None:
            raise RuntimeError("index not built")
        if self.index[0] == "faiss":
            _, idx = self.index
            import faiss 

            faiss.write_index(idx, str(out_dir / "index.faiss"))
            (out_dir / "index.kind").write_text("faiss", encoding="utf-8")
        else:
            _, nn, vecs = self.index
            from joblib import dump

            dump(nn, out_dir / "index.joblib")
            np.save(out_dir / "vecs.npy", vecs)
            (out_dir / "index.kind").write_text("sklearn", encoding="utf-8")

    def load(self, in_dir: str | Path):
        in_dir = Path(in_dir)
        kind = (in_dir / "index.kind").read_text(encoding="utf-8").strip()
        if kind == "faiss":
            import faiss  

            idx = faiss.read_index(str(in_dir / "index.faiss"))
            self.index = ("faiss", idx)
            self.dim = idx.d
        else:
            from joblib import load

            nn = load(in_dir / "index.joblib")
            vecs = np.load(in_dir / "vecs.npy")
            self.index = ("sklearn", nn, vecs)
            self.dim = vecs.shape[1]

def save_metadata(meta_path: str | Path, items: List[Dict]):
    with open(meta_path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

def load_metadata(meta_path: str | Path) -> List[Dict]:
    out: List[Dict] = []
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out

def write_manifest(out_dir: str | Path, artifacts: IndexArtifacts):
    out = Path(out_dir) / "manifest.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(artifacts.__dict__, f, indent=2)


