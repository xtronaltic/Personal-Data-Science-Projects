from __future__ import annotations
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
from .embed import make_embedder
from .index import VectorIndex, load_metadata

@dataclass
class Retrieved:
    text: str
    score: float
    meta: Dict

class RagStore:
    def __init__(self, root: str | Path = "rag/indices", collection: Optional[str] = None,
                 embed_backend: str = "auto", embed_model: Optional[str] = None, device: str = "cpu"):
        self.root = Path(root)
        self.collection = collection
        self.embed_backend = embed_backend
        self.embed_model = embed_model
        self.device = device
        self._index: Optional[VectorIndex] = None
        self._meta: Optional[List[Dict]] = None
        self._embedder = None

    def available_collections(self) -> List[str]:
        if not self.root.exists():
            return []
        return sorted([p.name for p in self.root.iterdir() if p.is_dir() and (p / "manifest.json").exists()])

    def set_collection(self, name: str):
        self.collection = name
        self._index = None
        self._meta = None

    def _ensure_loaded(self):
        assert self.collection, "collection not set"
        cpath = self.root / self.collection
        if self._index is None:
            self._index = VectorIndex(dim=384)
            self._index.load(cpath)
        if self._meta is None:
            self._meta = load_metadata(cpath / "meta.jsonl")
        if self._embedder is None:
            self._embedder = make_embedder(self.embed_backend, self.embed_model, device=self.device)

    def search(self, query: str, k: int = 3) -> List[Retrieved]:
        self._ensure_loaded()
        assert self._index and self._meta
        q = self._embedder.encode([query], normalize=True)
        sims, inds = self._index.search(q, k=k)
        sims = sims[0]
        inds = inds[0]
        out: List[Retrieved] = []
        for s, i in zip(sims.tolist(), inds.tolist()):
            if i < 0 or i >= len(self._meta):
                continue
            meta = self._meta[i]
            out.append(Retrieved(text=meta.get("text", ""), score=float(s), meta=meta))
        return out
