import os
import math
from typing import List, Iterable, Optional
import numpy as np

class BaseEmbedder:
    def __init__(self, device: str = "cpu"):
        self.device = device

    def encode(self, texts: List[str], batch_size: int = 32, normalize: bool = True) -> np.ndarray:
        raise NotImplementedError

class HashEmbedder(BaseEmbedder):

    def __init__(self, dim: int = 384):
        super().__init__(device="cpu")
        self.dim = dim

    def _tok(self, s: str) -> List[str]:
        return [t for t in s.lower().split() if t]

    def encode(self, texts: List[str], batch_size: int = 64, normalize: bool = True) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in self._tok(t):
                h = (hash(tok) % self.dim)
                out[i, h] += 1.0
        if normalize:
            n = np.linalg.norm(out, axis=1, keepdims=True) + 1e-12
            out = out / n
        return out

class STEmbedder(BaseEmbedder):
    def __init__(self, model_name: str, device: str = "cpu"):
        super().__init__(device=device)
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as e:
            raise RuntimeError("sentence-transformers is not installed") from e
        self.model = SentenceTransformer(model_name, device=device)

    def encode(self, texts: List[str], batch_size: int = 32, normalize: bool = True) -> np.ndarray:
        emb = self.model.encode(texts, batch_size=batch_size, convert_to_numpy=True, normalize_embeddings=normalize)
        return emb.astype(np.float32)

class HFEmbedder(BaseEmbedder):
    def __init__(self, model_name: str, device: str = "cpu"):
        super().__init__(device=device)
        from transformers import AutoModel, AutoTokenizer
        import torch

        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.mdl = AutoModel.from_pretrained(model_name)
        self.mdl.to(device)
        self.torch = torch

    def _mean_pool(self, last_hidden_state, attention_mask):
        mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        s = self.torch.sum(last_hidden_state * mask, dim=1)
        d = self.torch.clamp(mask.sum(dim=1), min=1e-9)
        return s / d

    def encode(self, texts: List[str], batch_size: int = 16, normalize: bool = True) -> np.ndarray:
        embs: List[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            enc = self.tok(chunk, padding=True, truncation=True, return_tensors="pt").to(self.mdl.device)
            with self.torch.inference_mode():
                out = self.mdl(**enc)
                pooled = self._mean_pool(out.last_hidden_state, enc["attention_mask"]) 
                if normalize:
                    pooled = self.torch.nn.functional.normalize(pooled, p=2, dim=1)
            embs.append(pooled.detach().cpu().numpy().astype(np.float32))
        return np.concatenate(embs, axis=0) if embs else np.zeros((0, 768), dtype=np.float32)

def make_embedder(backend: str = "auto", model_name: Optional[str] = None, device: str = "cpu") -> BaseEmbedder:
    backend = (backend or "auto").lower()
    model_name = model_name or "BAAI/bge-small-en-v1.5"
    if backend == "hash":
        return HashEmbedder()
    if backend == "sentence-transformers":
        return STEmbedder(model_name, device=device)
    if backend == "hf":
        return HFEmbedder(model_name, device=device)
    try:
        return STEmbedder(model_name, device=device)
    except Exception:
        try:
            return HFEmbedder(model_name, device=device)
        except Exception:
            return HashEmbedder()

