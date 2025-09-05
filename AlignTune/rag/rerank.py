from __future__ import annotations
from typing import List, Tuple

def maybe_rerank(query: str, candidates: List[Tuple[str, float]], model_name: str | None = None, device: str = "cpu") -> List[Tuple[str, float]]:
    try:
        from sentence_transformers import CrossEncoder
    except Exception:
        return candidates
    try:
        name = model_name or "BAAI/bge-reranker-base"
        model = CrossEncoder(name, device=device)
        pairs = [(query, c[0]) for c in candidates]
        scores = model.predict(pairs)
        reranked = sorted(zip([c[0] for c in candidates], scores.tolist()), key=lambda x: x[1], reverse=True)
        return reranked
    except Exception:
        return candidates

