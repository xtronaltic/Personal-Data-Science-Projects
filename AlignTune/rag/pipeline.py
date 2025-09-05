from __future__ import annotations
import time
from typing import List, Dict, Tuple
from .retriever import RagStore

def build_context_block(items: List[Dict], max_tokens: int, tok) -> str:
    parts: List[str] = []
    used = 0
    for i, it in enumerate(items, start=1):
        header = f"[DOC {i}] {it.get('title') or ''} ({it.get('source', '')})\n"
        body = it.get("text", "")
        segment = header + body
        tlen = len(tok(segment).get("input_ids", []))
        if used + tlen > max_tokens:
            break
        parts.append(segment)
        used += tlen
    if not parts:
        return ""
    return "\n\n".join(parts)

def retrieve_context(store: RagStore, query: str, k: int, ctx_tokens: int, tok) -> Tuple[str, List[Dict]]:
    start = time.time()
    results = store.search(query, k=k)
    payload = [
        {
            "text": r.text,
            "title": r.meta.get("title"),
            "source": r.meta.get("source"),
            "url": r.meta.get("url"),
            "score": r.score,
        }
        for r in results
    ]
    context = build_context_block(payload, max_tokens=ctx_tokens, tok=tok)
    latency_ms = int(1000 * (time.time() - start))
    return context, payload + [{"latency_ms": latency_ms}]
