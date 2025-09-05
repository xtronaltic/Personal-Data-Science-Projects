import argparse
import json
from pathlib import Path
from typing import List
import numpy as np
from rag.embed import make_embedder
from rag.ingest import build_chunks
from rag.index import VectorIndex, save_metadata, write_manifest, IndexArtifacts

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Source directory or glob of docs")
    ap.add_argument("--collection", required=True)
    ap.add_argument("--out_root", default="rag/indices")
    ap.add_argument("--embed_backend", default="auto", help="auto|sentence-transformers|hf|hash")
    ap.add_argument("--embed_model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--chunk_tokens", type=int, default=300)
    ap.add_argument("--overlap_tokens", type=int, default=40)
    args = ap.parse_args()

    chunks = build_chunks(args.src, args.chunk_tokens, args.overlap_tokens)
    texts = [c.text for c in chunks]
    meta = [
        {"id": c.id, "text": c.text, "source": c.source, "title": c.title, "url": c.url}
        for c in chunks
    ]

    emb = make_embedder(args.embed_backend, args.embed_model, device=args.device)
    vecs = emb.encode(texts, normalize=True)

    idx = VectorIndex(dim=vecs.shape[1])
    idx.build(vecs)

    out_dir = Path(args.out_root) / args.collection
    out_dir.mkdir(parents=True, exist_ok=True)
    idx.save(out_dir)
    save_metadata(out_dir / "meta.jsonl", meta)
    write_manifest(out_dir, artifacts=IndexArtifacts(index_type="faiss", dim=int(vecs.shape[1]), size=int(vecs.shape[0]), backend=args.embed_backend, embed_model=args.embed_model))
    print(f"[RAG] built collection='{args.collection}' with {len(texts)} chunks -> {out_dir}")

if __name__ == "__main__":
    main()
