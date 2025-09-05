import argparse
import json
from rag.retriever import RagStore

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", required=True)
    ap.add_argument("--query", required=True)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--root", default="rag/indices")
    ap.add_argument("--embed_backend", default="auto")
    ap.add_argument("--embed_model", default="BAAI/bge-small-en-v1.5")
    args = ap.parse_args()

    store = RagStore(root=args.root, collection=args.collection, embed_backend=args.embed_backend, embed_model=args.embed_model)
    res = store.search(args.query, k=args.k)
    out = [
        {"text": r.text[:200], "score": r.score, "source": r.meta.get("source"), "title": r.meta.get("title"), "url": r.meta.get("url")}
        for r in res
    ]
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()

