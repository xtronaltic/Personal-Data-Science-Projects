import argparse
import json
from pathlib import Path
from rag.retriever import RagStore

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", required=True)
    ap.add_argument("--root", default="rag/indices")
    ap.add_argument("--k", type=int, default=3)
    args = ap.parse_args()

    store = RagStore(root=args.root, collection=args.collection)

    from rag.index import load_metadata

    meta = load_metadata(Path(args.root) / args.collection / "meta.jsonl")
    total = 0
    hits = 0
    for m in meta[:100]:
        total += 1
        query = (m.get("text") or "").split(". ")[0][:200]
        res = store.search(query, k=args.k)
        sources = {r.meta.get("id") or r.meta.get("source") for r in res}
        if (m.get("id") in sources) or (m.get("source") in sources):
            hits += 1

    recall = hits / max(1, total)
    out = {"collection": args.collection, "k": args.k, "n": total, "recall_at_k": recall}
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()

