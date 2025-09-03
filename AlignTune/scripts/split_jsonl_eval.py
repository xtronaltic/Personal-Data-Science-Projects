import argparse
import json
from pathlib import Path
import numpy as np
import bisect

def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue

def get_text(ex):
    if "text" in ex:
        return ex["text"]
    instr = (ex.get("instruction") or "").strip()
    inp = (ex.get("input") or "").strip()
    out = (ex.get("output") or "").strip()
    parts = [instr, inp, out]
    return "\n".join([p for p in parts if p])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", required=True, help="input JSONL (SFT pack)")
    ap.add_argument("--out", default="data/pack/sft.eval.jsonl", help="output eval JSONL")
    ap.add_argument("--n", type=int, default=400, help="target size (300–500 recommended)")
    ap.add_argument("--bins", type=int, default=5, help="length bins for balanced sampling")
    args = ap.parse_args()

    lengths = []
    items = []
    for ex in read_jsonl(args.inp):
        txt = get_text(ex)
        l = len(txt)
        lengths.append(l)
        items.append(ex)

    if not items:
        raise SystemExit("No items found.")

    qs = np.linspace(0, 100, args.bins + 1)
    edges = np.percentile(lengths, qs).tolist()

    bins = [[] for _ in range(args.bins)]
    for ex, l in zip(items, lengths):
        idx = min(args.bins - 1, max(0, bisect.bisect_right(edges, l) - 1))
        bins[idx].append(ex)

    per_bin = max(1, args.n // args.bins)
    sampled = []
    rng = np.random.default_rng(42)

    for b in bins:
        if not b:
            continue
        if len(b) <= per_bin:
            sampled.extend(b)
        else:
            idxs = rng.choice(len(b), size=per_bin, replace=False)
            sampled.extend([b[i] for i in idxs])

    if len(sampled) < args.n:
        rest = [ex for ex in items if ex not in sampled]
        k = min(len(rest), args.n - len(sampled))
        if k > 0:
            idxs = rng.choice(len(rest), size=k, replace=False)
            sampled.extend([rest[i] for i in idxs])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for ex in sampled[: args.n]:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Wrote {min(len(sampled), args.n)} examples -> {out_path}")

if __name__ == "__main__":
    main()
