import argparse
import json
from pathlib import Path

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
    ap.add_argument("--inp", required=True, help="input SFT JSONL")
    ap.add_argument("--out", required=True, help="output JSONL of top-N longest prompts")
    ap.add_argument("--n", type=int, default=200, help="number of longest items to select")
    args = ap.parse_args()

    items = []
    for ex in read_jsonl(args.inp):
        txt = get_text(ex)
        items.append((len(txt), ex))

    if not items:
        raise SystemExit("No items found in input file.")

    items.sort(key=lambda p: p[0], reverse=True)
    top = [ex for _, ex in items[: max(1, int(args.n))]]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for ex in top:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Wrote {len(top)} longest examples -> {out_path}")

if __name__ == "__main__":
    main()

