import argparse
import json
import re
from pathlib import Path

SCAF_INSTR = re.compile(r"(?i)\binstructions?:")
SCAF_IO = re.compile(r"(?i)\b(input|output):")
SCAF_TAGS = re.compile(
    r"(?i)(\[/?(?:instruction|response|input)\]|###\s*(?:instruction|response)|<<\s*sys\s*>>|<</\s*sys\s*>>|<\|/?(?:system|assistant|user|eot_id)\|>|</?s>)"
)

def scan_file(path: Path, max_examples: int = 10):
    total = 0
    bad_json = 0
    has_instr = 0
    has_io = 0
    has_tags = 0
    examples = []

    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            total += 1
            try:
                obj = json.loads(line)
            except Exception:
                bad_json += 1
                continue
            chosen = (obj.get("chosen") or "")
            rejected = (obj.get("rejected") or "")
            blob = chosen + "\n" + rejected
            if SCAF_INSTR.search(blob):
                has_instr += 1
            if SCAF_IO.search(blob):
                has_io += 1
            if SCAF_TAGS.search(blob):
                has_tags += 1
            if len(examples) < max_examples and (SCAF_INSTR.search(blob) or SCAF_IO.search(blob) or SCAF_TAGS.search(blob)):
                examples.append(i)

    return {
        "total": total,
        "bad_json": bad_json,
        "has_instruction": has_instr,
        "has_input_output": has_io,
        "has_role_or_tags": has_tags,
        "example_lines": examples,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="data/pack/dpo.jsonl")
    ap.add_argument("--max_examples", type=int, default=10)
    args = ap.parse_args()

    p = Path(args.file)
    if not p.exists():
        raise SystemExit(f"File not found: {p}")
    rep = scan_file(p, args.max_examples)
    total = rep["total"] or 1
    print("[audit] DPO contamination scan")
    print(f"  file: {p}")
    print(f"  total: {rep['total']} | bad_json: {rep['bad_json']}")
    print(
        f"  has_instruction: {rep['has_instruction']} ({rep['has_instruction']/total:.3%}) | "
        f"has_input_output: {rep['has_input_output']} ({rep['has_input_output']/total:.3%}) | "
        f"has_role_or_tags: {rep['has_role_or_tags']} ({rep['has_role_or_tags']/total:.3%})"
    )
    if rep["example_lines"]:
        print(f"  example lines: {', '.join(map(str, rep['example_lines']))}")

if __name__ == "__main__":
    main()
