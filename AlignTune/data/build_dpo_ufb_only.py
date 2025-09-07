import os
import json
import re
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer
import argparse

BASE = os.environ.get("BASE_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
SYSTEM_PROMPT = "You are a helpful, concise assistant."
MAX_DPO = int(os.environ.get("MAX_DPO", "60000"))
NEAR_DUP_WINDOW = int(os.environ.get("DPO_NEAR_DUP_WINDOW", "20000"))
NEAR_DUP_JACCARD = float(os.environ.get("DPO_NEAR_DUP_JACCARD", "0.85"))

BAN_RE = re.compile(
    r"(answer briefly|you are a helpful|rewrite to be more professional|"
    r"summarize the key idea|turn bullets into a short paragraph|"
    r"turns out you can turn bullets)",
    re.I,
)
TOX_RE = re.compile(r"(kill|hate|violence|bomb|weapon|attack)", re.I)

def denoise(s: str) -> str:
    return BAN_RE.sub("", (s or "")).strip()

def to_str(x):
    """Normalize list/dict/str -> str (pick the best textual content)."""
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    if isinstance(x, (list, tuple)):
        cand = [to_str(e) for e in x]
        cand = [c for c in cand if c]
        if not cand:
            return ""
        return max(cand, key=len)
    if isinstance(x, dict):
        for k in ("text", "response", "content", "answer", "output", "generated_text"):
            v = x.get(k)
            if isinstance(v, str) and v.strip():
                return v
        parts = [v for v in x.values() if isinstance(v, str) and v.strip()]
        if parts:
            return " ".join(parts)
        return ""
    return str(x)

def jaccard(a: str, b: str) -> float:
    ta, tb = set(a.split()), set(b.split())
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / max(1, union)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=["fast", "balanced", "thorough"], default="balanced")
    ap.add_argument("--out", default="data/pack/dpo.jsonl")
    args = ap.parse_args()

    preset_sizes = {
        "fast": 15000,
        "balanced": 30000,
        "thorough": 60000,
    }
    max_dpo = preset_sizes[args.preset]
    os.environ.setdefault("MAX_DPO", str(max_dpo))

    out_dir = Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out)

    tok = AutoTokenizer.from_pretrained(BASE, use_fast=True)

    kept = 0
    recent_prompts = []

    with out_path.open("w", encoding="utf-8") as wf:
        ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs", streaming=True)
        for ex in ds:
            prompt = to_str(ex.get("prompt") or ex.get("question"))
            chosen = to_str(ex.get("chosen") or ex.get("response_j"))
            rejected = to_str(ex.get("rejected") or ex.get("response_k"))
            if not (prompt and chosen and rejected):
                continue

            chosen, rejected = denoise(chosen), denoise(rejected)
            if not (chosen and rejected):
                continue

            if TOX_RE.search(chosen) or TOX_RE.search(rejected):
                continue

            msgs = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
            prompt_serial = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

            if recent_prompts:
                sim = max(jaccard(prompt_serial.lower(), p) for p in recent_prompts[-NEAR_DUP_WINDOW:])
                if sim >= NEAR_DUP_JACCARD:
                    continue

            sid = (
                ex.get("id")
                or ex.get("pair_id")
                or ex.get("sample_id")
                or ex.get("source_id")
                or f"ufb:{kept}"
            )
            wf.write(
                json.dumps(
                    {
                        "prompt": prompt_serial,
                        "chosen": chosen,
                        "rejected": rejected,
                        "source": "HuggingFaceH4/ultrafeedback_binarized",
                        "source_id": sid,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

            kept += 1
            recent_prompts.append(prompt_serial.lower())

            if kept % 2000 == 0:
                print(f"[DPO/UFB] kept {kept}", flush=True)

            if kept >= max_dpo:
                break

    print(f"[DPO] wrote {kept} pairs -> {out_path}")

if __name__ == "__main__":
    main()
