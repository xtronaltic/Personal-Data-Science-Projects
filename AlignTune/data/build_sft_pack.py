import os
import re
import json
from pathlib import Path
from collections import deque
from datasets import load_dataset
from transformers import AutoTokenizer
import argparse

BASE = os.environ.get("BASE_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
SYSTEM_PROMPT = "You are a helpful, concise assistant."
MAX_SFT = int(os.environ.get("MAX_SFT", "120000"))
MAX_PER_SRC = int(os.environ.get("MAX_SFT_PER_SRC", "60000"))
NEAR_DUP_WINDOW = int(os.environ.get("SFT_NEAR_DUP_WINDOW", "10000"))
NEAR_DUP_JACCARD = float(os.environ.get("SFT_NEAR_DUP_JACCARD", "0.85"))

BAN_STEMS = [
    "answer briefly",
    "you are a helpful",
    "rewrite to be more professional",
    "summarize the key idea",
    "turn bullets into a short paragraph",
    "turns out you can turn bullets",
]
BAN_RE = re.compile("|".join([re.escape(s) for s in BAN_STEMS]), re.IGNORECASE)

TOXIC_STEMS = [
    "kill",
    "hate",
    "violence",
    "bomb",
    "weapon",
    "attack",
]
TOX_RE = re.compile("|".join([re.escape(s) for s in TOXIC_STEMS]), re.IGNORECASE)

SOURCES = [
    ("databricks/databricks-dolly-15k", "train"),
    ("yahma/alpaca-cleaned", "train"),
    ("Open-Orca/SlimOrca", "train"),
]

def has_banned(s: str) -> bool:
    return bool(BAN_RE.search(s or ""))

def looks_toxic(s: str) -> bool:
    return bool(TOX_RE.search(s or ""))

def jaccard(a: str, b: str) -> float:
    ta, tb = set(a.split()), set(b.split())
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / max(1, union)

def adapt_dolly(ex):
    instr = (ex.get("instruction") or "").strip()
    ctx = (ex.get("context") or "").strip()
    inp = ctx
    user = instr if not inp else f"{instr}\n{inp}"
    ans = (ex.get("response") or "").strip()
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
        {"role": "assistant", "content": ans},
    ]
    return msgs, instr, inp, ans

def adapt_alpaca(ex):
    instr = (ex.get("instruction") or "").strip()
    inp = (ex.get("input") or "").strip()
    user = instr if inp == "" else f"{instr}\n{inp}"
    ans = (ex.get("output") or "").strip()
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
        {"role": "assistant", "content": ans},
    ]
    return msgs, instr, inp, ans

def adapt_slimorca(ex):
    sys_prompt = (ex.get("system_prompt") or SYSTEM_PROMPT).strip()
    instr = (ex.get("instruction") or "").strip()
    inp = ""
    ans = (ex.get("response") or "").strip()
    msgs = [
        {"role": "system", "content": sys_prompt or SYSTEM_PROMPT},
        {"role": "user", "content": instr},
        {"role": "assistant", "content": ans},
    ]
    return msgs, instr, inp, ans

ADAPTERS = {
    "databricks/databricks-dolly-15k": adapt_dolly,
    "yahma/alpaca-cleaned": adapt_alpaca,
    "Open-Orca/SlimOrca": adapt_slimorca,
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=["fast", "balanced", "thorough"], default="balanced")
    ap.add_argument("--out", default="data/pack/sft.jsonl")
    ap.add_argument("--max_total", type=int, default=None, help="cap total kept examples (overrides preset if set)")
    ap.add_argument("--max_per_src", type=int, default=None, help="cap kept examples per source (overrides preset if set)")
    ap.add_argument("--max_scan_per_src", type=int, default=None, help="cap scanned examples per source before giving up")
    args = ap.parse_args()

    preset_sizes = {
        "fast": (20000, 8000),
        "balanced": (40000, 20000),
        "thorough": (120000, 60000),
    }
    preset_scan_limits = {
        "fast": 40000,
        "balanced": 120000,
        "thorough": 300000,
    }
    preset_dup_windows = {
        "fast": 2000,
        "balanced": 4000,
        "thorough": 10000,
    }

    max_total, max_per_src = preset_sizes[args.preset]
    max_scan_per_src = preset_scan_limits[args.preset]
    if args.max_total is not None:
        max_total = int(args.max_total)
    if args.max_per_src is not None:
        max_per_src = int(args.max_per_src)
    if args.max_scan_per_src is not None:
        max_scan_per_src = int(args.max_scan_per_src)
    env_total = os.environ.get("MAX_SFT")
    env_per_src = os.environ.get("MAX_SFT_PER_SRC")
    env_scan = os.environ.get("SFT_MAX_SCAN_PER_SRC")
    if env_total:
        try:
            max_total = int(env_total)
        except Exception:
            pass
    if env_per_src:
        try:
            max_per_src = int(env_per_src)
        except Exception:
            pass
    if env_scan:
        try:
            max_scan_per_src = int(env_scan)
        except Exception:
            pass
    dup_window = int(os.environ.get("SFT_NEAR_DUP_WINDOW", str(preset_dup_windows[args.preset])))
    dup_thresh = NEAR_DUP_JACCARD

    os.environ.setdefault("MAX_SFT", str(max_total))
    os.environ.setdefault("MAX_SFT_PER_SRC", str(max_per_src))

    out_dir = Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out)

    tok = AutoTokenizer.from_pretrained(BASE, use_fast=True)

    print(f"[SFT] effective caps: total={max_total} per_src={max_per_src} scan_cap={max_scan_per_src}")
    kept_total = 0
    recent_texts = deque(maxlen=dup_window)

    with out_path.open("w", encoding="utf-8") as wf:
        for hub, split in SOURCES:
            adapt = ADAPTERS[hub]
            kept_src = 0
            scanned_src = 0
            try:
                ds = load_dataset(hub, split=split, streaming=True)
            except Exception as e:
                print(f"[skip] {hub}: {e}")
                continue

            for ex in ds:
                scanned_src += 1
                out = adapt(ex)
                if not out:
                    continue
                msgs, instr, inp, ans = out

                for m in msgs:
                    if m["role"] == "assistant" and has_banned(m["content"]):
                        m["content"] = BAN_RE.sub("", m["content"]).strip()

                if any(m["role"] == "assistant" and looks_toxic(m["content"]) for m in msgs):
                    continue

                text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)

                if recent_texts:
                    sim = max(jaccard(text.lower(), t) for t in recent_texts)
                    if sim >= dup_thresh:
                        continue

                rec = {"text": text, "instruction": instr, "input": inp, "output": ans}
                wf.write(json.dumps(rec, ensure_ascii=False) + "\n")

                kept_src += 1
                kept_total += 1
                recent_texts.append(text.lower())

                if kept_total % 2000 == 0:
                    print(f"[SFT] kept_total={kept_total} (src={hub} kept={kept_src} scanned={scanned_src})", flush=True)

                if kept_src >= max_per_src or kept_total >= max_total:
                    break

                if scanned_src >= max_scan_per_src and kept_src == 0:
                    print(f"[SFT] giving up {hub}: scanned={scanned_src} kept={kept_src} (scan cap {max_scan_per_src})")
                    break

            print(f"[SFT] {hub}: kept {kept_src} (scanned {scanned_src})")
            if kept_total >= max_total:
                break

    print(f"[SFT] wrote {kept_total} examples (target {max_total}) -> {out_path}")

if __name__ == "__main__":
    main()
