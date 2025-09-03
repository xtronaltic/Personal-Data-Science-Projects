import os
import json
import argparse
import random
import math
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

def batched_generate(model, tok, prompts, batch_size=8, max_new=256, temp=0.0, top_p=1.0):
    outs = []
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i : i + batch_size]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=False).to(model.device)
        with torch.inference_mode():
            gen = model.generate(
                **enc,
                max_new_tokens=max_new,
                do_sample=(temp > 0),
                temperature=temp,
                top_p=top_p,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )
        for j in range(len(chunk)):
            in_len = (enc["input_ids"][j] != tok.pad_token_id).sum().item()
            outs.append(tok.decode(gen[j][in_len:], skip_special_tokens=True).strip())
    return outs

def load_peft_or_base(base, ckpt=None):
    tok = AutoTokenizer.from_pretrained(base, use_fast=True)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True)
    model = AutoModelForCausalLM.from_pretrained(base, device_map="auto", quantization_config=bnb)
    try:
        model.config.attn_implementation = "sdpa"
    except:
        pass
    if ckpt:
        model = PeftModel.from_pretrained(model, ckpt)
    model.eval()
    return model, tok

def gen(model, tok, prompt, max_new=256, temp=0.7, top_p=0.95):
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=(temp > 0),
            temperature=temp,
            top_p=top_p,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.eos_token_id,
        )
    return tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

def chat_prompt(tok, sys, user):
    msgs = [{"role": "system", "content": sys}, {"role": "user", "content": user}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

def _to_user_prompt(ex: dict) -> str:
    instr = (ex.get("instruction") or "").strip()
    inp = (ex.get("input") or "").strip()
    if instr and inp:
        return f"{instr}\n{inp}"
    return instr or inp

def _load_jsonl_prompts(path: str, n: int) -> list[str]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
            except Exception:
                continue
            out.append(_to_user_prompt(ex))
            if len(out) >= n:
                break
    return out

def load_prompts(n, prompts_file=None, long100_file=None, use_long100=False):
    if prompts_file and Path(prompts_file).exists():
        with open(prompts_file, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        return lines[:n]
    if use_long100 and long100_file and Path(long100_file).exists():
        return _load_jsonl_prompts(long100_file, n)
    prompts = [
        "Explain simply: what is a GPU?",
        "Write a two-line joke about GPUs.",
        "When should I prefer DPO over SFT?",
        "Explain LoRA to a junior engineer in 3 sentences.",
        "Compare SDPA vs Flash-Attn in one paragraph.",
        "What do temperature and top_p do? Give guidance.",
        "Summarize pros/cons of Python vs C++ for HPC.",
        "Draft a polite email declining a meeting due to conflict.",
        "Why must train/infer chat templates match?",
        "Explain overfitting vs underfitting simply.",
    ] * 50
    return prompts[:n]

def ci_normal(p, n, z=1.96):
    if n <= 0:
        return (0.0, 0.0)
    se = math.sqrt(max(1e-9, p * (1 - p) / n))
    return (max(0.0, p - z * se), min(1.0, p + z * se))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--ckpt_a", required=False, help="Model A adapter (e.g., SFT)")
    ap.add_argument("--ckpt_b", required=False, help="Model B adapter (e.g., DPO)")
    ap.add_argument("--ckpt", required=False, help="Legacy: single ckpt vs base")
    ap.add_argument("--judge", default="meta-llama/Llama-3.2-1B-Instruct")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--prompts_file", default=None)
    ap.add_argument("--use_long100", action="store_true", help="use Long-100 JSONL prompts for judging")
    ap.add_argument("--long100_file", default="data/pack/sft.eval.long100.jsonl")
    ap.add_argument("--temp", type=float, default=0.0, help="candidate generation temperature")
    ap.add_argument("--max_new", type=int, default=256)
    ap.add_argument("--judge_temp", type=float, default=0.0)
    ap.add_argument("--symmetric", action="store_true", help="judge both A>B and B>A and average win rates")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_cases", default=None, help="optional path to save per-case JSONL")
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    random.seed(args.seed)

    sys_prompt = "You are a helpful, concise assistant."
    prompts = load_prompts(args.n, args.prompts_file, args.long100_file, args.use_long100)

    if args.ckpt_a and args.ckpt_b:
        mA, tA = load_peft_or_base(args.base, args.ckpt_a)
        mB, tB = load_peft_or_base(args.base, args.ckpt_b)
        labelA, labelB = os.path.basename(args.ckpt_a), os.path.basename(args.ckpt_b)
    else:
        mA, tA = load_peft_or_base(args.base, args.ckpt)
        mB, tB = load_peft_or_base(args.base, None)
        labelA, labelB = os.path.basename(args.ckpt or "tuned"), "base"

    pAs = [chat_prompt(tA, sys_prompt, p) for p in prompts]
    pBs = [chat_prompt(tB, sys_prompt, p) for p in prompts]
    aAs = batched_generate(mA, tA, pAs, batch_size=8, max_new=args.max_new, temp=args.temp, top_p=1.0)
    aBs = batched_generate(mB, tB, pBs, batch_size=8, max_new=args.max_new, temp=args.temp, top_p=1.0)
    pairs = list(zip(prompts, aAs, aBs))

    del mA, tA, mB, tB
    torch.cuda.empty_cache()

    jm, jt = load_peft_or_base(args.judge, None)

    def judge(prompt, A, B):
        q = jt.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": f"Prompt:\n{prompt}\n\nResponse A:\n{A}\n\nResponse B:\n{B}\n\n"
                    "Which is better overall? Answer with a single letter: A or B.",
                }
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        out = gen(jm, jt, q, max_new=8, temp=args.judge_temp, top_p=1.0).upper()
        if ("A" in out) ^ ("B" in out):
            return "A" if "A" in out else "B"
        return "T"

    winsA = winsB = ties = 0
    cases = []
    for (p, aA, aB) in pairs:
        res = judge(p, aA, aB)
        if res == "A":
            winsA += 1
        elif res == "B":
            winsB += 1
        else:
            ties += 1
        if args.save_cases:
            cases.append({"prompt": p, "A": aA, "B": aB, "judge": res})

    if args.symmetric:
        winsA2 = winsB2 = ties2 = 0
        for (p, aA, aB) in pairs:
            res2 = judge(p, aB, aA)
            if res2 == "A":
                winsB2 += 1
            elif res2 == "B":
                winsA2 += 1
            else:
                ties2 += 1
        winsA = (winsA + winsA2) // 1 
        winsB = (winsB + winsB2) // 1

    n_eff = max(1, winsA + winsB)
    pA = winsA / n_eff
    ci_low, ci_high = ci_normal(pA, n_eff)

    os.makedirs("reports", exist_ok=True)
    out = {
        "model_a": labelA,
        "model_b": labelB,
        "n": len(pairs),
        "wins_a": winsA,
        "wins_b": winsB,
        "ties": ties,
        "win_rate_a": pA,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "judge": args.judge,
        "temp": args.temp,
        "prompts_source": ("long100" if args.use_long100 else (args.prompts_file or "builtin")),
        "symmetric": args.symmetric,
    }
    with open("reports/judge_winrate.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    if args.save_cases:
        with open(args.save_cases, "w", encoding="utf-8") as f:
            for c in cases:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
