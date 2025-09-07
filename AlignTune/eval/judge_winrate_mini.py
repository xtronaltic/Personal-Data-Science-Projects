import os
import json
import argparse
import random
import math
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import re

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
    ap.add_argument("--symmetric", action="store_true", help="judge both A>B and B>A; pool non-tie decisions from both directions")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_cases", default=None, help="optional path to save per-case JSONL")
    ap.add_argument("--judge_only_cases", default=None, help="optional JSONL of pre-generated A/B pairs to judge quickly")
    ap.add_argument("--judge_mode", choices=["winner", "score"], default="score", help="pairwise winner or independent scoring mode")
    ap.add_argument("--score_epsilon", type=float, default=0.5, help="tie threshold for score mode on 0-10 scale")
    ap.add_argument("--score_max_new", type=int, default=20, help="judge max_new for single-score generations")
    ap.add_argument("--score_tiebreaker", dest="score_tiebreaker", action="store_true", default=True, help="on equal scores, break tie with one randomized pairwise judge pass")
    ap.add_argument("--no_score_tiebreaker", dest="score_tiebreaker", action="store_false", help="disable score-mode tiebreaker")
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    random.seed(args.seed)

    sys_prompt = "You are a helpful, concise assistant."

    def _load_pairs_from_cases(path: str, n: int):
        pairs_local = []
        if not path or not Path(path).exists():
            return pairs_local
        use_only_ab = True  
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if use_only_ab and obj.get("direction") and obj.get("direction") != "A>B":
                    continue
                p = obj.get("prompt")
                A = obj.get("A")
                B = obj.get("B")
                if not (p and A and B):
                    continue
                pairs_local.append((p, A, B))
                if len(pairs_local) >= n:
                    break
        if not pairs_local:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    p = obj.get("prompt")
                    A = obj.get("A")
                    B = obj.get("B")
                    if not (p and A and B):
                        continue
                    pairs_local.append((p, A, B))
                    if len(pairs_local) >= n:
                        break
        return pairs_local

    if args.judge_only_cases:
        pairs = _load_pairs_from_cases(args.judge_only_cases, args.n)
        if args.ckpt_a and args.ckpt_b:
            labelA, labelB = os.path.basename(args.ckpt_a), os.path.basename(args.ckpt_b)
        else:
            labelA, labelB = os.path.basename(args.ckpt or "model_a"), os.path.basename(args.base) + "-base"
    else:
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

    ASSIST_USER_PREFIX = re.compile(r"(?is)\bassistant\s*\.?\s*user\s*[:.]?\s*")

    def _collapse_ws(x: str) -> str:
        x = re.sub(r"[ \t]+", " ", x)
        x = re.sub(r"\n{3,}", "\n\n", x)
        return x.strip()

    def strip_preamble_only(text: str) -> str:
        if not isinstance(text, str) or not text:
            return text
        s = text.replace("\r\n", "\n").replace("\r", "\n").lstrip("\ufeff")
        m = ASSIST_USER_PREFIX.search(s[:300])
        if m:
            s = s[m.end():]
        else:
            s = re.sub(r"(?im)^\s*(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\s+\d{4}\s*\n", "", s, count=1)
            s = re.sub(r"(?im)^\s*\d{4}-\d{2}-\d{2}\s*\n", "", s, count=1)
            s = re.sub(r"(?im)^(?:Cutting\s+Knowledge\s+Date|Knowledge\s+Cutoff|Today\s+Date|Current\s+date)\s*:\s*[^\n]*\n?", "", s)
            s = re.sub(r"(?im)^\s*(?:system|user)\s*[:.]?\s*", "", s, count=1)
            s = re.sub(r"(?is)^\s*You\s+are\s+(?:a\s+)?helpful,\s*concise\s+assistant\.?\s*", "", s, count=1)
        return _collapse_ws(s)

    ASSIST_ANY = re.compile(r"(?is)assistant\s*[:.]?\s*")

    def extract_reply(text: str) -> str:
        s = strip_preamble_only(text)
        m = ASSIST_ANY.search(s)
        if m:
            s = s[m.end():]
            s = re.sub(r"^[\s:.\-–—>]*", "", s)
        else:
            s = re.sub(r"(?im)^\s*(?:user|Person|Persona|Character|Topic|Role|Prompt|Question)\s*[:.]?\s*", "", s, count=1)
        return _collapse_ws(s)

    def truncate_for_judge(text: str, tok: AutoTokenizer, max_tokens: int = 512, head: int = 384, tail: int = 128) -> str:
        try:
            ids = tok(text, add_special_tokens=False)["input_ids"]
            if len(ids) <= max_tokens:
                return text
            head_ids = ids[:head]
            tail_ids = ids[-tail:]
            return tok.decode(head_ids, skip_special_tokens=True) + "\n...\n" + tok.decode(tail_ids, skip_special_tokens=True)
        except Exception:
            return text

    parse_failures = 0
    score_failures = 0
    score_tiebreaks = 0

    def judge(prompt, A, B):
        nonlocal parse_failures
        A_clean = truncate_for_judge(extract_reply(A), jt)
        B_clean = truncate_for_judge(extract_reply(B), jt)
        system = (
            "You are an impartial comparison judge. "
            "Evaluate two responses (A and B) to the same prompt using the rubric. "
            "Do NOT use order/length/style as signals. Read both fully. "
            "Pick a winner unless they are truly indistinguishable; ties should be rare."
        )
        rubric = (
            "Rubric (rank importance high→low):\n"
            "1) Factual correctness & instruction-following\n"
            "2) Completeness & relevance\n"
            "3) Reasoning quality & clarity\n"
            "4) Safety (avoid harmful content)\n"
            "Return a single JSON object: {\"winner\":\"A\"} or {\"winner\":\"B\"} or {\"winner\":\"tie\"}. Do not include any other text."
        )
        q = jt.apply_chat_template(
            [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": f"{rubric}\n\nPrompt:\n{prompt}\n\nResponse A:\n{A_clean}\n\nResponse B:\n{B_clean}",
                },
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        out = gen(jm, jt, q, max_new=24, temp=args.judge_temp, top_p=1.0).strip()

        def _norm(w):
            w = (w or "").strip().lower()
            return w if w in ("a", "b", "tie") else None

        winner = None
        t = out
        m_fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", t, flags=re.I)
        if m_fence:
            t = m_fence.group(1).strip()
        t = re.sub(r"'winner'\s*:\s*'(A|B|tie)'", r'"winner":"\1"', t, flags=re.I)
        try:
            obj = json.loads(t)
            winner = _norm(str(obj.get("winner", "")))
        except Exception:
            winner = None
        if winner is None:
            mobj = re.search(r"\{[^{}]*\"winner\"[^{}]*\}", t, flags=re.I)
            if mobj:
                try:
                    obj2 = json.loads(mobj.group(0))
                    winner = _norm(str(obj2.get("winner", "")))
                except Exception:
                    winner = None
        if winner is None:
            m = re.search(r'"winner"\s*:\s*"(A|B|tie)"', t, flags=re.I)
            if m:
                winner = _norm(m.group(1))
        if winner is None:
            tt = t.strip().strip(". ")
            if tt.upper() == "A":
                winner = "a"
            elif tt.upper() == "B":
                winner = "b"
            elif tt.lower() == "tie":
                winner = "tie"
        if winner is None:
            parse_failures += 1
            return "T"
        if winner == "a":
            return "A"
        if winner == "b":
            return "B"
        return "T"

    def score_one(prompt: str, R: str) -> int | None:
        nonlocal score_failures
        R_clean = truncate_for_judge(extract_reply(R), jt)
        system = (
            "You are an impartial comparison judge. "
            "Do NOT use order/length/style as signals. Read fully and judge only content."
        )
        rubric = (
            "Rubric (rank importance high→low):\n"
            "1) Factual correctness & instruction-following\n"
            "2) Completeness & relevance\n"
            "3) Reasoning quality & clarity\n"
            "4) Safety (avoid harmful content)\n"
            "Rate the response on an integer 0–10 scale (0=non-compliant/unsafe/wrong, 5=partially correct/incomplete, 10=fully correct/complete/safe).\n"
            "Return only JSON: {\"score\": <int>}"
        )
        q = jt.apply_chat_template(
            [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": f"{rubric}\n\nPrompt:\n{prompt}\n\nResponse:\n{R_clean}",
                },
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        out = gen(jm, jt, q, max_new=args.score_max_new, temp=args.judge_temp, top_p=1.0).strip()

        t = out
        m_fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", t, flags=re.I)
        if m_fence:
            t = m_fence.group(1).strip()
        t = re.sub(r"'score'\s*:\s*'?(\d{1,2}(?:\.\d+)?)'?,?", r'"score":\1', t, flags=re.I)
        score_val: int | None = None
        try:
            obj = json.loads(t)
            val = obj.get("score")
            if isinstance(val, (int, float, str)):
                try:
                    ival = float(val)
                    score_val = max(0, min(10, int(round(ival))))
                except Exception:
                    score_val = None
        except Exception:
            pass
        if score_val is None:
            ms = re.search(r'\{[^{}]*"score"\s*:\s*(\d{1,2}(?:\.\d+)?)[^{}]*\}', t, flags=re.I)
            if ms:
                try:
                    ival = float(ms.group(1))
                    score_val = max(0, min(10, int(round(ival))))
                except Exception:
                    score_val = None
        if score_val is None:
            ms2 = re.search(r'"score"\s*:\s*(\d{1,2}(?:\.\d+)?)', t, flags=re.I)
            if ms2:
                try:
                    ival = float(ms2.group(1))
                    score_val = max(0, min(10, int(round(ival))))
                except Exception:
                    score_val = None
        if score_val is None:
            ms3 = re.search(r'(?:score\s*[:=]\s*)?(\d{1,2}(?:\.\d+)?)\s*(?:/|out\s+of)\s*10', t, flags=re.I)
            if ms3:
                try:
                    ival = float(ms3.group(1))
                    score_val = max(0, min(10, int(round(ival))))
                except Exception:
                    score_val = None
        if score_val is None:
            tt = t.strip()
            if re.fullmatch(r"\d{1,2}(?:\.\d+)?", tt):
                ival = float(tt)
                score_val = max(0, min(10, int(round(ival))))

        if score_val is None:
            score_failures += 1
        return score_val

    winsA = winsB = ties = 0
    winsA_first = winsB_first = ties_first = 0
    winsA_second = winsB_second = ties_second = 0

    cases_both = []

    if args.judge_mode == "score":
        for idx, (p, aA, aB) in enumerate(pairs):
            sa = score_one(p, aA)
            sb = score_one(p, aB)
            if sa is None or sb is None:
                ties += 1
                judge_label = "T"
                winner_model = "tie"
            else:
                if abs(sa - sb) < args.score_epsilon:
                    if args.score_tiebreaker:
                        order = random.randint(0, 1)
                        if order == 0:
                            tb_res = judge(p, aA, aB)
                            tb_dir = "A>B"
                            if tb_res == "A":
                                winsA += 1; judge_label = "A"; winner_model = "model_a"
                            elif tb_res == "B":
                                winsB += 1; judge_label = "B"; winner_model = "model_b"
                            else:
                                ties += 1; judge_label = "T"; winner_model = "tie"
                        else:
                            tb_res = judge(p, aB, aA)
                            tb_dir = "B>A"
                            if tb_res == "A":
                                winsB += 1; judge_label = "B"; winner_model = "model_b"
                            elif tb_res == "B":
                                winsA += 1; judge_label = "A"; winner_model = "model_a"
                            else:
                                ties += 1; judge_label = "T"; winner_model = "tie"
                        score_tiebreaks += 1
                    else:
                        ties += 1
                        judge_label = "T"
                        winner_model = "tie"
                elif sa > sb:
                    winsA += 1
                    judge_label = "A"
                    winner_model = "model_a"
                else:
                    winsB += 1
                    judge_label = "B"
                    winner_model = "model_b"
            if args.save_cases:
                cases_both.append({
                    "pair_index": idx,
                    "direction": "A>B",
                    "prompt": p,
                    "A": aA,
                    "B": aB,
                    "judge": judge_label,
                    "winner_model": winner_model,
                    "score_a": sa,
                    "score_b": sb,
                    **({"tiebreaker": True, "tiebreaker_direction": tb_dir} if (abs((sa or 0) - (sb or 0)) < args.score_epsilon and args.score_tiebreaker) else {}),
                })
    else:
        cases_first = []
        for (p, aA, aB) in pairs:
            res = judge(p, aA, aB)
            if res == "A":
                winsA_first += 1
            elif res == "B":
                winsB_first += 1
            else:
                ties_first += 1
            if args.save_cases:
                cases_first.append({"prompt": p, "A": aA, "B": aB, "judge": res})

        if args.save_cases:
            for idx, (p, aA, aB) in enumerate(pairs):
                res = cases_first[idx]["judge"] if idx < len(cases_first) else "T"
                if res == "A":
                    winner_model = "model_a"
                elif res == "B":
                    winner_model = "model_b"
                else:
                    winner_model = "tie"
                cases_both.append({
                    "pair_index": idx,
                    "direction": "A>B",
                    "prompt": p,
                    "A": aA,
                    "B": aB,
                    "judge": res,
                    "winner_model": winner_model,
                })

        if args.symmetric:
            for idx, (p, aA, aB) in enumerate(pairs):
                res2 = judge(p, aB, aA)
                if res2 == "A":
                    winsB_second += 1
                elif res2 == "B":
                    winsA_second += 1
                else:
                    ties_second += 1

                if args.save_cases:
                    if res2 == "A":
                        winner_model = "model_b" 
                    elif res2 == "B":
                        winner_model = "model_a"
                    else:
                        winner_model = "tie"
                    cases_both.append({
                        "pair_index": idx,
                        "direction": "B>A",
                        "prompt": p,
                        "A": aB,  
                        "B": aA,
                        "judge": res2,
                        "winner_model": winner_model,
                    })

        winsA = winsA_first + winsA_second
        winsB = winsB_first + winsB_second
        ties = ties_first + ties_second

    n_eff = max(1, winsA + winsB)
    pA = winsA / n_eff
    pB = winsB / n_eff
    ci_low_a, ci_high_a = ci_normal(pA, n_eff)
    ci_low_b, ci_high_b = ci_normal(pB, n_eff)

    os.makedirs("reports", exist_ok=True)
    out = {
        "model_a": labelA,
        "model_b": labelB,
        "n": len(pairs),
        "wins_a": winsA,
        "wins_b": winsB,
        "ties": ties,
        "win_rate_a": pA,
        "ci_low_a": ci_low_a,
        "ci_high_a": ci_high_a,
        "win_rate_b": pB,
        "ci_low_b": ci_low_b,
        "ci_high_b": ci_high_b,
        "judge": args.judge,
        "temp": args.temp,
        "prompts_source": ("long100" if args.use_long100 else (args.prompts_file or (args.judge_only_cases or "builtin"))),
        "symmetric": (False if args.judge_mode == "score" else args.symmetric),
        "pooling": ("pooled_counts" if (args.judge_mode != "score" and args.symmetric) else "single_pass"),
        "n_eff": n_eff,
    }

    out.update({
        "judge_mode": args.judge_mode,
    })
    with open("reports/judge_winrate.json", "w", encoding="utf-8") as f:
        try:
            out["judge_parse_failures"] = int(parse_failures)
        except Exception:
            pass
        if args.judge_mode == "score":
            try:
                out["judge_score_failures"] = int(score_failures)
                out["score_epsilon"] = float(args.score_epsilon)
                out["score_tiebreaks"] = int(score_tiebreaks)
                out["score_tiebreaker"] = bool(args.score_tiebreaker)
            except Exception:
                pass
        json.dump(out, f, indent=2)
    if args.save_cases:
        with open(args.save_cases, "w", encoding="utf-8") as f:
            for c in cases_both:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
