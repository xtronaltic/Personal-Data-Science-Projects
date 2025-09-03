import os
import json
import argparse
import heapq
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

def load_base_and_tok(base_name: str, load_in_4bit: bool = True):
    tok = AutoTokenizer.from_pretrained(base_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bnb = (
        BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
        if load_in_4bit
        else None
    )
    mdl = AutoModelForCausalLM.from_pretrained(base_name, device_map="auto", quantization_config=bnb)
    try:
        mdl.config.attn_implementation = "sdpa"
    except Exception:
        pass
    mdl.eval()
    return mdl, tok

def sum_logp_from_loss(loss: torch.Tensor, n_tokens: int) -> float:
    return float((-loss).item() * max(1, int(n_tokens)))

def make_labels(prompt_ids: torch.Tensor, answer_ids: torch.Tensor, pad_id: int):
    input_ids = torch.cat([prompt_ids, answer_ids], dim=-1)
    labels = input_ids.clone()
    labels[..., : prompt_ids.shape[-1]] = -100
    return input_ids, labels

@torch.no_grad()
def score_pair(model, tok, prompt: str, answer: str, max_len: int):
    p_ids = tok(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"]
    a_ids = tok(answer, add_special_tokens=False, return_tensors="pt")["input_ids"]

    total = p_ids.shape[-1] + a_ids.shape[-1]
    if total > max_len:
        keep_p = max(1, max_len - a_ids.shape[-1])
        p_ids = p_ids[:, -keep_p:]
        total = p_ids.shape[-1] + a_ids.shape[-1]
        if total > max_len:
            a_ids = a_ids[:, : max_len - p_ids.shape[-1]]

    input_ids, labels = make_labels(p_ids, a_ids, tok.pad_token_id)
    input_ids = input_ids.to(model.device)
    labels = labels.to(model.device)

    out = model(input_ids=input_ids, labels=labels)
    n_toks = (labels != -100).sum().item()
    return sum_logp_from_loss(out.loss, n_toks)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base",
        required=False,
        default=None,
        help="base model name; default from config if not given",
    )
    ap.add_argument("--ckpt", required=True, help="LoRA checkpoint to score with")
    ap.add_argument("--in_file", default="data/pack/dpo.jsonl")
    ap.add_argument("--out_file", default="data/pack/dpo.hard.jsonl")
    ap.add_argument("--top_k", type=int, default=15000)
    ap.add_argument("--max_items", type=int, default=60000)
    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--fourbit", action="store_true", help="load base in 4-bit (default true)")
    args = ap.parse_args()

    base_name = args.base or os.environ.get("BASE_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
    base_m, tok = load_base_and_tok(base_name, load_in_4bit=True)
    model = PeftModel.from_pretrained(base_m, args.ckpt)
    model.eval()

    heap = []
    total = 0

    in_path = Path(args.in_file)
    with in_path.open("r", encoding="utf-8") as rf:
        for idx, line in enumerate(rf):
            if args.max_items and idx >= args.max_items:
                break
            try:
                ex = json.loads(line)
            except Exception:
                continue

            prompt = ex.get("prompt") or ""
            chosen = ex.get("chosen") or ""
            rejected = ex.get("rejected") or ""
            if not (prompt and chosen and rejected):
                continue

            c = score_pair(model, tok, prompt, chosen, args.max_length)
            r = score_pair(model, tok, prompt, rejected, args.max_length)
            margin = c - r

            if len(heap) < args.top_k:
                heapq.heappush(heap, (-margin, idx, ex))
            else:
                if -heap[0][0] > margin:
                    heapq.heapreplace(heap, (-margin, idx, ex))

            total += 1
            if total % 500 == 0:
                print(f"[mine] scored {total} pairs")

    hardest = sorted([(-m, i, ex) for (m, i, ex) in heap])
    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as wf:
        for _, _, ex in hardest:
            wf.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"[mine] wrote top-{len(hardest)} hard pairs -> {out_path}")

if __name__ == "__main__":
    main()
