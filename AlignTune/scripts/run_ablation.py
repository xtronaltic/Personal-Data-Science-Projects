import os
import json
import argparse
import yaml
import numpy as np
import torch
from datasets import load_dataset
from rouge_score import rouge_scorer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import time
import math
from peft import PeftModel
from scripts.utils import format_sft, safety_flag

def load_cfg(p):
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def eval_adapter(base, ckpt, val_file, gen_cfg, batch_size: int = 8):
    tok = AutoTokenizer.from_pretrained(base, use_fast=True)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True)
    base_m = AutoModelForCausalLM.from_pretrained(base, device_map="auto", quantization_config=bnb)
    try:
        base_m.config.attn_implementation = "sdpa"
    except Exception:
        pass
    model = PeftModel.from_pretrained(base_m, ckpt)

    ds = load_dataset("json", data_files={"val": val_file})["val"]

    ems, rouges, tox = [], [], 0
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    def _batched(prompts, bsz=8):
        outs = []
        total_batches = math.ceil(len(prompts) / max(1, bsz))
        print_every = max(1, total_batches // 20)
        t0 = time.time()
        for i in range(0, len(prompts), bsz):
            batch_idx = i // bsz + 1
            chunk = prompts[i : i + bsz]
            enc = tok(chunk, return_tensors="pt", padding=True, truncation=False).to(model.device)
            with torch.inference_mode():
                gen = model.generate(
                    **enc,
                    max_new_tokens=gen_cfg.get("max_new_tokens", 256),
                    do_sample=True,
                    temperature=gen_cfg.get("temperature", 0.7),
                    top_p=gen_cfg.get("top_p", 0.9),
                    eos_token_id=tok.eos_token_id,
                    pad_token_id=tok.pad_token_id or tok.eos_token_id,
                )
            for j in range(len(chunk)):
                in_len = (enc["input_ids"][j] != (tok.pad_token_id or tok.eos_token_id)).sum().item()
                outs.append(tok.decode(gen[j][in_len:], skip_special_tokens=True))

            if (batch_idx % print_every == 0) or (batch_idx == total_batches):
                dt = time.time() - t0
                avg = dt / max(1, batch_idx)
                rem = avg * max(0, total_batches - batch_idx)
                try:
                    alloc = torch.cuda.memory_allocated() / 1e9
                    reserv = torch.cuda.memory_reserved() / 1e9
                    vram_txt = f"alloc={alloc:.1f}G reserved={reserv:.1f}G"
                except Exception:
                    vram_txt = "vram=n/a"
                print(
                    f"[ablation] batches {batch_idx}/{total_batches} | ETA ~ {rem/60:.1f} min | {vram_txt}",
                    flush=True,
                )
        return outs

    prompts = [format_sft(ex) for ex in ds]
    eff_bsz = int(batch_size) if batch_size and batch_size > 0 else 8
    preds = _batched(prompts, bsz=eff_bsz)
    for ex, pred in zip(ds, preds):
        pred = pred.strip()
        ems.append(1.0 if pred.strip().lower() == ex["output"].strip().lower() else 0.0)
        rouges.append(float(scorer.score(ex["output"], pred)["rougeL"].fmeasure))
        if safety_flag(pred):
            tox += 1

    return {
        "n": len(ds),
        "exact_match": float(np.mean(ems)),
        "rougeL": float(np.mean(rouges)),
        "toxic_flags": int(tox),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--ckpt", nargs="+", required=True, help="list of ckpt paths to compare")
    ap.add_argument("--labels", nargs="*", help="optional labels for ckpts")
    ap.add_argument("--eval_file", default=None, help="optional eval JSONL file (overrides cfg.sft.val_file)")
    ap.add_argument("--out", default="reports/ablation_summary.json", help="output summary JSON path")
    ap.add_argument("--batch_size", type=int, default=None, help="optional override for generation batch size")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    base = cfg.get("model_name")
    val_file = args.eval_file or cfg["sft"]["val_file"]
    gen_cfg = cfg.get("generation", {})

    rows = []
    for i, ck in enumerate(args.ckpt):
        label = args.labels[i] if args.labels and i < len(args.labels) else os.path.basename(ck)
        print(f"[ablation] evaluating {label} -> {ck}")
        metrics = eval_adapter(base, ck, val_file, gen_cfg, batch_size=(args.batch_size or 8))
        rows.append({"label": label, **metrics})

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"runs": rows}, f, indent=2)

    print("label\tn\texact_match\trougeL\ttoxic_flags")
    for r in rows:
        print(f"{r['label']}\t{r['n']}\t{r['exact_match']:.3f}\t{r['rougeL']:.3f}\t{r['toxic_flags']}")

if __name__ == "__main__":
    main()
