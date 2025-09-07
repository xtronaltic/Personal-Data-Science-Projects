import os
import json
import argparse
import yaml
import numpy as np
from datasets import load_dataset
from rouge_score import rouge_scorer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from peft import PeftModel
from scripts.utils import format_sft, safety_flag

def load_cfg(p):
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def exact_match(a, b):
    return 1.0 if a.strip().lower() == b.strip().lower() else 0.0

def rouge_l(a, b):
    s = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return float(s.score(b, a)["rougeL"].fmeasure)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--ckpt", required=True)
    args = ap.parse_args()

    cfg = load_cfg(args.config)

    base = cfg.get("model_name")
    tok = AutoTokenizer.from_pretrained(base, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    bnb = (
        BitsAndBytesConfig(load_in_4bit=True)
        if cfg["lora"].get("qlora", True) and cfg["lora"].get("load_in_4bit", True)
        else None
    )
    base_m = AutoModelForCausalLM.from_pretrained(base, device_map="auto", quantization_config=bnb)
    try:
        base_m.config.attn_implementation = "sdpa"
    except Exception:
        pass
    model = PeftModel.from_pretrained(base_m, args.ckpt)

    ds = load_dataset("json", data_files={"val": cfg["sft"]["val_file"]})["val"]
    pipe = pipeline("text-generation", model=model, tokenizer=tok, device_map="auto")

    ems, rouges, tox = [], [], 0
    total_generated_tokens = 0
    gen_cfg = cfg.get("generation", {})

    for ex in ds:
        prompt = format_sft(ex)
        out = pipe(
            prompt,
            max_new_tokens=gen_cfg.get("max_new_tokens", 256),
            do_sample=True,
            temperature=gen_cfg.get("temperature", 0.7),
            top_p=gen_cfg.get("top_p", 0.9),
            eos_token_id=tok.eos_token_id,
        )[0]["generated_text"]
        pred = out.split("[RESPONSE]\n")[-1].strip()
        ems.append(exact_match(pred, ex["output"]))
        rouges.append(rouge_l(pred, ex["output"]))
        if safety_flag(pred):
            tox += 1
        try:
            total_generated_tokens += len(tok(pred, add_special_tokens=False)["input_ids"]) 
        except Exception:
            pass

    report = {
        "n": len(ds),
        "exact_match": float(np.mean(ems)),
        "rougeL": float(np.mean(rouges)),
        "toxic_flags": int(tox),
        "total_generated_tokens": int(total_generated_tokens),
        "toxicity_per_1k_tokens": (float(tox) / max(1.0, float(total_generated_tokens) / 1000.0)),
    }

    os.makedirs("reports", exist_ok=True)
    with open("reports/eval_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
