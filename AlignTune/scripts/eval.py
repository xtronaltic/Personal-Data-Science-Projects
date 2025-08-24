import os, json, argparse, yaml, numpy as np
from datasets import load_dataset
from rouge_score import rouge_scorer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from peft import PeftModel
from scripts.utils import format_sft, safety_flag

def load_cfg(p): 
    with open(p,"r",encoding="utf-8") as f: return yaml.safe_load(f)

def exact_match(a,b): return 1.0 if a.strip().lower()==b.strip().lower() else 0.0
def rouge_l(a,b):
    s = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return float(s.score(b,a)["rougeL"].fmeasure)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--ckpt", required=True)
    args = ap.parse_args()
    cfg = load_cfg(args.config)

    base = cfg.get("model_name")
    tok = AutoTokenizer.from_pretrained(base, use_fast=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True) if cfg["lora"].get("qlora", True) else None
    base_m = AutoModelForCausalLM.from_pretrained(base, device_map="auto", quantization_config=bnb)
    model = PeftModel.from_pretrained(base_m, args.ckpt)

    ds = load_dataset("json", data_files={"val": cfg["sft"]["val_file"]})["val"]
    pipe = pipeline("text-generation", model=model, tokenizer=tok, device_map="auto")

    ems, rouges, tox = [], [], 0
    for ex in ds:
        prompt = format_sft(ex)
        out = pipe(prompt, max_new_tokens=cfg["generation"]["max_new_tokens"], do_sample=True,
                   temperature=cfg["generation"]["temperature"], top_p=cfg["generation"]["top_p"],
                   eos_token_id=tok.eos_token_id)[0]["generated_text"]
        pred = out.split("[RESPONSE]\n")[-1].strip()
        ems.append(exact_match(pred, ex["output"]))
        rouges.append(rouge_l(pred, ex["output"]))
        if safety_flag(pred): tox += 1

    report = {"n": len(ds), "exact_match": float(np.mean(ems)), "rougeL": float(np.mean(rouges)), "toxic_flags": int(tox)}
    os.makedirs("reports", exist_ok=True)
    with open("reports/eval_report.json","w",encoding="utf-8") as f: json.dump(report,f,indent=2)
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
