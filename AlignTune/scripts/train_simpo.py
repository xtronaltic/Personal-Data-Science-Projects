import os, yaml, torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, LoraConfig, get_peft_model
from scripts.utils import format_dpo_prompt

def load_cfg(p): 
    with open(p,"r",encoding="utf-8") as f: return yaml.safe_load(f)

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--sft_ckpt", required=True)
    args = ap.parse_args()
    cfg = load_cfg(args.config)

    try:
        from trl import SimPOTrainer
    except Exception as e:
        print("[SimPO] SimPOTrainer not available in your TRL version.")
        print("Install a newer TRL or use DPO/ORPO instead.")
        return

    model_name = cfg.get("model_name")
    tok_name = cfg.get("tokenizer_name") or model_name
    out_dir = os.path.join(cfg.get("output_dir","outputs"),"simpo", os.path.basename(args.sft_ckpt)+"-simpo")

    tok = AutoTokenizer.from_pretrained(tok_name, use_fast=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token

    bnb = BitsAndBytesConfig(load_in_4bit=True) if (cfg["lora"].get("qlora", True) and cfg["lora"].get("load_in_4bit", True)) else None
    base = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", quantization_config=bnb)

    l = cfg["lora"]
    peft_cfg = LoraConfig(r=l["r"], lora_alpha=l["alpha"], lora_dropout=l["dropout"],
                          target_modules=l["target_modules"], task_type="CAUSAL_LM")
    base = get_peft_model(base, peft_cfg)
    model = PeftModel.from_pretrained(base, args.sft_ckpt)

    ds = load_dataset("json", data_files={"train": cfg["pref"]["train_file"], "validation": cfg["pref"]["val_file"]})
    ds = ds.map(lambda ex: {"prompt": format_dpo_prompt(ex["prompt"]), "chosen": ex["chosen"], "rejected": ex["rejected"]})

    trainer = SimPOTrainer(
        model=model, ref_model=None,
        args=dict(output_dir=out_dir,
                  per_device_train_batch_size=cfg["simpo"]["per_device_train_batch_size"],
                  per_device_eval_batch_size=cfg["simpo"]["per_device_eval_batch_size"],
                  gradient_accumulation_steps=cfg["simpo"]["gradient_accumulation_steps"],
                  learning_rate=cfg["simpo"]["lr"], logging_steps=cfg["simpo"]["logging_steps"],
                  evaluation_strategy="steps", eval_steps=cfg["simpo"]["eval_steps"],
                  save_steps=cfg["simpo"]["save_steps"], bf16=torch.cuda.is_available(),
                  max_steps=cfg["simpo"]["max_steps"], report_to=[]),
        beta=cfg["simpo"]["gamma"],
        train_dataset=ds["train"], eval_dataset=ds["validation"],
        tokenizer=tok, max_length=1024, max_prompt_length=768
    )
    trainer.train()
    trainer.save_model(out_dir); tok.save_pretrained(out_dir)
    print(f"Saved SimPO LoRA to {out_dir}")

if __name__ == "__main__":
    main()
