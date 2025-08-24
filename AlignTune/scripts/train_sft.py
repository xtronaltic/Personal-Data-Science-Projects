# scripts/train_sft.py
# --- Keep Unsloth imports FIRST so it can patch transformers/trl/peft ---
import unsloth
from unsloth import FastLanguageModel

import os, yaml, torch, inspect
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTTrainer
try:
    from trl import SFTConfig
    _HAS_SFTCONFIG = True
except Exception:
    _HAS_SFTCONFIG = False
from peft import LoraConfig, get_peft_model

def load_cfg(p):
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    args = ap.parse_args()
    cfg = load_cfg(args.config)

    model_name = cfg.get("model_name")
    tok_name   = cfg.get("tokenizer_name") or model_name
    out_dir    = os.path.join(
        cfg.get("output_dir", "outputs"),
        "sft",
        os.path.basename(model_name).replace("/", "-") + "-lora",
    )

    # Use Unsloth path if enabled OR if you're doing QLoRA (4-bit)
    use_unsloth = bool(cfg.get("unsloth", {}).get("enable", False)) \
                  or (bool(cfg["lora"].get("qlora", False)) and bool(cfg["lora"].get("load_in_4bit", False)))

    # ---------- Load model + tokenizer ----------
    if use_unsloth:
        model, tok = FastLanguageModel.from_pretrained(
            model_name = model_name,                                   # you can also try: "unsloth/tinyllama-chat-bnb-4bit"
            max_seq_length = int(cfg["sft"].get("max_seq_len", 1024)),
            load_in_4bit = bool(cfg["lora"].get("load_in_4bit", True)),
        )
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        # Attach LoRA with Unsloth helper (ensures trainable params > 0)
        l = cfg["lora"]
        model = FastLanguageModel.get_peft_model(
            model,
            r = int(l["r"]),
            lora_alpha = int(l["alpha"]),
            lora_dropout = float(l["dropout"]),
            target_modules = l["target_modules"],
        )
    else:
        tok = AutoTokenizer.from_pretrained(tok_name, use_fast=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        bnb = None
        if cfg["lora"].get("qlora", False) and cfg["lora"].get("load_in_4bit", False):
            bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            quantization_config=bnb,
        )
        # Attach LoRA with PEFT in the vanilla path
        l = cfg["lora"]
        peft_cfg = LoraConfig(
            r = int(l["r"]),
            lora_alpha = int(l["alpha"]),
            lora_dropout = float(l["dropout"]),
            target_modules = l["target_modules"],
            task_type = "CAUSAL_LM",
        )
        model = get_peft_model(model, peft_cfg)

    # ---------- Dataset ----------
    ds = load_dataset(
        "json",
        data_files={
            "train": cfg["sft"]["train_file"],
            "validation": cfg["sft"]["val_file"],
        },
    )
    from scripts.utils import format_sft
    ds = ds.map(
        lambda ex: {"text": format_sft(ex) + ex["output"]},
        remove_columns=ds["train"].column_names,
    )

    # ---------- Numeric coercion (Unsloth/TRL can be strict) ----------
    sft_lr     = float(cfg["sft"].get("lr", 2e-4))
    sft_warmup = float(cfg["sft"].get("warmup_ratio", 0.03))
    sft_bs_tr  = int(cfg["sft"].get("per_device_train_batch_size", 4))
    sft_bs_ev  = int(cfg["sft"].get("per_device_eval_batch_size", 4))
    sft_gas    = int(cfg["sft"].get("gradient_accumulation_steps", 4))
    sft_maxseq = int(cfg["sft"].get("max_seq_len", 1024))
    sft_eval   = int(cfg["sft"].get("eval_steps", 100))
    sft_save   = int(cfg["sft"].get("save_steps", 100))
    sft_log    = int(cfg["sft"].get("logging_steps", 10))
    sft_steps  = int(cfg["sft"].get("max_steps", 300))

    # ---------- Build trainer in a TRL-version-proof way ----------
    _tr_sig = inspect.signature(SFTTrainer.__init__)
    trainer_kwargs = {
        "model": model,
        "train_dataset": ds["train"],
        "eval_dataset": ds["validation"],
    }
    # tokenizer vs processing_class
    if "processing_class" in _tr_sig.parameters:
        trainer_kwargs["processing_class"] = tok
    elif "tokenizer" in _tr_sig.parameters:
        trainer_kwargs["tokenizer"] = tok
    # dataset_text_field (older TRL)
    if "dataset_text_field" in _tr_sig.parameters:
        trainer_kwargs["dataset_text_field"] = "text"
    # max_seq_length sometimes lives on the Trainer
    if "max_seq_length" in _tr_sig.parameters:
        trainer_kwargs["max_seq_length"] = sft_maxseq

    if _HAS_SFTCONFIG:
        _cfg_sig = inspect.signature(SFTConfig)
        cfg_dict = {
            "output_dir": out_dir,
            "per_device_train_batch_size": sft_bs_tr,
            "per_device_eval_batch_size": sft_bs_ev,
            "gradient_accumulation_steps": sft_gas,
            "learning_rate": sft_lr,
            "warmup_ratio": sft_warmup,
            "logging_steps": sft_log,
            "evaluation_strategy": "steps",
            "eval_steps": sft_eval,
            "save_steps": sft_save,
            "bf16": torch.cuda.is_available(),
            "num_train_epochs": 1,
            "max_steps": sft_steps,
            "lr_scheduler_type": "cosine",
            "report_to": [],
            # some TRL versions accept max_seq_length here too:
        }
        if "max_seq_length" in _cfg_sig.parameters:
            cfg_dict["max_seq_length"] = sft_maxseq

        sft_args = SFTConfig(**{k: v for k, v in cfg_dict.items() if k in _cfg_sig.parameters})
        trainer = SFTTrainer(args=sft_args, **trainer_kwargs)
    else:
        from transformers import TrainingArguments
        ta = TrainingArguments(
            output_dir=out_dir,
            per_device_train_batch_size=sft_bs_tr,
            per_device_eval_batch_size=sft_bs_ev,
            gradient_accumulation_steps=sft_gas,
            learning_rate=sft_lr,
            warmup_ratio=sft_warmup,
            logging_steps=sft_log,
            evaluation_strategy="steps",
            eval_steps=sft_eval,
            save_steps=sft_save,
            bf16=torch.cuda.is_available(),
            num_train_epochs=1,
            max_steps=sft_steps,
            lr_scheduler_type="cosine",
            report_to=[],
        )
        trainer = SFTTrainer(args=ta, **trainer_kwargs)

    # ---------- Train ----------
    trainer.train()
    trainer.save_model(out_dir)
    tok.save_pretrained(out_dir)
    print(f"Saved SFT LoRA to {out_dir}")

if __name__ == "__main__":
    main()
