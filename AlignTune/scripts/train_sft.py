import random
import numpy as np
import torch

torch.backends.cuda.matmul.allow_tf32 = True
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

import unsloth
from unsloth import FastLanguageModel
import os
import yaml
import inspect
import time
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTTrainer
try:
    from trl import SFTConfig

    _HAS_SFTCONFIG = True
except Exception:
    _HAS_SFTCONFIG = False

from peft import LoraConfig, get_peft_model
from transformers import TrainerCallback

USE_GC = bool(int(os.environ.get("ALIGNTUNE_USE_GC", "0")))

def load_cfg(p):
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

class SafeSaveCallback(TrainerCallback):
    def __init__(self, model, tok, out_dir, every_steps=100, keep_last=2):
        from pathlib import Path
        import shutil

        self.model = model
        self.tok = tok
        self.snap_dir = Path(out_dir) / "snapshots"
        self.snap_dir.mkdir(parents=True, exist_ok=True)
        self.every = max(1, int(every_steps))
        self.keep = max(1, int(keep_last))
        self._shutil = shutil

    def _rotate(self):
        snaps = sorted(self.snap_dir.glob("step-*"), key=lambda p: int(p.name.split("-")[-1]))
        while len(snaps) > self.keep:
            old = snaps.pop(0)
            self._shutil.rmtree(old, ignore_errors=True)

    def on_step_end(self, args, state, control, **kwargs):
        step = state.global_step
        if step > 0 and step % self.every == 0:
            dest = self.snap_dir / f"step-{step}"
            tmp = self.snap_dir / f".tmp-step-{step}"
            tmp.mkdir(parents=True, exist_ok=True)
            try:
                self.model.save_pretrained(str(tmp))
                self.tok.save_pretrained(str(tmp))
                if dest.exists():
                    self._shutil.rmtree(dest, ignore_errors=True)
                tmp.rename(dest)
                self._rotate()
                print(f"[autosave] lightweight snapshot @ step {step} -> {dest}")
            except Exception as e:
                print(f"[warn] autosave failed at step {step}: {e}")
                self._shutil.rmtree(tmp, ignore_errors=True)
        return control

class ETACallback(TrainerCallback):
    def __init__(self, total_steps: int, log_every: int = 10):
        self.total = max(1, int(total_steps))
        self.log_every = max(1, int(log_every))
        self._start = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        step = state.global_step
        if step <= 0 or step % self.log_every != 0:
            return control
        dt = time.time() - self._start
        sps = step / dt if dt > 0 else 0.0
        rem = (self.total - step) / sps if sps > 0 else float("inf")
        print(f"[ETA] step {step}/{self.total} | {sps:.2f} steps/s | ETA ~ {rem/3600:.2f} h")
        return control

def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--preset", choices=["fast", "balanced", "thorough"], help="override preset from config")
    ap.add_argument("--autosave_steps", type=int, default=100)
    ap.add_argument("--autosave_keep", type=int, default=2)
    ap.add_argument("--no_autosave", action="store_true")
    ap.add_argument("--assistant_only_loss", action="store_true", help="Mask loss to assistant spans only (response-only training)")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    preset = args.preset or cfg.get("preset", "balanced")

    def pick(x):
        return x.get(preset, next(iter(x.values()))) if isinstance(x, dict) else x

    unsloth_len = int(pick(cfg["unsloth"]["max_seq_len"]))
    sft_maxseq = int(pick(cfg["sft"]["max_seq_len"]))
    sft_steps = int(pick(cfg["sft"]["max_steps"]))
    print(f"[SFT] preset={preset} | unsloth_len={unsloth_len} | sft_maxseq={sft_maxseq} | sft_steps={sft_steps}")

    model_name = cfg.get("model_name")
    tok_name = cfg.get("tokenizer_name") or model_name
    out_dir = os.path.join(
        cfg.get("output_dir", "outputs"),
        "sft",
        os.path.basename(model_name).replace("/", "-") + "-lora",
    )

    use_unsloth = bool(cfg.get("unsloth", {}).get("enable", False)) or (
        bool(cfg["lora"].get("qlora", False)) and bool(cfg["lora"].get("load_in_4bit", False))
    )

    if use_unsloth:
        model, tok = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=unsloth_len,
            load_in_4bit=bool(cfg["lora"].get("load_in_4bit", True)),
        )
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        try:
            model.config.attn_implementation = "sdpa"
        except Exception:
            pass
        l = cfg["lora"]
        model = FastLanguageModel.get_peft_model(
            model,
            r=int(pick(l.get("r", 16))),
            lora_alpha=int(l.get("alpha", 32)),
            lora_dropout=float(l.get("dropout", 0.0)),
            target_modules=l["target_modules"],
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
        try:
            model.config.attn_implementation = "sdpa"
        except Exception:
            pass

        l = cfg["lora"]
        peft_cfg = LoraConfig(
            r=int(pick(l.get("r", 16))),
            lora_alpha=int(l.get("alpha", 32)),
            lora_dropout=float(l.get("dropout", 0.0)),
            target_modules=l["target_modules"],
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_cfg)

    ds = load_dataset(
        "json",
        data_files={
            "train": cfg["sft"]["train_file"],
            "validation": cfg["sft"]["val_file"],
        },
    )

    if "text" in ds["train"].column_names and "text" in ds["validation"].column_names:
        pass
    else:
        system_txt = cfg.get("system_prompt", "You are a helpful, concise assistant.")

        def to_text(ex):
            instr = (ex.get("instruction") or "").strip()
            inp = (ex.get("input") or "").strip()
            user = instr if inp == "" else (f"{instr}\n{inp}" if instr else inp)
            messages = [
                {"role": "system", "content": system_txt},
                {"role": "user", "content": user},
                {"role": "assistant", "content": (ex.get("output") or "").strip()},
            ]
            text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            return {"text": text}

        ds = ds.map(to_text, remove_columns=ds["train"].column_names)

    sft_lr = float(cfg["sft"].get("lr", 2e-4))
    sft_warmup = float(cfg["sft"].get("warmup_ratio", 0.03))
    sft_bs_tr = int(cfg["sft"].get("per_device_train_batch_size", 4))
    sft_bs_ev = int(cfg["sft"].get("per_device_eval_batch_size", 4))
    sft_gas = int(cfg["sft"].get("gradient_accumulation_steps", 4))
    sft_eval = int(cfg["sft"].get("eval_steps", 100))
    sft_save = int(cfg["sft"].get("save_steps", 100))
    sft_log = int(cfg["sft"].get("logging_steps", 10))

    _tr_sig = inspect.signature(SFTTrainer.__init__)
    trainer_kwargs = {
        "model": model,
        "train_dataset": ds["train"],
        "eval_dataset": ds["validation"],
    }
    if "processing_class" in _tr_sig.parameters:
        trainer_kwargs["processing_class"] = tok
    elif "tokenizer" in _tr_sig.parameters:
        trainer_kwargs["tokenizer"] = tok
    if "dataset_text_field" in _tr_sig.parameters:
        trainer_kwargs["dataset_text_field"] = "text"
    if "max_seq_length" in _tr_sig.parameters:
        trainer_kwargs["max_seq_length"] = sft_maxseq

    RESP_TMPL = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    if args.assistant_only_loss:
        if "response_template" in _tr_sig.parameters:
            trainer_kwargs["response_template"] = RESP_TMPL
        if "train_on_inputs" in _tr_sig.parameters:
            trainer_kwargs["train_on_inputs"] = False

    if _HAS_SFTCONFIG:
        import inspect as _insp

        _cfg_sig = _insp.signature(SFTConfig)
        cfg_dict = dict(
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
            seed=42,
            overwrite_output_dir=True,
            tf32=True,
            gradient_checkpointing=USE_GC,
        )
        if "max_seq_length" in _cfg_sig.parameters:
            cfg_dict["max_seq_length"] = sft_maxseq
        if args.assistant_only_loss:
            if "train_on_inputs" in _cfg_sig.parameters:
                cfg_dict["train_on_inputs"] = False
            if "response_template" in _cfg_sig.parameters:
                cfg_dict["response_template"] = RESP_TMPL
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
            seed=42,
            overwrite_output_dir=True,
            tf32=True,
            gradient_checkpointing=USE_GC,
        )
        trainer = SFTTrainer(args=ta, **trainer_kwargs)

    if USE_GC and hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable()
            print("[GC] Enabled gradient checkpointing")
        except Exception:
            pass

    if not args.no_autosave and args.autosave_steps > 0:
        try:
            trainer.add_callback(SafeSaveCallback(model, tok, out_dir, args.autosave_steps, args.autosave_keep))
        except Exception:
            pass

    try:
        trainer.add_callback(ETACallback(total_steps=sft_steps, log_every=sft_log))
    except Exception:
        pass

    trainer.train()
    trainer.save_model(out_dir)
    tok.save_pretrained(out_dir)
    print(f"Saved SFT LoRA to {out_dir}")

if __name__ == "__main__":
    main()
