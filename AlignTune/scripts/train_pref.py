import os
import argparse
import inspect
import random
import yaml
import time
import signal
import shutil
from pathlib import Path
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    TrainerCallback,
)
from transformers.trainer_utils import get_last_checkpoint
from peft import PeftModel, prepare_model_for_kbit_training

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
torch.backends.cuda.matmul.allow_tf32 = True
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

USE_COMPILE = bool(int(os.environ.get("ALIGNTUNE_COMPILE", "0")))
USE_GC = bool(int(os.environ.get("ALIGNTUNE_USE_GC", "0")))
OPTIM_8BIT = bool(int(os.environ.get("ALIGNTUNE_OPTIM_8BIT", "0")))

try:
    from trl import DPOTrainer
except Exception:
    try:
        from trl.trainer import DPOTrainer 
    except Exception:
        DPOTrainer = None
try:
    from trl import SimPOTrainer
except Exception:
    try:
        from trl.trainer import SimPOTrainer 
    except Exception:
        SimPOTrainer = None
try:
    from trl import DPOConfig
except Exception:
    try:
        from trl.trainer import DPOConfig  
    except Exception:
        DPOConfig = None
try:
    from trl import SimPOConfig
except Exception:
    try:
        from trl.trainer import SimPOConfig 
    except Exception:
        SimPOConfig = None
try:
    from trl import CPOTrainer
except Exception:
    try:
        from trl.trainer import CPOTrainer
    except Exception:
        CPOTrainer = None
try:
    from trl import CPOConfig
except Exception:
    try:
        from trl.trainer import CPOConfig
    except Exception:
        CPOConfig = None
try:
    from trl import ORPOTrainer
except Exception:
    try:
        from trl.trainer import ORPOTrainer 
    except Exception:
        ORPOTrainer = None
try:
    from trl import ORPOConfig
except Exception:
    try:
        from trl.trainer import ORPOConfig 
    except Exception:
        ORPOConfig = None

def load_cfg(p):
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _pick(val, preset: str):
    if isinstance(val, dict):
        if preset in val:
            return val[preset]
        for _, v in val.items():
            return v
    return val

def _gi(sec, key, default, preset, cast=int):
    return cast(_pick(sec.get(key, default), preset))

def _gf(sec, key, default, preset):
    return float(_pick(sec.get(key, default), preset))

def _gb(sec, key, default, preset):
    v = _pick(sec.get(key, default), preset)
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.lower() in ("1", "true", "yes", "y", "t")
    return bool(v)

def _norm_text(x):
    if x is None:
        return ""
    if isinstance(x, list):
        return "\n".join(str(t) for t in x if t is not None)
    return str(x)

def _ensure_pref_columns(ds, tok, system_txt):
    """Ensure dataset has {prompt, chosen, rejected}. If not, build from instruction/input."""
    cols = set(ds.column_names)
    if {"prompt", "chosen", "rejected"}.issubset(cols):

        def norm(ex):
            return {
                "prompt": _norm_text(ex.get("prompt")),
                "chosen": _norm_text(ex.get("chosen")),
                "rejected": _norm_text(ex.get("rejected")),
            }

        return ds.map(norm, remove_columns=[])

    def to_pref(ex):
        instr = _norm_text(ex.get("instruction") or ex.get("prompt"))
        inp = _norm_text(ex.get("input"))
        user = instr if not inp else (f"{instr}\n{inp}" if instr else inp)
        messages = [
            {"role": "system", "content": system_txt},
            {"role": "user", "content": user},
        ]
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return {
            "prompt": prompt,
            "chosen": _norm_text(ex.get("chosen")),
            "rejected": _norm_text(ex.get("rejected")),
        }

    drop_cols = [c for c in ds.column_names if c not in {"instruction", "input", "prompt", "chosen", "rejected"}]
    return ds.map(to_pref, remove_columns=drop_cols or None)

def _add_prompt_lengths_for_grouping(ds, tok):
    """Add a lightweight 'length' column for bucketing; avoid adding input_ids to
    prevent interfering with TRL trainers' internal tokenization/collation.
    """

    def _batched(batch):
        toks = tok(batch["prompt"], add_special_tokens=False, padding=False, truncation=False)
        lens = [len(x) for x in toks["input_ids"]]
        return {"length": lens}

    return ds.map(_batched, batched=True, remove_columns=None)

def load_base_and_tok(base):
    tok = AutoTokenizer.from_pretrained(base, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    mdl = AutoModelForCausalLM.from_pretrained(base, device_map="auto", quantization_config=bnb)

    try:
        mdl.config.attn_implementation = "sdpa"
    except:
        pass
    try:
        mdl.config.use_cache = False
    except:
        pass

    mdl = prepare_model_for_kbit_training(mdl)
    return mdl, tok

class SafeSaveCallback(TrainerCallback):
    """Lightweight LoRA-only snapshot every N steps, with rotation."""

    def __init__(self, model, tok, out_dir, every_steps=100, keep_last=2):
        self.model = model
        self.tok = tok
        self.snap_dir = Path(out_dir) / "snapshots"
        self.snap_dir.mkdir(parents=True, exist_ok=True)
        self.every = max(1, int(every_steps))
        self.keep = max(1, int(keep_last))

    def _rotate(self):
        snaps = sorted(self.snap_dir.glob("step-*"), key=lambda p: int(p.name.split("-")[-1]))
        while len(snaps) > self.keep:
            old = snaps.pop(0)
            shutil.rmtree(old, ignore_errors=True)

    def _light_save(self, step):
        dest = self.snap_dir / f"step-{step}"
        tmp = self.snap_dir / f".tmp-step-{step}"
        tmp.mkdir(parents=True, exist_ok=True)
        try:
            self.model.save_pretrained(str(tmp))
            self.tok.save_pretrained(str(tmp))
            if dest.exists():
                shutil.rmtree(dest, ignore_errors=True)
            tmp.rename(dest)
            self._rotate()
            print(f"[autosave] lightweight snapshot @ step {step} -> {dest}")
        except Exception as e:
            print(f"[warn] autosave failed at step {step}: {e}")
            shutil.rmtree(tmp, ignore_errors=True)

    def on_step_end(self, args, state, control, **kwargs):
        step = state.global_step
        if step > 0 and step % self.every == 0:
            self._light_save(step)
        return control

class ETACallback(TrainerCallback):
    def __init__(self, total_steps: int, log_every: int = 10):
        self.total = max(1, int(total_steps))
        self.log_every = max(1, int(log_every))
        self._start = time.time()
        self._last_step = 0

    def on_step_end(self, args, state, control, **kwargs):
        step = state.global_step
        if step <= 0 or step % self.log_every != 0:
            return control
        now = time.time()
        dt = now - self._start
        sps = step / dt if dt > 0 else 0.0
        rem = (self.total - step) / sps if sps > 0 else float("inf")
        print(f"[ETA] step {step}/{self.total} | {sps:.2f} steps/s | ETA ~ {rem/3600:.2f} h")
        return control

def emergency_save(model, tok, out_dir):
    """Best-effort save even if we hit an unexpected exception or signal."""
    try:
        path = Path(out_dir) / "checkpoint-last"
        path_tmp = Path(out_dir) / ".checkpoint-last-tmp"
        path_tmp.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(path_tmp))
        tok.save_pretrained(str(path_tmp))
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
        path_tmp.rename(path)
        print(f"[emergency-save] wrote {path}")
    except Exception as e:
        print(f"[warn] emergency save failed: {e}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--algo", choices=["dpo", "simpo", "orpo"], default="dpo")
    ap.add_argument("--preset", choices=["fast", "balanced", "thorough"], help="override preset from config")
    ap.add_argument("--sft_ckpt", required=True)
    ap.add_argument("--train_file", help="override preference train file (JSONL)")
    ap.add_argument("--resume", action="store_true", help="resume from latest checkpoint if present")
    ap.add_argument("--autosave_steps", type=int, default=100, help="light LoRA snapshot interval")
    ap.add_argument("--autosave_keep", type=int, default=2, help="how many light snapshots to retain")
    ap.add_argument("--no_autosave", action="store_true", help="disable lightweight autosave snapshots")
    ap.add_argument("--beta", type=float, help="Override beta value for KL penalty (SimPO/DPO)")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    preset = args.preset or cfg.get("preset", "balanced")

    base_name = cfg.get("model_name", "meta-llama/Llama-3.1-8B-Instruct")
    system_txt = cfg.get("system_prompt", "You are a helpful, concise assistant.")

    dpo_file = args.train_file or cfg["dpo"].get("train_file")
    if not dpo_file:
        raise ValueError("No train file provided. Set dpo.train_file in config or pass --train_file.")
    raw = load_dataset("json", data_files={"train": dpo_file})["train"]

    base_m, tok = load_base_and_tok(base_name)

    try:
        model = PeftModel.from_pretrained(base_m, args.sft_ckpt, is_trainable=True)
    except TypeError:
        model = PeftModel.from_pretrained(base_m, args.sft_ckpt)
        try:
            model.enable_adapter_layers()
        except AttributeError:
            pass
    try:
        model.config.use_cache = False
    except:
        pass

    ds = _ensure_pref_columns(raw, tok, system_txt)

    gbl = _gb(cfg["dpo"], "group_by_length", False, preset)
    if gbl:
        ds = _add_prompt_lengths_for_grouping(ds, tok)

    bs = _gi(cfg["dpo"], "per_device_train_batch_size", 4, preset, int)
    gas = _gi(cfg["dpo"], "gradient_accumulation_steps", 4, preset, int)
    lr = _gf(cfg["dpo"], "lr", 5e-6, preset)
    log = _gi(cfg["dpo"], "logging_steps", 10, preset, int)
    save = _gi(cfg["dpo"], "save_steps", 200, preset, int)
    steps = _gi(cfg["dpo"], "max_steps", 800, preset, int)
    if args.beta is not None:
        beta = float(args.beta)
        print(f"[override] Using beta from CLI: {beta}")
    else:
        beta = _gf(cfg["dpo"], "beta", 0.1, preset)
    ref_free = _gb(cfg["dpo"], "reference_free", False, preset)
    precomp = _gb(cfg["dpo"], "precompute_ref_log_probs", True, preset)

    mlen = int(_pick(cfg.get("pref_max_seq_len", _pick(cfg["sft"]["max_seq_len"], preset)), preset))
    mplen = int(_pick(cfg.get("pref_max_prompt_len", min(mlen, 512)), preset))

    out_dir = os.path.join(
        cfg.get("output_dir", "outputs"),
        args.algo,
        os.path.basename(base_name).replace("/", "-") + f"-lora-{args.algo}",
    )

    optim_name = "paged_adamw_8bit" if OPTIM_8BIT else "adamw_torch_fused"

    print(f"[PREF] algo={args.algo} preset={preset} steps={steps} bs={bs} gas={gas} lr={lr}")
    print(f"[PREF] seq={mlen} prompt={mplen} group_by_length={gbl}")
    print(f"[PREF] data={dpo_file}")
    print(f"[PREF] out_dir={out_dir}")

    trainer = None

    if (args.algo == "dpo") and (DPOConfig is not None) and (DPOTrainer is not None):
        cfg_sig = inspect.signature(DPOConfig.__init__)
        common = dict(
            output_dir=out_dir,
            per_device_train_batch_size=bs,
            gradient_accumulation_steps=gas,
            learning_rate=lr,
            logging_steps=log,
            save_steps=save,
            bf16=torch.cuda.is_available(),
            max_steps=steps,
            report_to=[],
            seed=42,
            overwrite_output_dir=True,
            tf32=True,
            padding_value=tok.pad_token_id,
            label_pad_token_id=-100,
            max_length=mlen,
            max_prompt_length=mplen,
            beta=beta,
            remove_unused_columns=False,
            optim=optim_name,
            dataloader_num_workers=6,
            dataloader_pin_memory=True,
            group_by_length=gbl,
            precompute_ref_log_probs=(False if args.algo == "simpo" else precomp),
            disable_reference_model=(True if args.algo == "simpo" else (not ref_free)),
            reference_free=(True if args.algo == "simpo" else ref_free),
            gradient_checkpointing=USE_GC,
            save_total_limit=_gi(cfg["dpo"], "save_total_limit", 2, preset, int),
            save_safetensors=_gb(cfg["dpo"], "save_safetensors", True, preset),
        )
        if "loss_type" in cfg_sig.parameters:
            desired_loss = "simpo" if args.algo == "simpo" else _pick(cfg["dpo"].get("loss_type", "sigmoid"), preset)
            common["loss_type"] = desired_loss
        if "simpo_gamma" in cfg_sig.parameters:
            common["simpo_gamma"] = float(_pick(cfg["dpo"].get("simpo_gamma", 0.5), preset))
        dpo_args = DPOConfig(**{k: v for k, v in common.items() if k in cfg_sig.parameters})

        tr_sig = inspect.signature(DPOTrainer.__init__)
        tkwargs = {"model": model, "args": dpo_args, "train_dataset": ds}
        if "processing_class" in tr_sig.parameters:
            tkwargs["processing_class"] = tok
        elif "tokenizer" in tr_sig.parameters:
            tkwargs["tokenizer"] = tok
        for k in ("precompute_ref_log_probs", "disable_reference_model", "reference_free", "group_by_length", "loss_type", "simpo_gamma"):
            if k in tr_sig.parameters and k not in tkwargs:
                tkwargs[k] = common[k]
        trainer = DPOTrainer(**tkwargs)

    elif args.algo == "simpo" and (CPOConfig is not None) and (CPOTrainer is not None):
        cfg_sig = inspect.signature(CPOConfig.__init__)
        common = dict(
            output_dir=out_dir,
            per_device_train_batch_size=bs,
            gradient_accumulation_steps=gas,
            learning_rate=lr,
            logging_steps=log,
            save_steps=save,
            bf16=torch.cuda.is_available(),
            max_steps=steps,
            report_to=[],
            seed=42,
            overwrite_output_dir=True,
            tf32=True,
            padding_value=tok.pad_token_id,
            label_pad_token_id=-100,
            max_length=mlen,
            max_prompt_length=mplen,
            remove_unused_columns=False,
            optim=optim_name,
            dataloader_num_workers=6,
            dataloader_pin_memory=True,
            group_by_length=gbl,
            gradient_checkpointing=USE_GC,
            save_total_limit=_gi(cfg["dpo"], "save_total_limit", 2, preset, int),
            save_safetensors=_gb(cfg["dpo"], "save_safetensors", True, preset),
            loss_type="simpo",
            beta=beta,
        )
        if "simpo_gamma" in cfg_sig.parameters:
            common["simpo_gamma"] = float(_pick(cfg["dpo"].get("simpo_gamma", 0.5), preset))
        if "cpo_alpha" in cfg_sig.parameters:
            common["cpo_alpha"] = 0.0
        sp_args = CPOConfig(**{k: v for k, v in common.items() if k in cfg_sig.parameters})

        tr_sig = inspect.signature(CPOTrainer.__init__)
        tkwargs = {"model": model, "args": sp_args, "train_dataset": ds}
        if "processing_class" in tr_sig.parameters:
            tkwargs["processing_class"] = tok
        elif "tokenizer" in tr_sig.parameters:
            tkwargs["tokenizer"] = tok
        if "group_by_length" in tr_sig.parameters:
            tkwargs["group_by_length"] = gbl
        trainer = CPOTrainer(**tkwargs)

    elif args.algo == "simpo" and (SimPOConfig is not None) and (SimPOTrainer is not None):
        cfg_sig = inspect.signature(SimPOConfig.__init__)
        common = dict(
            output_dir=out_dir,
            per_device_train_batch_size=bs,
            gradient_accumulation_steps=gas,
            learning_rate=lr,
            logging_steps=log,
            save_steps=save,
            bf16=torch.cuda.is_available(),
            max_steps=steps,
            report_to=[],
            seed=42,
            overwrite_output_dir=True,
            tf32=True,
            padding_value=tok.pad_token_id,
            label_pad_token_id=-100,
            max_length=mlen,
            max_prompt_length=mplen,
            beta=beta,
            remove_unused_columns=False,
            optim=optim_name,
            dataloader_num_workers=6,
            dataloader_pin_memory=True,
            group_by_length=gbl,
            gradient_checkpointing=USE_GC,
            save_total_limit=_gi(cfg["dpo"], "save_total_limit", 2, preset, int),
            save_safetensors=_gb(cfg["dpo"], "save_safetensors", True, preset),
        )
        sp_args = SimPOConfig(**{k: v for k, v in common.items() if k in cfg_sig.parameters})

        tr_sig = inspect.signature(SimPOTrainer.__init__)
        tkwargs = {"model": model, "args": sp_args, "train_dataset": ds}
        if "processing_class" in tr_sig.parameters:
            tkwargs["processing_class"] = tok
        elif "tokenizer" in tr_sig.parameters:
            tkwargs["tokenizer"] = tok
        if "group_by_length" in tr_sig.parameters:
            tkwargs["group_by_length"] = gbl
        trainer = SimPOTrainer(**tkwargs)

    elif args.algo == "orpo" and (ORPOConfig is not None) and (ORPOTrainer is not None):
        cfg_sig = inspect.signature(ORPOConfig.__init__)
        common = dict(
            output_dir=out_dir,
            per_device_train_batch_size=bs,
            gradient_accumulation_steps=gas,
            learning_rate=lr,
            logging_steps=log,
            save_steps=save,
            bf16=torch.cuda.is_available(),
            max_steps=steps,
            report_to=[],
            seed=42,
            overwrite_output_dir=True,
            tf32=True,
            padding_value=tok.pad_token_id,
            label_pad_token_id=-100,
            max_length=mlen,
            max_prompt_length=mplen,
            beta=beta,
            remove_unused_columns=False,
            optim=optim_name,
            dataloader_num_workers=6,
            dataloader_pin_memory=True,
            group_by_length=gbl,
            gradient_checkpointing=USE_GC,
            save_total_limit=_gi(cfg["dpo"], "save_total_limit", 2, preset, int),
            save_safetensors=_gb(cfg["dpo"], "save_safetensors", True, preset),
        )
        orpo_args = ORPOConfig(**{k: v for k, v in common.items() if k in cfg_sig.parameters})

        tr_sig = inspect.signature(ORPOTrainer.__init__)
        tkwargs = {"model": model, "args": orpo_args, "train_dataset": ds}
        if "processing_class" in tr_sig.parameters:
            tkwargs["processing_class"] = tok
        elif "tokenizer" in tr_sig.parameters:
            tkwargs["tokenizer"] = tok
        if "group_by_length" in tr_sig.parameters:
            tkwargs["group_by_length"] = gbl
        trainer = ORPOTrainer(**tkwargs)

    else:
        if args.algo == "simpo":
            raise RuntimeError("SimPO not supported by this TRL. Use --algo dpo or upgrade TRL.")
        if DPOTrainer is None:
            raise RuntimeError("DPOTrainer not available. Upgrade TRL (pip install -U trl).")

        ta_sig = inspect.signature(TrainingArguments.__init__)
        ta_kwargs = dict(
            output_dir=out_dir,
            per_device_train_batch_size=bs,
            gradient_accumulation_steps=gas,
            learning_rate=lr,
            max_steps=steps,
            logging_steps=log,
            save_steps=save,
            report_to=[],
            bf16=torch.cuda.is_available(),
            seed=42,
            overwrite_output_dir=True,
            tf32=True,
        )
        if "optim" in ta_sig.parameters:
            ta_kwargs["optim"] = optim_name
        if "dataloader_num_workers" in ta_sig.parameters:
            ta_kwargs["dataloader_num_workers"] = 6
        if "dataloader_pin_memory" in ta_sig.parameters:
            ta_kwargs["dataloader_pin_memory"] = True
        if "evaluation_strategy" in ta_sig.parameters:
            ta_kwargs["evaluation_strategy"] = "no"
        if "save_total_limit" in ta_sig.parameters:
            ta_kwargs["save_total_limit"] = _gi(cfg["dpo"], "save_total_limit", 2, preset, int)
        if "save_safetensors" in ta_sig.parameters:
            ta_kwargs["save_safetensors"] = _gb(cfg["dpo"], "save_safetensors", True, preset)
        if "gradient_checkpointing" in ta_sig.parameters:
            ta_kwargs["gradient_checkpointing"] = USE_GC

        targs = TrainingArguments(**ta_kwargs)

        tr_sig = inspect.signature(DPOTrainer.__init__)
        tkwargs = {"model": model, "args": targs, "train_dataset": ds}
        if "processing_class" in tr_sig.parameters:
            tkwargs["processing_class"] = tok
        elif "tokenizer" in tr_sig.parameters:
            tkwargs["tokenizer"] = tok
        if "max_length" in tr_sig.parameters:
            tkwargs["max_length"] = mlen
        if "max_prompt_length" in tr_sig.parameters:
            tkwargs["max_prompt_length"] = mplen
        if "beta" in tr_sig.parameters:
            tkwargs["beta"] = beta
        if args.algo == "simpo" and "loss_type" in tr_sig.parameters:
            tkwargs["loss_type"] = "simpo"
        if args.algo == "simpo" and "simpo_gamma" in tr_sig.parameters:
            tkwargs["simpo_gamma"] = float(_pick(cfg["dpo"].get("simpo_gamma", 0.5), preset))
        for k, v in (("precompute_ref_log_probs", (False if args.algo=="simpo" else precomp)), ("disable_reference_model", (True if args.algo=="simpo" else (not ref_free)))):
            if k in tr_sig.parameters:
                tkwargs[k] = v
        if "group_by_length" in tr_sig.parameters:
            tkwargs["group_by_length"] = gbl

        trainer = DPOTrainer(**tkwargs)

    resume_path = None
    if args.resume:
        try:
            last = get_last_checkpoint(out_dir)
            if last:
                resume_path = last
            print(f"[resume] Resuming from {resume_path}")
        except Exception as e:
            print(f"[warn] get_last_checkpoint failed: {e}")

    if USE_GC and hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable()
            print("[GC] Enabled gradient checkpointing")
        except Exception:
            pass

    if not args.no_autosave and args.autosave_steps > 0:
        cb = SafeSaveCallback(model, tok, out_dir, every_steps=args.autosave_steps, keep_last=args.autosave_keep)
        trainer.add_callback(cb)

    try:
        eta_cb = ETACallback(total_steps=steps, log_every=max(1, log))
        trainer.add_callback(eta_cb)
    except Exception:
        pass

    def _signal_handler(sig, frame):
        print(f"\n[signal] Caught {sig}. Saving emergency checkpoint...")
        emergency_save(model, tok, out_dir)
        raise SystemExit(0)

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    try:
        trainer.train(resume_from_checkpoint=resume_path)
    except Exception as e:
        print(f"[error] training crashed: {e}")
        emergency_save(model, tok, out_dir)
        raise
    finally:
        emergency_save(model, tok, out_dir)

    trainer.save_model(out_dir)
    tok.save_pretrained(out_dir)
    print(f"[{args.algo.upper()}] saved LoRA to {out_dir}")

if __name__ == "__main__":
    main()
