import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--lora", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    args = ap.parse_args()

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    tok = AutoTokenizer.from_pretrained(args.base, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print("Loading base...")
    base = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=dtype, device_map="auto")

    print("Attaching LoRA...")
    model = PeftModel.from_pretrained(base, args.lora)

    print("Merging...")
    merged = model.merge_and_unload()
    try:
        merged.config.attn_implementation = "sdpa"
    except:
        pass

    print(f"Saving to {args.out} ...")
    merged.save_pretrained(args.out, safe_serialization=True)
    tok.save_pretrained(args.out)
    print("Done.")

if __name__ == "__main__":
    main()
