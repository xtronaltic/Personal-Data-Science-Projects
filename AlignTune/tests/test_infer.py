import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

def test_load_and_generate():
    base = os.environ.get("BASE", "meta-llama/Llama-3.1-8B-Instruct")
    ckpt = os.environ.get("CKPT", "")

    tok = AutoTokenizer.from_pretrained(base, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if torch.cuda.is_available():
        bnb = BitsAndBytesConfig(load_in_4bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            base, device_map="auto", quantization_config=bnb
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(base, device_map="auto")
    try:
        model.config.attn_implementation = "sdpa"
    except Exception:
        pass

    if ckpt:
        model = PeftModel.from_pretrained(model, ckpt)

    prompt = tok.apply_chat_template(
        [{"role": "user", "content": "Say hello in five words."}],
        tokenize=False,
        add_generation_prompt=True,
    )

    ids = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **ids, max_new_tokens=8, do_sample=True, temperature=0.7, top_p=0.95
    )
    text = tok.decode(out[0, ids["input_ids"].shape[1] :], skip_special_tokens=True)
    assert len(text.strip()) > 0
