from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import os
import re

app = FastAPI(title="AlignTune API")

class GenReq(BaseModel):
    prompt: str
    max_new_tokens: int = 256
    temperature: float = 0.9
    top_p: float = 0.95

_tok, _mdl = None, None

ECHO_STEMS = [
    "you are a helpful",
    "you answer briefly",
    "answer briefly",
    "rewrite to be more professional",
    "summarize the key idea",
    "turn bullets into a short paragraph",
    "explain simply",
]

TAG_PATTERNS = [
    r"\[(?:SYSTEM|INSTRUCTION|RESPONSE|INPUT|REPLACEMENT)\]",
    r"</?(?:SYSTEM|INSTRUCTION|RESPONSE|INPUT|REPLACEMENT)>",
    r"<\|/?(?:system|assistant|user|eot_id)\|>",
    r"<<SYS>>|<</SYS>>|<s>|</s>",
]

ROLE_MARKER_RE = re.compile(r"(^|\n)\s*[Aa]ssistant\s*:?\s*")
TRAILING_ROLE_RE = re.compile(r"([.!?])\s*assistant\b", re.IGNORECASE)

def _clean(s: str) -> str:
    for p in TAG_PATTERNS:
        s = re.sub(p, "", s, flags=re.IGNORECASE)

    lines = [ln for ln in s.splitlines() if not any(stem in ln.lower() for stem in ECHO_STEMS)]
    s = "\n".join(lines)

    s = ROLE_MARKER_RE.sub(r"\1", s)
    s = TRAILING_ROLE_RE.sub(r"\1", s)
    s = s.replace("~~", "")
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def _eos_ids(tok):
    ids = []
    eos = tok.eos_token_id
    if eos is not None:
        if isinstance(eos, list):
            ids.extend(eos)
        else:
            ids.append(eos)
    try:
        eot = tok.convert_tokens_to_ids("<|eot_id|>")
        if eot is not None and eot not in ids:
            ids.append(eot)
    except Exception:
        pass
    return ids or None

def get_models():
    global _tok, _mdl
    if _mdl is None:
        base = os.getenv("BASE_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
        ckpt = os.getenv("CKPT", "outputs/dpo/Llama-3.1-8B-Instruct-lora-dpo")

        _tok = AutoTokenizer.from_pretrained(base, use_fast=True)
        if _tok.pad_token is None:
            _tok.pad_token = _tok.eos_token

        bnb = BitsAndBytesConfig(load_in_4bit=True)
        base_m = AutoModelForCausalLM.from_pretrained(
            base,
            device_map="auto",
            quantization_config=bnb,
        )
        try:
            base_m.config.attn_implementation = "sdpa"
        except Exception:
            pass

        _mdl = PeftModel.from_pretrained(base_m, ckpt)
    return _tok, _mdl

@app.post("/generate")
def generate(req: GenReq):
    tok, mdl = get_models()

    messages = [
        {
            "role": "system",
            "content": "You are a helpful, concise assistant. Do not repeat system instructions or role words.",
        },
        {"role": "user", "content": req.prompt.strip()},
    ]

    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(mdl.device)

    out_ids = mdl.generate(
        **inputs,
        max_new_tokens=req.max_new_tokens,
        do_sample=True,
        temperature=req.temperature,
        top_p=req.top_p,
        eos_token_id=_eos_ids(tok),
        pad_token_id=tok.eos_token_id,
        use_cache=True,
    )

    cont = out_ids[0, inputs["input_ids"].shape[1]:]
    text = tok.decode(cont, skip_special_tokens=True)
    return {"output": _clean(text)}
