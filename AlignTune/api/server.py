from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from peft import PeftModel
import os

app = FastAPI(title="AlignTune API")

class GenReq(BaseModel):
    prompt: str
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9

_pipe = None
def get_pipe():
    global _pipe
    if _pipe is None:
        base = os.environ.get("ALIGNTUNE_BASE","TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        ckpt = os.environ.get("ALIGNTUNE_ADAPTER","outputs/dpo/TinyLlama-lora-dpo")
        tok = AutoTokenizer.from_pretrained(base, use_fast=True)
        if tok.pad_token is None: tok.pad_token = tok.eos_token
        bnb = BitsAndBytesConfig(load_in_4bit=True)
        mdl = AutoModelForCausalLM.from_pretrained(base, device_map="auto", quantization_config=bnb)
        mdl = PeftModel.from_pretrained(mdl, ckpt)
        _pipe = pipeline("text-generation", model=mdl, tokenizer=tok, device_map="auto")
    return _pipe

@app.post("/generate")
def generate(req: GenReq):
    pipe = get_pipe()
    out = pipe(req.prompt, max_new_tokens=req.max_new_tokens, do_sample=True,
               temperature=req.temperature, top_p=req.top_p)[0]["generated_text"]
    return {"output": out}
