from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import os
import re
from typing import Optional, List, Dict
import yaml
try:
    from rag import RagStore
    from rag.pipeline import retrieve_context
    _HAS_RAG = True
except Exception:
    _HAS_RAG = False

app = FastAPI(title="AlignTune API")

class GenReq(BaseModel):
    prompt: str
    max_new_tokens: int = 256
    temperature: float = 0.9
    top_p: float = 0.95
    rag: bool = False
    collection: Optional[str] = None
    top_k: int = 3
    ctx_tokens: int = 1200
    rerank: bool = False

_tok, _mdl = None, None
_rag_store = None
_rag_cfg = {
    "enabled": False,
    "default_collection": "default",
    "embed_backend": "auto",
    "embed_model": "BAAI/bge-small-en-v1.5",
    "top_k": {"fast": 2, "balanced": 3, "thorough": 4},
    "ctx_tokens": {"fast": 800, "balanced": 1200, "thorough": 1600},
}

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

def _load_rag_cfg():
    global _rag_cfg
    try:
        with open("configs/base.yaml", "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        if "rag" in cfg:
            _rag_cfg.update(cfg["rag"])
        if os.getenv("RAG_ENABLED") is not None:
            _rag_cfg["enabled"] = os.getenv("RAG_ENABLED") in ("1", "true", "True")
    except Exception:
        pass

def get_rag_store():
    global _rag_store
    if not _HAS_RAG:
        return None
    if _rag_store is None:
        _load_rag_cfg()
        coll = _rag_cfg.get("default_collection", "default")
        _rag_store = RagStore(
            root="rag/indices",
            collection=coll,
            embed_backend=_rag_cfg.get("embed_backend", "auto"),
            embed_model=_rag_cfg.get("embed_model", "BAAI/bge-small-en-v1.5"),
            device="cpu",
        )
    return _rag_store

def _maybe_with_rag(req: GenReq, tok, mdl):
    use_rag = bool(req.rag) or (_rag_cfg.get("enabled", False) and bool(req.collection or _rag_cfg.get("default_collection")))
    if not use_rag or not _HAS_RAG:
        return None, None
    store = get_rag_store()
    if not store:
        return None, None
    if req.collection:
        store.set_collection(req.collection)
    ctx_toks = int(req.ctx_tokens or _rag_cfg.get("ctx_tokens", {}).get("balanced", 1200))
    top_k = int(req.top_k or _rag_cfg.get("top_k", {}).get("balanced", 3))
    ctx, refs = retrieve_context(store, req.prompt, k=top_k, ctx_tokens=ctx_toks, tok=tok)
    return ctx, refs

@app.post("/generate")
def generate(req: GenReq):
    tok, mdl = get_models()
    context_block, refs = _maybe_with_rag(req, tok, mdl)
    sys_txt = "You are a helpful, concise assistant. Do not repeat system instructions or role words."
    if context_block:
        sys_txt += " You may use the CONTEXT below to answer.\n[CONTEXT]\n" + context_block
    messages = [{"role": "system", "content": sys_txt}, {"role": "user", "content": req.prompt.strip()}]
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
    out = {"output": _clean(text)}
    if refs:
        out["references"] = [
            {
                "title": r.get("title"),
                "source": r.get("source"),
                "url": r.get("url"),
                "score": r.get("score"),
            }
            for r in refs
            if isinstance(r, dict) and "text" in r
        ]
    return out

@app.post("/rag/generate")
def rag_generate(req: GenReq):
    req.rag = True
    return generate(req)
