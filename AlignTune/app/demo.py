import argparse
import os
import sys
import re
import random

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    from transformers import BitsAndBytesConfig

    _HAS_BNB = True
except Exception:
    BitsAndBytesConfig = None
    _HAS_BNB = False

from peft import PeftModel

SYSTEM_PROMPT = (
    "You are a helpful, concise assistant. "
    "Do not repeat system instructions or role words. "
    "Reply directly to the user."
)

SIMPLE_BLOCKLIST = [
    "ignore previous",
    "jailbreak",
    "do anything now",
    "bypass safety",
    "how to make a bomb",
    "make explosives",
    "credit card numbers",
]


def screen_for_obvious_unsafe(text: str) -> str | None:
    low = (text or "").lower()
    for k in SIMPLE_BLOCKLIST:
        if k in low:
            return "Sorry — I can’t help with that."
    return None


ECHO_STEMS = [
    "you are a helpful",
    "you answer briefly",
    "answer briefly",
    "rewrite to be more professional",
    "summarize the key idea",
    "turn bullets into a short paragraph",
    "explain simply",
    "roll over to answer briefly",
    "answer in one sentence",
]

TAG_PATTERNS = [
    r"\[(?:SYSTEM|INSTRUCTION|RESPONSE|INPUT|REPLACEMENT)\]",
    r"</?(?:SYSTEM|INSTRUCTION|RESPONSE|INPUT|REPLACEMENT)>",
    r"<\|/?(?:system|assistant|user|eot_id)\|>",
    r"<<SYS>>|<</SYS>>|<s>|</s>",
]

ROLE_MARKER_RE = re.compile(r"(^|\n)\s*[Aa]ssistant\s*:?\s*")
TRAILING_ROLE_RE = re.compile(r"([.!?])\s*assistant\b", re.IGNORECASE)
BANNED_LINE_RE = re.compile(
    r"(?im)^\s*(answer\s+(briefly|concisely)|rewrite\s+to\s+be\s+more\s+professional|"
    r"summarize\s+the\s+key\s+idea|turn\s+bullets\s+into\s+a\s+short\s+paragraph|"
    r"\[/?(system|instruction|response|input|replacement)\])[:.\s]*$"
)


def strip_instruction_echo(s: str) -> str:
    for p in TAG_PATTERNS:
        s = re.sub(p, "", s, flags=re.IGNORECASE)

    lines = [ln for ln in s.splitlines() if not BANNED_LINE_RE.match(ln)]
    s = "\n".join(lines)

    lines = [ln for ln in s.splitlines() if not any(stem in ln.lower() for stem in ECHO_STEMS)]
    s = "\n".join(lines)

    s = ROLE_MARKER_RE.sub(r"\1", s)
    s = TRAILING_ROLE_RE.sub(r"\1", s)

    s = s.replace("~~", "")
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def looks_incomplete(t: str) -> bool:
    t = t.strip()
    if not t:
        return True
    return (len(t.split()) < 6) or (t[-1:] not in ".!?”") or t.endswith((":", ";", ","))


def detect_intent(user_text: str):
    p = user_text.lower()
    if "joke" in p:
        if ("two" in p and "line" in p) or ("one-liner" in p) or ("one liner" in p):
            return "joke_short"
        if "long" in p:
            return "joke_long"
        return "joke"
    if any(w in p for w in ["explain", "teach", "what is", "what are", "how does", "how do"]):
        return "explain"
    if any(w in p for w in ["rewrite", "polish", "improve", "professional"]):
        return "rewrite"
    return "qa"


def params_for_intent(kind: str):
    profiles = {
        "joke_short": (24, 96, 0.95, 0.95, 1.02, 0, 4, 6),
        "joke_long": (64, 320, 0.95, 0.95, 1.02, 0, 4, 6),
        "joke": (40, 192, 0.95, 0.95, 1.02, 0, 4, 6),
        "explain": (32, 256, 0.75, 0.90, 1.03, 0, 3, 5),
        "rewrite": (24, 192, 0.85, 0.95, 1.05, 3, 3, 5),
        "qa": (16, 160, 0.80, 0.95, 1.03, 0, 3, 5),
    }
    return profiles.get(kind, profiles["qa"])


def load_model_and_tok(ckpt: str, base: str):
    tok = AutoTokenizer.from_pretrained(base, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    quant_cfg = BitsAndBytesConfig(load_in_4bit=True) if _HAS_BNB else None
    base_m = AutoModelForCausalLM.from_pretrained(
        base, device_map="auto", quantization_config=quant_cfg, torch_dtype="auto"
    )
    try:
        base_m.config.attn_implementation = "sdpa"
    except Exception:
        pass
    model = PeftModel.from_pretrained(base_m, ckpt)
    return model, tok


def _safe_generate(model, **kwargs):
    try:
        return model.generate(**kwargs)
    except TypeError:
        kwargs.pop("min_new_tokens", None)
        kwargs.pop("no_repeat_ngram_size", None)
        return model.generate(**kwargs)


def build_chat_prompt(tok, user_text: str) -> str:
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text.strip()},
    ]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def eos_ids(tok):
    ids = [tok.eos_token_id]
    try:
        eot = tok.convert_tokens_to_ids("<|eot_id|>")
        if eot is not None and eot not in ids:
            ids.append(eot)
    except Exception:
        pass
    return ids


def sample_once(model, tok, prompt, min_new, max_new, temp, top_p, rep_pen, no_repeat):
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    gen_ids = _safe_generate(
        model,
        **inputs,
        min_new_tokens=int(min_new),
        max_new_tokens=int(max_new),
        do_sample=True,
        temperature=float(temp),
        top_p=float(top_p),
        repetition_penalty=float(rep_pen),
        no_repeat_ngram_size=int(no_repeat),
        eos_token_id=eos_ids(tok),
        pad_token_id=tok.eos_token_id,
        use_cache=True,
    )
    cont = gen_ids[0, inputs["input_ids"].shape[1]:]
    text = tok.decode(cont, skip_special_tokens=True)
    return strip_instruction_echo(text)


def score_candidate(text: str, intent: str):
    if not text:
        return -1e9
    L = len(text)
    score = 0.0
    if any(stem in text.lower() for stem in ECHO_STEMS):
        score -= 3.0
    if text[-1:] in ".!?":
        score += 1.0
    if intent.startswith("joke"):
        if "\n" in text:
            score += 1.0
        score += min(L, 240) / 240.0
    elif intent == "explain":
        score += min(L, 400) / 400.0
    else:
        score += min(L, 220) / 220.0
    uniq = len(set(text.split()))
    if uniq > 0 and (L / max(1, uniq)) > 6.0:
        score -= 0.5
    return score


def finish_the_thought(model, tok, prompt, text, extra_tokens=96, temp=0.9, top_p=0.95):
    if not looks_incomplete(text):
        return text
    cont_prompt = prompt + text + " "
    inputs = tok(cont_prompt, return_tensors="pt").to(model.device)
    gen_ids2 = _safe_generate(
        model,
        **inputs,
        max_new_tokens=int(extra_tokens),
        do_sample=True,
        temperature=float(temp),
        top_p=float(top_p),
        repetition_penalty=1.0,
        no_repeat_ngram_size=0,
        eos_token_id=eos_ids(tok),
        pad_token_id=tok.eos_token_id,
        use_cache=True,
    )
    more = tok.decode(gen_ids2[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return strip_instruction_echo((text + " " + more).strip())


def generate_best(model, tok, user_text):
    intent = detect_intent(user_text)
    min_new, max_new, temp, top_p, rep_pen, no_repeat, best_of, max_attempts = params_for_intent(intent)
    prompt = build_chat_prompt(tok, user_text)
    cands = []
    attempts = 0
    while len(cands) < best_of and attempts < max_attempts:
        text = sample_once(model, tok, prompt, min_new, max_new, temp, top_p, rep_pen, no_repeat)
        attempts += 1
        if any(stem in text.lower() for stem in ECHO_STEMS):
            continue
        cands.append((score_candidate(text, intent), text))
    if not cands:
        text = sample_once(model, tok, prompt, min_new, max_new, temp, top_p, rep_pen, no_repeat)
        return finish_the_thought(
            model, tok, prompt, text, extra_tokens=max(64, max_new // 2), temp=temp, top_p=top_p
        ).strip()
    cands.sort(key=lambda x: x[0], reverse=True)
    best = cands[0][1]
    best = finish_the_thought(
        model, tok, prompt, best, extra_tokens=max(64, max_new // 2), temp=temp, top_p=top_p
    )
    return best.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/dpo/Llama-3.1-8B-Instruct-lora-dpo")
    ap.add_argument("--base", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()

    model, tok = load_model_and_tok(args.ckpt, args.base)

    with gr.Blocks() as demo:
        gr.Markdown("# AlignTune — Tianpeng Gai (Leo)")
        chat = gr.Chatbot(type="messages", height=500)
        with gr.Row():
            msg = gr.Textbox(placeholder="Ask anything… e.g., tell me a joke", scale=8)
            send = gr.Button("Send", variant="primary", scale=1)

        def on_send(user_msg, messages):
            messages = messages or []
            text = (user_msg or "").strip()
            if not text:
                return messages, ""
            messages.append({"role": "user", "content": text})
            blocked = screen_for_obvious_unsafe(text)
            if blocked:
                messages.append({"role": "assistant", "content": blocked})
                return messages, ""
            ans = generate_best(model, tok, text)
            messages.append({"role": "assistant", "content": ans})
            return messages, ""

        send.click(on_send, inputs=[msg, chat], outputs=[chat, msg])
        msg.submit(on_send, inputs=[msg, chat], outputs=[chat, msg])

    demo.launch(server_name="127.0.0.1", server_port=args.port)


if __name__ == "__main__":
    main()
