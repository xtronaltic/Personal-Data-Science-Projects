from typing import Dict
SYS = "You are a helpful, concise assistant."
def format_sft(example: Dict) -> str:
    instr = example.get("instruction","").strip()
    inp = example.get("input","").strip()
    parts = [f"<s>[SYSTEM]\n{SYS}\n</s>"]
    if instr: parts.append(f"[INSTRUCTION]\n{instr}")
    if inp: parts.append(f"[INPUT]\n{inp}")
    parts.append("[RESPONSE]\n")
    return "\n".join(parts)
def format_dpo_prompt(prompt: str) -> str:
    return f"<s>[SYSTEM]\n{SYS}\n</s>\n[INSTRUCTION]\n{prompt}\n[RESPONSE]\n"
BAD_WORDS = {"kill","hate","violence","bomb"}
def safety_flag(text: str) -> bool:
    t = text.lower()
    return any(w in t for w in BAD_WORDS)
REFUSAL_HINTS = ("cannot","can not","i can't","i cannot","sorry","i won't","i will not","i’m unable","not able")
def looks_like_refusal(text: str) -> bool:
    t = text.strip().lower()
    return any(h in t for h in REFUSAL_HINTS)
