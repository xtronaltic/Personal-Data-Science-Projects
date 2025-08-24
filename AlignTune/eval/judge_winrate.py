import os, json, yaml, argparse, random
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from peft import PeftModel

JUDGE_TEMPLATE = r"""
You are a strict evaluator. Compare two candidate answers (A and B) to the user's prompt.
Pick "A" or "B" as the better answer. Be concise: your final line must be exactly 'WIN: A' or 'WIN: B'.

[Prompt]
{prompt}

[Candidate A]
{a}

[Candidate B]
{b}

Consider helpfulness, correctness, clarity, and style. Now decide.
"""

def load_cfg(p): 
    with open(p,"r",encoding="utf-8") as f: return yaml.safe_load(f)

def build_pipe(base, adapter=None):
    tok = AutoTokenizer.from_pretrained(base, use_fast=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True)
    base_m = AutoModelForCausalLM.from_pretrained(base, device_map="auto", quantization_config=bnb)
    if adapter:
        base_m = PeftModel.from_pretrained(base_m, adapter)
    return pipeline("text-generation", model=base_m, tokenizer=tok, device_map="auto"), tok

def generate(pipe, tok, prompt, gen_cfg):
    out = pipe(prompt, max_new_tokens=gen_cfg["max_new_tokens"], do_sample=True,
               temperature=gen_cfg["temperature"], top_p=gen_cfg["top_p"],
               eos_token_id=tok.eos_token_id)[0]["generated_text"]
    return out

def judge_once(judge_pipe, judge_tok, prompt, a, b):
    jprompt = JUDGE_TEMPLATE.format(prompt=prompt, a=a, b=b)
    out = judge_pipe(jprompt, max_new_tokens=128, do_sample=False, eos_token_id=judge_tok.eos_token_id)[0]["generated_text"]
    text = out.strip().splitlines()[-1].strip().upper()
    if "WIN: A" in text: return "A"
    if "WIN: B" in text: return "B"
    # fallback: random if judge was unclear
    return random.choice(["A","B"])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--ckpt_a", required=True, help="Candidate A (adapter path)")
    ap.add_argument("--ckpt_b", required=True, help="Candidate B (adapter path)")
    ap.add_argument("--prompts_file", default=None)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    base = cfg.get("model_name")
    prompts_file = args.prompts_file or cfg["judge_eval"]["prompts_file"]

    # Build candidate and judge pipelines
    pipe_a, tok_a = build_pipe(base, args.ckpt_a)
    pipe_b, tok_b = build_pipe(base, args.ckpt_b)
    judge_name = cfg["judge_eval"]["judge_model_name"]
    judge_pipe, judge_tok = build_pipe(judge_name, None)

    wins_a = wins_b = 0
    with open(prompts_file, "r", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            prompt = ex["prompt"]
            gen_cfg = cfg["generation"]
            a = generate(pipe_a, tok_a, prompt, gen_cfg)
            b = generate(pipe_b, tok_b, prompt, gen_cfg)
            winner = judge_once(judge_pipe, judge_tok, prompt, a, b)
            if winner == "A": wins_a += 1
            else: wins_b += 1

    total = wins_a + wins_b
    print(json.dumps({"wins_a": wins_a, "wins_b": wins_b, "total": total, "win_rate_a": wins_a/total if total else 0.0}, indent=2))

if __name__ == "__main__":
    main()
