import argparse
from vllm import LLM, SamplingParams

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--ckpt", default=None, help="Optional LoRA adapter (not applied in this minimal example)")
    ap.add_argument("--port", type=int, default=8001)
    args = ap.parse_args()

    # Minimal vLLM run for base model
    llm = LLM(model=args.base)
    sp = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=256)

    print(f"vLLM loaded: {args.base}. This example prints one response and exits.")
    print("Prompt: Define PEFT in one line.")
    out = llm.generate(["Define PEFT in one line."], sp)[0].outputs[0].text
    print("Response:", out)

if __name__ == "__main__":
    main()
