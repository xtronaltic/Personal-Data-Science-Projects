# AlignTune: Modern LLM Fine‑Tuning & Preference Optimization (SFT → DPO/SimPO/ORPO)

A **recruiter-ready**, 2025‑style project that demonstrates **parameter‑efficient fine‑tuning** and **preference alignment** with multiple objectives:

- **SFT (LoRA/QLoRA)** using `transformers` + `peft` (+ optional **Unsloth** acceleration).
- **Preference tuning:** **DPO**, **SimPO**, and **ORPO** (choose via config; auto‑fallbacks if classes not available).
- **Evaluation:**
  - Task metrics (Exact‑Match, ROUGE‑L).
  - **Judge‑based win rate** (MT‑Bench/AlpacaEval style) using a local “judge” model (or wire your own API).
  - **Safety checks** (toy refusal/harm heuristics) + hooks for HarmBench/JailbreakBench.
- **Serving:** FastAPI server (default) + optional **vLLM** server for throughput.
- **DX:** Config‑first design, CI tests, Model Card, Dockerfile, Gradio demo.

> Ships with tiny synthetic data so it runs anywhere. Swap in your domain data for meaningful results.

## Quickstart (CPU works; GPU recommended)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip && pip install -r requirements.txt

# If you have NO NVIDIA GPU: open configs/base.yaml and set:
#   lora.qlora: false
#   lora.load_in_4bit: false

# 1) SFT (LoRA/QLoRA)
python scripts/train_sft.py --config configs/base.yaml

# 2) Preference tuning (choose one; DPO is default)
python scripts/train_dpo.py   --config configs/base.yaml --sft_ckpt outputs/sft/TinyLlama-lora
python scripts/train_simpo.py --config configs/base.yaml --sft_ckpt outputs/sft/TinyLlama-lora   # if SimPOTrainer available
python scripts/train_orpo.py  --config configs/base.yaml --sft_ckpt outputs/sft/TinyLlama-lora   # if ORPOTrainer available

# 3) Evaluate
python scripts/eval.py --config configs/base.yaml --ckpt outputs/dpo/TinyLlama-lora-dpo
python eval/judge_winrate.py --config configs/base.yaml --ckpt_a outputs/dpo/TinyLlama-lora-dpo --ckpt_b outputs/sft/TinyLlama-lora
python eval/safety_check.py  --config configs/base.yaml --ckpt outputs/dpo/TinyLlama-lora-dpo

# 4) Serve + demo
uvicorn api.server:app --reload
python app/demo.py --ckpt outputs/dpo/TinyLlama-lora-dpo

# (Optional) vLLM server (needs vllm installed)
python api/server_vllm.py --ckpt outputs/dpo/TinyLlama-lora-dpo --base TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

## Switch models
- Default: `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (small; no license gate).
- Larger: `meta-llama/Llama-3.1-8B-Instruct` (accept HF license). Update `configs/base.yaml:model_name`.

## Data formats
- **SFT** (`data/sft/*.jsonl`): `{ "instruction", "input", "output" }`
- **DPO/SimPO/ORPO** (`data/dpo/*.jsonl`): `{ "prompt", "chosen", "rejected" }`

See `data/DATA_CARD.md` for details.
