# AlignTune: Modern LLM Fine‑Tuning & Preference Optimization

## Overview:

* Production‑style project showing how to train, align, and evaluate chat LLMs efficiently. Quantify gains from inserting SimPO before DPO polish and validate via automated ablation + human-style judge comparisons and safety probes. 

## Baseline flows:

* With SimPO: SFT → SimPO (beta=2.0) → mine hard pairs → short DPO polish (beta=0.1)
* Without SimPO: SFT → short DPO (beta=0.1)
* ORPO
   * Supported: Yes — scripts/train_pref.py implements --algo orpo and Makefile has orpo target..
   * Not in default E2E flows, balanced and nosimpo don’t include ORPO by default. I can add it to ablations if a 3‑way comparison is needed.

## End‑to‑End engineering:

* Fully automated E2E flow: data build → SFT → preference tuning (SimPO/DPO/ORPO) → evaluation → bundle.
* Fast & full suite tests (CI‑style, CPU‑friendly, switch models between Llama-3.1-8B-Instruct and TinyLlama-1.1B-Chat-v1.0)
* Repro knobs: Makefile targets, seeded prompt sampling, env toggles for memory.
* Interactive demo & serving: lightweight Gradio chat demo (`app/demo.py`) with leak-free prompt/templates and minimal post-processing for safe previews; production FastAPI server (`api/server.py`) for programmatic serving.

## Data Engineering:

* Streaming ingestion: Hugging Face datasets processed in a memory‑safe way.
* Sensible defaults: Balanced preset capped ~34k SFT; DPO ~30k.
* Quality filters: Simple toxicity skim + banned template stems; near‑dup removal via Jaccard.
* Preset caps: Fast/balanced/thorough sizes; per‑source caps; progress logs; bounded scans for predictability.
* Data formats:
   * SFT (data/sft/*.jsonl): { "instruction", "input", "output" }
   * DPO/SimPO/ORPO (data/dpo/*.jsonl): { "prompt", "chosen", "rejected" }
* Deterministic RNG seed (42) used across data sampling and judge prompt selection
* Repro knobs: Env overrides for caps and windows; clear DATA_CARD.md.

## Training Stack:

* transformers: Base model and tokenization layer
   * Backbone: Llama/TinyLlama with SDPA attention backend
   * Left-padded decoder optimization in eval
   * FastAPI integration for serving
   * TF32 matrix multiply acceleration with minimal precision trade-off

* peft: Parameter-efficient adaptation layer (LoRA)
   * LoRA adapters (r=8/16, alpha=32)
   * Target: q/k/v/o + gate/up/down projections
   * Efficient parameter fine-tuning (~0.1% trainable params)

* bitsandbytes: Quantization and optimization layer
   * QLoRA: 4-bit quantized base model
   * 8-bit optimizer (ALIGNTUNE_OPTIM_8BIT=1)
   * NF4/GPTQ compatibility for inference

* trl: Preference learning algorithms layer
   * DPO with β=0.1 for conservative updates
   * SimPO (β=2.0) via CPOTrainer fallback
   * ORPO with version-aware dispatch
   * Unified preference trainer interface

* unsloth:
   * Flash attention optimizations
   * Length-aware sequence batching
   * Preset-driven memory configs:
   * fast: 1536, balanced: 2048, thorough: 3072

* Key Optimizations:
   * VRAM: 4-bit model + 8-bit Adam + grad checkpointing
   * Training: Autosave + emergency snapshots + CLI overrides + ETA logging
   * Inference: Left-padding + SDPA backend + batched eval
   * Efficiency: 4‑bit base loading, length grouping, and near‑dup windows tuned per preset; fail‑soft behavior skips problematic sources, logs progress, and stops on caps reliably, preset-driven pref_max_seq_len and pref_max_prompt_len to control memory and prompt context per run, DPO advanced flags support precompute_ref_log_probs and reference_free modes (configurable per-preset) to trade compute vs determinism.

## Evaluation & Reporting

* Automated ablations for EM/ROUGE-L, judge flow now uses a seeded 1000-sample prompt file; judge runs support symmetric scoring and case export for qualitative review, and safety probes baked in.
* Two eval modes:Full (34k) and Credible (Balanced‑400 + Long‑100) for representative reporting.
* Results rollup: scripts.make_results creates a scannable reports/RESULTS.md.
* Portfolio ZIP: scripts.portfolio_export bundles configs, reports, and data samples for easy sharing.

## Developer Experience

* Config‑first: configs/base.yaml presets control sequence length, steps, LoRA, and preference settings.
* Make targets: Clean verbs for common tasks.
* Tests: Config smoke + inference test
