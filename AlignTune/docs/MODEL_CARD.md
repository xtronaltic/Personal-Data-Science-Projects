# Model Card — Fine-tuned Llama-3.1-8B Instruct (SFT + SimPO/ORPO + DPO polish)

Short summary
- Purpose: a reproducible, config-driven stack for rapid iteration on instruction-following models. The pipeline combines supervised fine-tuning (SFT), a fast preference-shaping stage (SimPO or ORPO), and a conservative DPO polish to deliver measurable gains with limited compute overhead.
- Snapshot artifacts live under `outputs/` (LoRA adapter `.safetensors`, tokenizer and config snapshots); canonical defaults are in `configs/base.yaml`.

Top engineering highlights
- Config-first reproducibility: all runtime and training knobs live in `configs/base.yaml` and the Makefile presets (fast | balanced | thorough) so experiments are predictable and easy to reproduce.
- Low-cost adaptation with QLoRA + LoRA:
  - Base model loaded in 4-bit using `bitsandbytes` and adapted via LoRA adapters (targets: attention and MLP gating projections). This enables experiments on modest infra while keeping the backbone frozen.
  - LoRA presets balance expressivity vs VRAM (typical r=8 for fast/balanced, r=16 for thorough; alpha=32, dropout=0.0 by default).
- Two-stage preference design:
  - SimPO/ORPO provides fast, higher-variance preference shaping (useful for rapid iteration and mining informative pairs).
  - DPO is applied as a conservative polish to stabilize preference signals and keep instruction fidelity.
  - The split-stage design drives quick improvements while limiting disruptive model drift.
- Robust training ergonomics:
  - Autosnapshots, emergency save, gradient checkpointing and optional 8-bit optimizer support make runs resilient to preemption and memory pressure.
  - Group-by-length batching and left-padding optimize throughput for decoder-only models.

Reproducibility contract (inputs / outputs / error modes)
- Inputs: `configs/base.yaml` + `data/pack/sft.jsonl` (SFT: 34,000 examples) + `data/pack/dpo.jsonl` (DPO: 30,000 pairs). Preset selection controls max steps / seq lengths.
- Outputs: LoRA adapter artifacts, tokenizer snapshots, training checkpoints under `outputs/` and aggregated reports under `reports/`.
- Errors / mitigations: preset switching (fast/balanced/thorough) trades memory/compute; autosnapshots/emergency save reduce risk from OOM and preemptions.

Training (high-level)
- SFT: QLoRA + LoRA adapters, bf16 when available. Typical per-device batch size = 4, gradient_accumulation_steps = 4. Default max_steps: fast=400, balanced=800, thorough=1500.
- Preference tuning: SimPO/ORPO stage for fast shaping, then DPO polish (configurable in `configs/base.yaml`). Optional `precompute_ref_log_probs` for deterministic workflows.
- Seed: 42 (used across data sampling and the judge flows for reproducibility).

Evaluation & reporting
- Two canonical eval modes:
  - Credible eval: balanced small eval (Balanced-400) + long-context stress set (Long-100).
  - Full eval: ablation across the full SFT pack (34k) for final reporting.
- Judge: Llama-3.2-1B-Instruct (greedy, temp=0.0) used in batched comparisons; scripts live in `eval/` and `scripts/` and results are rolled up into `reports/RESULTS.md`.

Safety & limitations
- The repo applies a lightweight toxicity skim and a small demo blocklist but does not claim production-level safety. Downstream deployments must add stronger safety layers and PII checks.

How to reproduce / run pointers
- Canonical config: `configs/base.yaml` (preset-driven settings).
- Primary scripts and flows: `scripts/train_sft.py`, `scripts/train_pref.py`, `scripts/mine_hard_pairs.py`, and `scripts/make_results.py`.
- Demo and serving: `app/demo.py` (Gradio) and `api/server.py` (FastAPI).
