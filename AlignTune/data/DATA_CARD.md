# Data Card — AlignTune Packs (SFT & Preference)

Short summary
- Purpose: curated packs designed to produce robust instruction-following and preference signals while keeping runs reproducible and auditable. The balanced SFT pack targets 34,000 examples; the preference (UltraFeedback-derived) pack targets 30,000 pairwise entries.

Top engineering highlights
- Deterministic, capped packer:
  - `data/build_sft_pack.py` and `data/build_dpo_ufb_only.py` perform streaming ingestion with a fixed RNG seed (42), per-source caps, and preset-driven behavior (fast | balanced | thorough). Balanced defaults produce SFT=34k and DPO=30k for stable experiments.
- Provenance and per-example metadata:
  - Each output record carries provenance fields (`source`, `source_id`, and optional original metadata) so downstream audits can trace model outputs back to their origin.
- Streaming denoising and filters:
  - Toxicity skim using a conservative blocklist and lightweight heuristics to remove explicit unsafe content early in the pipeline.
  - Near-duplicate removal via a rolling Jaccard similarity scan scoped to recent items; tuned per-preset to trade coverage vs uniqueness.
- Schema harmonization and normalization:
  - SFT items normalized into a standard chat template: {"instruction", "input", "output", "source", "source_id"}.
  - Preference items normalized to {"prompt", "chosen", "rejected", "source", "source_id"} for stable trainer inputs.

Processing & pack generation
- SFT pack (`data/pack/sft.jsonl`): built from Dolly-15k, Alpaca-cleaned and optional extra sources (SlimOrca, etc.). The packer honors per-source caps and records per-source counts in process logs and per-example provenance fields.
- Preference pack (`data/pack/dpo.jsonl`): UltraFeedback-derived pairs are denoised, normalized and capped; trimming and prompt-length caps are applied to ensure stable training behavior across batch sizes.
- Credible eval splits:
  - `scripts/split_jsonl_eval.py`: length-balanced small eval (Balanced-400).
  - `scripts/build_long_eval.py`: selects top-N longest items for long-context stress tests (Long-100).

Schema examples
- SFT (one-line JSON example):
  - {"instruction": "Explain LoRA in 3 sentences.", "input": "", "output": "LoRA is ...", "source": "Dolly-15k", "source_id": "dolly_1234"}
- DPO (one-line JSON example):
  - {"prompt": "Translate to French: 'Hello'", "chosen": "Bonjour", "rejected": "Salut", "source": "UltraFeedback", "source_id": "ufb_5678"}

Operational engineering choices
- Streaming + caps: streaming ingestion avoids memory spikes and makes builds practical on a workstation or CI; caps ensure predictable runtime and reproducible experiment sizes.
- Preset-driven tuning: fast/balanced/thorough presets provide a controlled trade-off between speed and coverage and reduce accidental divergent experiments.
- Minimal in-place modification: normalization and metadata addition are preferred over destructive edits so original content can be audited or reconstituted when needed.

Reproducibility
- Inputs: raw source datasets (Dolly, Alpaca, UltraFeedback or local equivalents), `data/build_sft_pack.py`, `data/build_dpo_ufb_only.py`, `configs/base.yaml` (preset controls), and seed=42.
- Outputs: `data/pack/sft.jsonl` (34,000 examples) and `data/pack/dpo.jsonl` (30,000 pairs). Per-source counts are available from packer logs and per-example provenance fields in the emitted JSONL.

How this data design helps model engineering
- Predictable experiments: fixed caps and presets make every run comparable.
- Rapid iteration: small, credible eval splits and long-context subsets enable quick hypothesis testing without running the full-scale pipeline.
- Auditability: provenance fields per example enable tracing and downstream bias/error analysis.

Provenance, licensing and redistribution notes
- The packer preserves source attribution per item so license adherence can be enforced. The repository itself is MIT, but original dataset licenses may restrict redistribution or require attribution—inspect source metadata when reproducing or publishing packs.