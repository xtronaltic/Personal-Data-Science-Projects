# Model Card — AlignTune (LoRA + Preference Tuning)

**Base model:** TinyLlama/TinyLlama-1.1B-Chat-v1.0 (default)  
**Methods:** SFT (LoRA/QLoRA) → DPO / SimPO / ORPO (config‑selectable)  
**Intended Use:** Portfolio demonstration.

## Training Data
Synthetic examples included for portability. 

## Evaluation
- Task metrics: Exact‑Match, ROUGE‑L
- Judge‑based pairwise preference win‑rate (MT‑Bench/AlpacaEval‑style) with a local judge model
- Basic safety screen (toy refusal/harm heuristics). Hooks included for HarmBench/JailbreakBench.

## Risks & Limitations
- Small demo data ⇒ metrics not meaningful. Add real data + larger base models.
- Safety checks here are minimal. Use robust, benchmarked guardrails before deployment.

## License
MIT.
