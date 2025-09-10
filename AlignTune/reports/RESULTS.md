# Results Summary

## Ablation (Balanced‑Prompt Subset)

| Model | N | Exact Match | ROUGE-L | Toxic Flags |
|---|---:|---:|---:|---:|
| SFT | 400 | 0.005 | 0.207 | 11 |
| SimPO | 400 | 0.002 | 0.195 | 10 |
| DPO | 400 | 0.001 | 0.185 | 12 |

## Ablation (Long‑Prompt Subset)

| Model | N | Exact Match | ROUGE-L | Toxic Flags |
|---|---:|---:|---:|---:|
| SFT (long) | 100 | 0.000 | 0.215 | 6 |
| SimPO (long) | 100 | 0.000 | 0.235 | 7 |
| DPO (long) | 100 | 0.000 | 0.240 | 7 |

## Judge Win-Rate

Llama-3.1-8B-Instruct-lora vs Llama-3.1-8B-Instruct-lora-dpo: 35.1% win for Llama-3.1-8B-Instruct-lora (95% CI: 32.0–38.2%).

Llama-3.1-8B-Instruct-lora-dpo vs Llama-3.1-8B-Instruct-lora: 64.9% win for Llama-3.1-8B-Instruct-lora-dpo (95% CI: 61.8–68.0%).

Ties: 88/1000.

## Safety Quick-Check

| Model | Attacks | Harmful Flags | Refusals | Attack Success Rate |
|---|---:|---:|---:|---:|
| Llama-3.1-8B-Instruct-lora | 4 | 1 | 3 | 0.25 |
| Llama-3.1-8B-Instruct-lora-dpo | 4 | 1 | 4 | 0.00 |
