# Results Summary

## Ablation (Blanced‑Prompt Subset)

| Model | N | Exact Match | ROUGE-L | Toxic Flags |
|---|---:|---:|---:|---:|
| SFT | 400 | 0.007 | 0.196 | 9 |
| SimPO | 400 | 0.000 | 0.172 | 10 |
| DPO | 400 | 0.003 | 0.172 | 8 |

Exact Match (ASCII bars):
- SFT:                          0.007
- SimPO:                          0.000
- DPO:                          0.003


## Ablation (Long‑Prompt Subset)

| Model | N | Exact Match | ROUGE-L | Toxic Flags |
|---|---:|---:|---:|---:|
| SFT (long) | 100 | 0.000 | 0.224 | 5 |
| SimPO (long) | 100 | 0.000 | 0.227 | 6 |
| DPO (long) | 100 | 0.000 | 0.234 | 5 |

## Judge Win-Rate

Llama-3.1-8B-Instruct-lora vs Llama-3.1-8B-Instruct-lora-dpo: 48.5% win for Llama-3.1-8B-Instruct-lora (95% CI: 46.0–51.1%), ties: 303/1000.

## Safety Quick-Check

| Model | Attacks | Harmful Flags | Refusals | Attack Success Rate |
|---|---:|---:|---:|---:|
| Llama-3.1-8B-Instruct-lora | 4 | 1 | 3 | 0.25 |
| Llama-3.1-8B-Instruct-lora-dpo | 4 | 1 | 4 | 0.00 |
