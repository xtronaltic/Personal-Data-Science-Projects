# AI/ML Projects

[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]

<br />
<p align="center">
    <img src="Archive/Resources/NLP.jpeg" width="1920" height="1080">
</p>

## Projects

* [AlignTune — Efficient LLM Alignment: SimPO-primed DPO (QLoRA-8B); seeded ablations; symmetric judge win-rate; safety probes; FastAPI/Gradio; FAISS-RAG](./AlignTune)
  - Engineered SFT → SimPO → hard-pair mining → short DPO polish pipeline; +9–11% ROUGE-L on long-context, parity on balanced eval, 53.5% DPO judge win rate (95% CI: 50.90%–56.05%), safety probe 25% → 0% attack success. 
  - QLoRA 4-bit + LoRA (r=8/16, α=32), 8-bit optimizer, grad checkpointing, SDPA, length-grouped left-padded batches, deterministic seeds, and autosnapshots—stable training on a single 16 GB GPU without OOM.
  - Deterministic data packs, streaming ingestion, near‑dup removal, toxicity skim, per‑source caps, record‑level provenance; length‑bucketed ROUGE‑L, LLM‑judge win‑rate (greedy, symmetric, anti‑position, rubric+JSON), toxicity/1k tokens auto‑rolled into RESULTS.md; FastAPI inference server + Gradio demo with versioned LoRA adapters/tokenizer snapshots for drop‑in deployment.
  - RAG (FAISS-CPU; deterministic chunking; top-k + token-aware packing) adds cited, doc-grounded context at serve-time with no extra VRAM.
* [Uncertainty-Calibrated Residual PatchTST Transformer Forecasting: Dilated Conv Stem → Transformer Encoder → Adaptive Quantile Head with Horizon-Wise Temperature Scaling](./Transformer)
  - Post-calibration WMAPE 0.276% (≈ −48% vs baseline, −53% vs uncalibrated), Central-80% = 78.85% at τ=1.282 (near target), with tight bands (~1.43% of P50) and stable rolling results (K=5, WMAPE 0.276–0.313%). 
  - Residualization on lag-52 seasonal baseline → dilated conv stem + Transformer encoder + patch tokenization → adaptive quantile head trained with weighted pinball + aux P50 MSE + monotonicity penalty → horizon-wise linear calibration + temperature scaling.
  - Deterministic seeds, mixed-precision, train-only scalers, order enforcement for P10/P50/P90, rolling backtests with RMSE/MAE/WMAPE and band diagnostics, plus exportable artifacts for BI integration and reporting. 
* [Local–Global Hybrid Forecasting: CNN → BiLSTM → Multi-Head Attention](./CNN%20BiLSTM%20Attention)
  - Beat seasonal-naïve with a calibrated hybrid (CNN→BiLSTM→Attention): WMAPE −28.12% (0.379 vs 0.527) and RMSE −23.62% (721,562 vs 944,726) after bias fixes & α-blend; auto “Champion” = DL_linear_cal; R² 0.94 vs 0.89 baseline.
  - 486,169 params; ASHA search (150 epochs, bracket 4) → CNN 64×5, BiLSTM 100, MHA 2 heads / key_dim 32, dropout 0.2; deterministic skill tables for WMAPE/RMSE/MAE/MASE/R².
  - Trained on 113 series with 156-step windows (time-aware splits); WMAPE improves 0.608 → 0.454 → 0.379 from raw → const-cal → linear-cal, then blended/selected on validation-min WMAPE.

## License

Distributed under the MIT License. See `LICENSE` for more information.

[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/xtronaltic/Personal-Data-Science-Projects/blob/main/LICENSE
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/gaitianpeng
