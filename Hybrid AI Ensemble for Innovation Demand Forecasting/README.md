# Hybrid AI Ensemble for Innovation Demand Forecasting

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-grade forecasting system designed to predict demand for **new product innovations** where historical data is nonexistent (Zero-Shot).

This system outperforms traditional methods by 68% (Relative Interval Width) using a **Horizon-Aware Hybrid Ensemble** that dynamically weights three distinct forecasting paradigms:
1.  **Analog Forecasting:** Domain-specific look-alike modeling (Nearest Neighbors).
2.  **Google TimesFM:** A decoder-only foundation model trained on 100B+ time points.
3.  **Amazon Chronos-Bolt:** A T5-based foundation model treating time series as language.

Precision is guaranteed via **V7 Context-Aware Calibration** (Conditional Conformal Prediction), achieving **89% Coverage** with tight, actionable bounds.

---

## 🚀 Key Features

*   **Meta-Learning (Stacking):** An XGBoost meta-learner trains on backtest residuals to optimally combine Base Models based on forecast horizon, volatility, and trend.
*   **Zero-Shot Inference:** Generates forecasts for completely new SKUs using only 4 weeks of early-read data.
*   **Context-Aware Uncertainty:** Intervals expand/contract dynamically based on "signal quality" (not just historical error), maintaining 89% coverage.
*   **Production Ready:** Includes automated data sanitization, safety scans, and PDF reporting pipelines.

---

## 📊 Performance Benchmarks

Verified on 5,500+ historical launch simulations (Leave-One-Brand-Out) on the provided dataset:

| Metric | Performance | vs. Baseline (V6) |
| :--- | :--- | :--- |
| **WMAPE** | **16.39%** | Best-in-Class |
| **Coverage (80% PI)** | **88.93%** | +8.7% Improvement |
| **Relative Width** | **67.6%** | **47% Narrower** |

---

## 🛠️ Installation

```bash
# Install dependencies
pip install -r requirements.txt
pip install -e .
```

---

## ⚡ Production Workflows

The following workflows drive the core business logic. All inputs and outputs are standardized.

### 1. Run a Production Forecast
Generate a forward-looking forecast for a new product innovation using early-read sales data.

*   **Command:**
    ```bash
    python scripts/run_forecast.py
    ```
*   **Configuration:** Uses Horizon-Aware Hybrid Ensemble + V7 Context-Aware Calibration.
*   **Input:** `Dataset/New_Innovations.csv` (Standardized template).
*   **Output:** `outputs/production_forecast_BRAND_001.csv` (Weekly calibrated forecast).
*   **Optional Flags:**
    *   `--lto`: Force analog-heavy weights for Limited Time Offers (e.g., Holiday items).
    *   `--end_date YYYY-MM-DD`: Hard stop for planned delists.

### 2. Generate Case Study & Readiness Report
Validate model performance on a specific historical launch to build stakeholder confidence.

**Step 1: Run Single Case Study**
Generates a detailed validation plot for a specific Brand x Market combination.

*   **Command:**
    ```bash
    python Case_Study.py \
      --channel "MARKET_001" \
      --manufacturer "MFR_001" \
      --category "CAT_001" \
      --trademark "TM_001" \
      --brand "BRAND_001" \
      --metric "dollars"
    ```
    *(Note: Repeat with `--metric units` and `--metric eq` for volume views)*

**Step 2: Generate Executive PDF**
Compiles all recent results into a "Production Readiness Report".

*   **Command:**
    ```bash
    python scripts/generate_production_report.py
    ```
*   **Output:** `outputs/Production Readiness Report/Production_Readiness_Report.pdf`

### 3. Retrain / Update Model
Refresh the system when new historical data arrives. This runs the full "From Scratch" pipeline:
1.  **LOBO Backtest:** Generates training examples from history.
2.  **Meta-Learner Training:** Retrains XGBoost to weight ensemble components.
3.  **Calibration Training:** Retrains V7 Conditional Conformal Predictor.

*   **Command:**
    ```bash
    python scripts/run_full_comparison_scratch.py
    ```
*   **Input:** `Dataset/Historical_Data.csv`
*   **Output:** `outputs/hybrid_production_results.csv` (Validation log)

### 4. Generate System Architecture Documentation
Produce technical diagrams and system specifications for engineering audit.

*   **Command:**
    ```bash
    python scripts/generate_architecture_report.py
    ```
*   **Output:** `outputs/Production Readiness Report/System_Architecture_Report.pdf`

---

## 📂 Repository Structure

```text
├── Case_Study.py                 # Individual launch validation tool
├── Dataset/
│   ├── Historical_Data.csv       # Anonymized historical sales (Training)
│   └── New_Innovations.csv       # New product early reads (Inference)
├── models/
│   ├── meta_learner/             # XGBoost Stacking Model
│   └── calibration_v7/           # Conformal Prediction Model
├── outputs/                      # Forecasts, PDFs, and Logs
├── scripts/
│   ├── run_forecast.py           # PRODUCTION ENTRY POINT
│   ├── run_full_comparison...    # Retraining Pipeline
│   └── ...
└── src/
    └── retail_forecast/          # Core Library (Ensemble, IO, Utils)
```

<br />
<p align="center">
    <img src="/Archive/Resources/Hybrid%20AI%20Ensemble.png" width="1920" height="1080">
</p>
