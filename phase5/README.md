# Phase 5 — Vital Sign Forecasting with Waveform Correlation Features

**Task:** Given 6 hours of 4 vital signs + 7 waveform-derived correlation features at 5-min stride, forecast the next 2 hours of vital signs with uncertainty quantification (10th, 50th, 90th percentile prediction intervals).

## What's New (vs Phase 4.2)
- **+7 correlation features** from waveform analysis as additional input
- **5-min stride** with 20-min context windows (vs 15-min bins in Phase 4.2) — each data point summarizes 20 min of high-res waveform analysis, spaced 5 min apart
- **6h→2h forecast** (vs 18.75h→6.25h) — shorter but more clinically actionable
- **Vitals-only baseline** — same models with vitals-only input (no correlations) to isolate correlation feature contribution

## Results (44/50 extraction parts, 2,486 test samples)

### Vitals+Waveform (12-dim input: 7 correlations + 4 vitals + time)

| Model | Overall MAE | Overall RMSE |
|-------|-------------|--------------|
| iTransformer | **3.08** | **5.46** |
| TFT | 3.23 | 5.63 |

### iTransformer Per-Vital (Vitals+Waveform)

| Vital | MAE | Correlation | Calibration |
|-------|-----|-------------|-------------|
| ABPMean | 5.35 mmHg | 0.888 | 78.4% |
| PULSE | 3.79 bpm | 0.915 | 82.0% |
| SpO2 | 0.88% | 0.799 | 79.2% |
| RESP | 2.30 br/min | 0.807 | 75.8% |

### Vitals+Waveform vs Vitals-Only

**iTransformer:**

| Vital | V+W MAE | Vitals-Only MAE | Δ MAE | V+W Corr | Vitals-Only Corr | Δ Corr |
|-------|----------|--------------|-------|-----------|---------------|--------|
| ABPMean | **5.35** | 5.42 | +1.3% | **0.888** | 0.886 | -0.2% |
| PULSE | **3.79** | 3.85 | +1.6% | **0.915** | 0.913 | -0.2% |
| SpO2 | **0.88** | 0.89 | +1.1% | 0.799 | **0.800** | +0.1% |
| RESP | **2.30** | 2.31 | +0.4% | **0.807** | 0.807 | 0% |
| **Overall** | **3.08** | 3.12 | +1.3% | — | — | — |

| Vital | V+W Calib | Vitals-Only Calib |
|-------|------------|----------------|
| ABPMean | 78.4% | **77.9%** |
| PULSE | 82.0% | **81.9%** |
| SpO2 | 79.2% | **78.5%** |
| RESP | 75.8% | **76.2%** |

**TFT:**

| Vital | V+W MAE | Vitals-Only MAE | Δ MAE | V+W Corr | Vitals-Only Corr | Δ Corr |
|-------|----------|--------------|-------|-----------|---------------|--------|
| ABPMean | 5.63 | **5.59** | -0.7% | 0.879 | **0.884** | +0.6% |
| PULSE | **4.00** | 4.20 | +5.0% | **0.908** | 0.906 | -0.2% |
| SpO2 | **0.93** | 0.99 | +6.5% | **0.788** | 0.784 | -0.5% |
| RESP | 2.36 | **2.34** | -0.9% | 0.802 | **0.806** | +0.5% |
| **Overall** | **3.23** | 3.28 | +1.5% | — | — | — |

| Vital | V+W Calib | Vitals-Only Calib |
|-------|------------|----------------|
| ABPMean | **80.0%** | 78.7% |
| PULSE | **87.2%** | 83.0% |
| SpO2 | 79.8% | **84.0%** |
| RESP | 77.1% | **75.5%** |

**Summary:**

| Model | V+W (12-dim) MAE | Vitals-Only (5-dim) MAE | Improvement from correlations |
|-------|--------------------|-----------------------|-------------------------------|
| iTransformer | **3.08** | 3.12 | 1.3% |
| TFT | **3.23** | 3.28 | 1.5% |

**Key finding:** Waveform correlation features provide only ~1–1.5% MAE improvement. The vast majority of predictive power comes from vital sign history alone at this 6h→2h horizon. The correlations help most for PULSE and SpO2 in the TFT model (+5–6.5%), but are negligible for the iTransformer.

### vs Phase 4.2

| Metric | Phase 4.2 (iTransformer) | Phase 5 (iTransformer) | Δ |
|--------|--------------------------|------------------------|---|
| Overall MAE | 4.13 | **3.08** | -25% |
| Correlation (mean) | 0.792 | **0.852** | +7.6% |
| Calibration (mean) | 78.8% | **78.9%** | +0.1pp |

**⚠️ Caveat:** Phase 5 uses a shorter forecast horizon (2h vs 6.25h) and finer stride (5-min vs 15-min). Most of the improvement is attributable to the easier forecasting task, not the correlation features (as confirmed by the ablation).

## Data

**Raw source:** MIMIC-III Waveform Database Matched Subset (`/gpfs/data/eh3828lab/datasets/mimic3_waveforms_matched`)

**Extraction pipeline** (`data_extraction/`): Reads 125 Hz waveforms (II, PLETH, RESP, ABP) → computes 19 beat-by-beat features per 30s sub-window → Pearson correlations across ~118 sub-windows → combines with numerics-derived vital signs → outputs 11-dim feature vectors at 5-min stride.

**Feature vector (11-dim):**
- [0–6] Pearson correlations between waveform-derived feature pairs (computed across ~118 sub-windows within each 20-min window)
- [7–10] Vital signs: ABPMean, PULSE, SpO2, RESP (median over ~20 numerics samples within each 20-min window)

**Processing:** Group by patient/segment → sort by time → split at temporal gaps → sliding windows (96 steps = 72 input + 24 output) → discard windows with any NaN → split by patient (70/15/15) → normalize → save tensors

**Current dataset (44 parts):** 1,225,024 raw windows from 1,808 patients → Train 8,599 / Val 1,832 / Test 2,486 forecast windows

## Quick Start

```bash
# Full pipeline from scratch:
cd data_extraction && sbatch slurm_extract.sh    # Extract features (50 parallel jobs)
sbatch slurm_merge.sh                            # Merge outputs
cd ../tft && sbatch prepare_data.sh              # Prepare vitals+waveform tensors
cd ../ablation && sbatch prepare_data.sh         # Prepare vitals-only tensors
cd ../tft && sbatch train.sh                     # Train vitals+waveform TFT
cd ../iTransformer && sbatch train.sh            # Train vitals+waveform iTransformer
cd ../ablation/iTransformer && sbatch train.sh   # Train vitals-only iTransformer
cd ../ablation/tft && sbatch train.sh            # Train vitals-only TFT
```

## Architecture

**Single model** predicts all 4 vital signs simultaneously.

| | Vitals+Waveform | Vitals-Only |
|---|---|---|
| Input | (72, 12) = 7 corr + 4 vitals + time | (72, 5) = 4 vitals + time |
| Output | (24, 4, 3) = 4 vitals × 3 quantiles | Same |
| Same windows | ✓ (NaN filter on all 11 features) | ✓ |
| Same split | ✓ (seed=42, 70/15/15) | ✓ |

| | TFT | iTransformer |
|---|-----|--------------|
| Key params | state=240, LSTM×2, heads=2 | d_model=256, layers=3, heads=4 |
| LR | 1e-3 | 1e-4 |
| Loss | Quantile (pinball) [0.1, 0.5, 0.9] | Same |

## Directory Layout
```
phase5/
├── data_extraction/          # Feature extraction from raw waveforms
│   ├── extract_features.py   # Main pipeline
│   ├── merge_outputs.py      # Combine parallel outputs
│   ├── config/pipeline_config.yaml
│   ├── slurm_extract.sh, slurm_merge.sh
│   └── output/               # part_XXX/ + merged/
├── tft/                      # Vitals+Waveform model (12-dim input)
│   ├── model.py, preprocess.py, prepare_data.py
│   ├── train.py, test.py, plot_predictions.py, plot_full_forecast.py
│   └── prepare_data.sh, train.sh
├── iTransformer/             # Vitals+Waveform model (12-dim input)
│   ├── model.py, preprocess.py
│   ├── train.py, test.py, plot_predictions.py, plot_full_forecast.py
│   └── train.sh
├── ablation/                 # Vitals-only (5-dim input)
│   ├── prepare_data.py, prepare_data.sh
│   ├── iTransformer/  (model, train, test, train.sh)
│   └── tft/           (model, train, test, train.sh)
├── phase5_data/processed/    # Vitals+Waveform tensors
├── CLAUDE.md, README.md, .gitignore
```
