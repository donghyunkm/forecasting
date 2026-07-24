# Phase 5 — Vital Sign Forecasting with Waveform Correlation Features

**Task:** Given 6 hours of 4 vital signs + 7 waveform-derived correlation features at 5-min stride, forecast the next 2 hours of vital signs with uncertainty quantification (10th, 50th, 90th percentile prediction intervals).

## What's New (vs Phase 4.2)
- **+7 correlation features** from waveform analysis as additional input
- **5-min stride** with 20-min context windows (vs 15-min bins in Phase 4.2) — each data point summarizes 20 min of high-res waveform analysis, spaced 5 min apart
- **6h→2h forecast** (vs 18.75h→6.25h) — shorter but more clinically actionable
- **Vitals-only baseline** — same models with vitals-only input (no correlations) to isolate correlation feature contribution

## Results (50/50 extraction parts, 2,886 test samples, 2,060 patients)

### Vitals+Waveform (12-dim input: 7 correlations + 4 vitals + time)

| Model | Overall MAE | Overall RMSE |
|-------|-------------|--------------|
| iTransformer | **3.12** | **5.86** |
| TFT | 3.23 | 5.90 |

### iTransformer Per-Vital (Vitals+Waveform)

| Vital | MAE | Correlation | Calibration |
|-------|-----|-------------|-------------|
| ABPMean | 5.12 mmHg | 0.831 | 79.1% |
| PULSE | 4.41 bpm | 0.897 | 80.0% |
| SpO2 | 0.83% | 0.861 | 80.5% |
| RESP | 2.11 br/min | 0.834 | 77.3% |

### TFT Per-Vital (Vitals+Waveform)

| Vital | MAE | Correlation | Calibration |
|-------|-----|-------------|-------------|
| ABPMean | 5.24 mmHg | 0.826 | 83.0% |
| PULSE | 4.62 bpm | 0.896 | 75.9% |
| SpO2 | 0.88% | 0.863 | 80.9% |
| RESP | 2.18 br/min | 0.829 | 76.6% |

### Vitals+Waveform vs Vitals-Only

**iTransformer:**

| Vital | V+W MAE | Vitals-Only MAE | Δ MAE |
|-------|----------|--------------|-------|
| ABPMean | 5.12 | 5.12 | 0% |
| PULSE | 4.41 | 4.41 | 0% |
| SpO2 | 0.83 | 0.84 | +1.2% |
| RESP | 2.11 | 2.10 | -0.5% |
| **Overall** | 3.12 | **3.11** | -0.3% |

**TFT:**

| Vital | V+W MAE | Vitals-Only MAE | Δ MAE |
|-------|----------|--------------|-------|
| ABPMean | **5.24** | 5.30 | +1.1% |
| PULSE | **4.62** | 4.69 | +1.5% |
| SpO2 | 0.88 | **0.86** | -2.3% |
| RESP | 2.18 | **2.17** | -0.5% |
| **Overall** | **3.23** | 3.26 | +0.9% |

**Summary:**

| Model | V+W (12-dim) MAE | Vitals-Only (5-dim) MAE | Δ |
|-------|--------------------|-----------------------|---|
| iTransformer | 3.12 | **3.11** | -0.3% (vitals-only better) |
| TFT | **3.23** | 3.26 | +0.9% |

**Key finding:** With the complete dataset (all 2,060 patients), waveform correlation features provide **no meaningful benefit**. The iTransformer vitals-only model actually slightly outperforms the vitals+waveform variant. TFT shows <1% improvement from correlations. The vast majority of predictive power comes from vital sign history alone at this 6h→2h horizon.

### vs Phase 4.2

| Metric | Phase 4.2 (iTransformer) | Phase 5 (iTransformer) | Δ |
|--------|--------------------------|------------------------|---|
| Overall MAE | 4.13 | **3.12** | -24% |
| Correlation (mean) | 0.792 | **0.856** | +8.1% |
| Calibration (mean) | 78.8% | **79.2%** | +0.4pp |

**⚠️ Caveat:** Phase 5 uses a shorter forecast horizon (2h vs 6.25h) and finer stride (5-min vs 15-min). Most of the improvement is attributable to the easier forecasting task, not the correlation features (as confirmed by the ablation).

## Data

**Raw source:** MIMIC-III Waveform Database Matched Subset (`/gpfs/data/eh3828lab/datasets/mimic3_waveforms_matched`)

**Extraction pipeline** (`data_extraction/`): Reads 125 Hz waveforms (II, PLETH, RESP, ABP) → computes 19 beat-by-beat features per 30s sub-window → Pearson correlations across ~118 sub-windows → combines with numerics-derived vital signs → outputs 11-dim feature vectors at 5-min stride.

**Feature vector (11-dim):**
- [0–6] Pearson correlations between waveform-derived feature pairs (computed across ~118 sub-windows within each 20-min window)
- [7–10] Vital signs: ABPMean, PULSE, SpO2, RESP (median over ~20 numerics samples within each 20-min window)

**Processing:** Group by patient/segment → sort by time → split at temporal gaps → sliding windows (96 steps = 72 input + 24 output) → discard windows with any NaN → split by patient (70/15/15) → normalize → save tensors

**Current dataset (50 parts):** 1,432,407 raw windows from 2,060 patients → Train/Val/Test forecast windows (70/15/15 by patient)

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
