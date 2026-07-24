# Phase 6.1 — Correlation Forecasting with Physiological Features

**Task:** Given 2 hours of 7 waveform-derived correlation features PLUS 38 summarized physiological statistics at 2.5-min stride, forecast the next 30 minutes of correlations (point predictions, Huber loss in Fisher z-space).

## What's New (vs Phase 6)
- **+38 physiological input features** from X_stats.npy (19 mean + 19 std per window)
- Input width: 8 → 46 channels per time step
- Tests whether underlying physiological state (HR, BP, perfusion, etc.) helps predict correlation dynamics
- Same target, same patient split, same architecture hyperparameters — only input width changes

## Motivation
- Phase 6 showed correlations are moderately forecastable from their own history (r ≈ 0.5–0.7)
- Correlations are *derived* from the underlying physiological signals
- Adding the source physiology may provide causal context that captures state changes earlier
- If physio features improve correlation forecasting, it validates that correlations encode physiological coupling dynamics that are partially explainable by raw vitals

## Data

**Source:** `/gpfs/data/eh3828lab/mimic_derived_data/data_m3_120s_prediction/`
- `corr_features_focused.npy` — (564596, 7) correlation values, zero NaN
- `X_stats.npy` — (564596, 19, 109) physiological features × 109 sub-windows
- `patient_ids.npy`, `seg_names.npy`, `window_times.npy`, `block_start_times.npy`

**X_stats summarization:** Mean + std over 109 sub-windows → 38 features per data point

**Resolution:** 2.5 minutes (150 seconds) between windows

**Input feature vector (46 channels):**
- [0–6] 7 pairwise correlations (same as Phase 6)
- [7–25] 19 physiological feature means (HR, RR, SBP, DBP, PP, MAP, ABP_area, PLETH_ACDC, PLETH_amp, ECG_Ramp, HRV_RMSSD, HR_range, ShockIdx, PPV, PVI, PTT, dPdt_max, ABP_tau, RESP_amp)
- [26–44] 19 physiological feature stds
- [45] Time position

**Window parameters:**
| Parameter | Value |
|-----------|-------|
| Resolution | 2.5 min (150s) |
| Input window | 48 steps (2 hours) |
| Forecast window | 12 steps (30 min) |
| Total window | 60 steps (2.5 hours) |
| Stride | 12 steps (30 min) |
| Patient split | seed=42, 70/15/15 |

**Normalization:**
- Correlations: clip to ±0.9999 → Fisher z-transform (arctanh) → z-score
- Physio stats: z-score normalize → NaN imputed with 0 (population mean)

## Architecture

| | TFT | iTransformer |
|---|-----|--------------|
| Input | (48, 46) = 7 corr + 38 physio + time | Same |
| Output | (12, 7) = 7 correlations | Same |
| Params | 18.6M | 1.6M |
| Key config | state=240, LSTM×2, heads=2 | d_model=256, layers=3, heads=4 |
| LR | 1e-3 | 1e-4 (cosine anneal) |
| Loss | Huber (δ=1.0) | Huber (δ=1.0) |
| Early stop | patience=20 | patience=20 |

## Results

Both models trained with early stopping (patience=20).

### Overall Performance

| Model | MAE | RMSE | Best Val Loss | Best Epoch | Params |
|-------|-----|------|---------------|------------|--------|
| **iTransformer (Phase 6.1)** | **0.273** | **0.384** | 0.269 | 12 | 1.6M |
| TFT (Phase 6.1) | 0.279 | 0.389 | 0.274 | 11 | 18.6M |

### Comparison to Phase 6 (correlation-only input)

| Model | Phase 6 MAE | Phase 6.1 MAE | Δ MAE | Phase 6 RMSE | Phase 6.1 RMSE | Δ RMSE |
|-------|-------------|---------------|-------|--------------|----------------|--------|
| iTransformer | 0.274 | **0.273** | −0.001 | 0.385 | **0.384** | −0.002 |
| TFT | 0.277 | 0.279 | +0.002 | 0.388 | 0.389 | +0.001 |

### Per-Correlation Results (Phase 6.1)

| Correlation | iTransformer MAE | TFT MAE | iTransformer r | TFT r |
|-------------|-----------------|---------|---------------|-------|
| PLETH_ACDC × PLETH_amp | 0.062 | 0.061 | 0.633 | 0.652 |
| ABP_area × ABP_tau | 0.311 | 0.328 | **0.698** | 0.677 |
| ABP_area × ShockIdx | 0.247 | 0.255 | **0.694** | 0.685 |
| PLETH_amp × ShockIdx | 0.316 | 0.323 | **0.515** | 0.499 |
| PLETH_ACDC × ShockIdx | 0.320 | 0.327 | **0.544** | 0.531 |
| ShockIdx × ABP_tau | 0.332 | 0.337 | **0.587** | 0.576 |
| PLETH_ACDC × ABP_tau | 0.320 | 0.323 | **0.511** | 0.508 |

### Key Observations
- **Minimal improvement from adding physio features** — the additional 38 input channels provide negligible benefit (≤0.2% change in overall MAE)
- iTransformer still slightly outperforms TFT with 11.6× fewer parameters
- TFT slightly *worsened* with the wider input (18.6M params vs 7.0M in phase6) — likely overfitting to the high-dimensional input
- Correlations appear to already capture the relevant information from the underlying physiology — the raw physio stats are largely redundant for this forecasting task
- Both models converge at similar speed (epoch 11–12) as Phase 6 (epoch 10)

## Quick Start

```bash
# 1. Prepare data
cd /gpfs/home/dk5565/forecasting/phase61
sbatch prepare_data.sh

# 2. Train models (with dependency on data prep job)
cd tft && sbatch --dependency=afterok:<data_job_id> train.sh
cd ../iTransformer && sbatch --dependency=afterok:<data_job_id> train.sh
```

### Job History
| Job ID | Task | Status |
|--------|------|--------|
| 25795818 | Data preparation | ✓ Completed |
| 25795824 | TFT train + test | ✓ Completed |
| 25795825 | iTransformer train + test | ✓ Completed |

## Directory Layout
```
phase61/
├── prepare_data.py           # Data preparation (Fisher z + physio z-score + sliding windows)
├── prepare_data.sh           # SLURM: data prep (CPU, 64GB)
├── phase61_data/processed/   # Tensors + norm params + split info
├── tft/                      # TFT model
│   ├── model.py, preprocess.py, train.py, test.py
│   └── train.sh
├── iTransformer/             # iTransformer model
│   ├── model.py, preprocess.py, train.py, test.py
│   └── train.sh
├── CLAUDE.md, README.md, .gitignore
```
