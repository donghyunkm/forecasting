# Phase 6 — Correlation-Only Forecasting

**Task:** Given 2 hours of 7 waveform-derived correlation features at 2.5-min stride, forecast the next 30 minutes of correlations (point predictions, Huber loss in Fisher z-space).

## What's New (vs Phase 5)
- **New data source** — uses `data_m3_120s_prediction` (2.5-min resolution, 564K windows, 0% NaN)
- **Higher temporal resolution** — 2.5 min (150s) vs Phase 5's 5 min (300s)
- **Fisher z-transform** — correlations are transformed via arctanh before normalization (handles bounded [-1,1] range properly)
- **Huber loss** — robust to outliers in Fisher z-space
- **Point predictions** — single value output, no uncertainty quantification
- **Much more data** — 30K forecast windows from 4,092 continuous segments

## Motivation
- Phase 5 showed correlations provide only ~1.3% benefit for vital sign forecasting
- This phase asks: **are correlations themselves forecastable?**
- Higher resolution (2.5 min) captures faster dynamics than the 5-min Phase 5 data
- If correlations can be accurately predicted, they could serve as early warning signals for physiological state changes

## Data

**Source:** `/gpfs/data/eh3828lab/mimic_derived_data/data_m3_120s_prediction/`
- `corr_features_focused.npy` — (564596, 7) correlation values, zero NaN
- `patient_ids.npy`, `seg_names.npy`, `window_times.npy`, `block_start_times.npy`

**Resolution:** 2.5 minutes (150 seconds) between windows

**Feature vector (7 correlations):**
- [0] PLETH_ACDC × PLETH_amp (PPG coupling)
- [1] ABP_area × ABP_tau (stroke volume × resistance)
- [2] ABP_area × ShockIdx (BP × shock)
- [3] PLETH_amp × ShockIdx (perfusion × shock)
- [4] PLETH_ACDC × ShockIdx (perfusion index × shock)
- [5] ShockIdx × ABP_tau (shock × resistance)
- [6] PLETH_ACDC × ABP_tau (perfusion × resistance)

**Window parameters:**
| Parameter | Value |
|-----------|-------|
| Resolution | 2.5 min (150s) |
| Input window | 48 steps (2 hours) |
| Forecast window | 12 steps (30 min) |
| Total window | 60 steps (2.5 hours) |
| Stride | 12 steps (30 min) |
| Patient split | seed=42, 70/15/15 |

**Data scale:**
| Split | Patients | Windows |
|-------|----------|---------|
| Train | 1,010 | 20,879 |
| Val | 216 | 4,444 |
| Test | 217 | 4,763 |
| **Total** | **1,443** | **30,086** |

**Normalization:** clip to ±0.9999 → Fisher z-transform (arctanh) → z-score normalize using training set statistics.

## Architecture

| | TFT | iTransformer |
|---|-----|--------------|
| Input | (48, 8) = 7 corr + time | (48, 8) = 7 corr + time |
| Output | (12, 7) = 7 correlations | Same |
| Params | 7.0M | 1.6M |
| Key config | state=240, LSTM×2, heads=2 | d_model=256, layers=3, heads=4 |
| LR | 1e-3 | 1e-4 (cosine anneal) |
| Loss | Huber (δ=1.0) | Huber (δ=1.0) |
| Early stop | patience=20 | patience=20 |

## Results

Both models trained for 30 epochs (early stopped at epoch 10, patience=20).

### Overall Performance

| Model | MAE | RMSE | Best Val Loss | Params |
|-------|-----|------|---------------|--------|
| **iTransformer** | **0.274** | **0.385** | 0.270 | 1.6M |
| TFT | 0.277 | 0.388 | 0.275 | 7.0M |

### Per-Correlation Results

| Correlation | iTransformer MAE | TFT MAE | iTransformer r | TFT r |
|-------------|-----------------|---------|---------------|-------|
| PLETH_ACDC × PLETH_amp | 0.064 | 0.063 | 0.625 | 0.645 |
| ABP_area × ABP_tau | 0.313 | 0.317 | **0.694** | 0.688 |
| ABP_area × ShockIdx | 0.250 | 0.251 | **0.684** | 0.681 |
| PLETH_amp × ShockIdx | 0.317 | 0.322 | **0.509** | 0.500 |
| PLETH_ACDC × ShockIdx | 0.321 | 0.325 | **0.539** | 0.533 |
| ShockIdx × ABP_tau | 0.333 | 0.336 | **0.587** | 0.581 |
| PLETH_ACDC × ABP_tau | 0.320 | 0.324 | **0.510** | 0.507 |

### Key Observations
- Both models perform similarly, with iTransformer slightly better overall despite having 4.4× fewer parameters
- `PLETH_ACDC × PLETH_amp` is the most predictable (MAE 0.06, highly autocorrelated near +1)
- The other 6 correlations have MAE ~0.25–0.34, with Pearson r between 0.5–0.7
- Correlations involving ShockIdx are the hardest to predict (r ≈ 0.5)
- Early stopping at epoch 10 suggests the models converge quickly on this dataset

## Outputs

After training + testing, each model produces in `outputs/`:
- `training_curves.png` — train/val loss over epochs
- `sample_forecasts.png` — full 2h input + 30min forecast trajectories
- `error_by_horizon.png` — MAE at each forecast step (2.5–30 min)
- `scatter_pred_vs_actual.png` — predicted vs actual per correlation
- `bland_altman.png` — Bland-Altman agreement plots per correlation
- `metrics_bar.png` — per-correlation MAE and Pearson r
- `test_metrics.json` — all metrics in JSON
- `test_predictions.npy`, `test_targets.npy` — raw arrays

## Quick Start

```bash
# 1. Prepare data (already done — regenerate if source changes)
sbatch prepare_data.sh

# 2. Train models (runs train + test + plots)
cd tft && sbatch train.sh
cd ../iTransformer && sbatch train.sh
```

## Directory Layout
```
phase6/
├── prepare_data.py           # Data preparation (Fisher z + sliding windows)
├── prepare_data.sh           # SLURM: data prep (CPU)
├── phase6_data/processed/    # Tensors + norm params + split info
├── tft/                      # TFT model
│   ├── model.py, preprocess.py, train.py, test.py
│   └── train.sh
├── iTransformer/             # iTransformer model
│   ├── model.py, preprocess.py, train.py, test.py
│   └── train.sh
├── CLAUDE.md, README.md, .gitignore
```
