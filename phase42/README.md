# Phase 4.2 — Vital Sign Forecasting with Complete Windows Only

Same task as Phase 4.1 but **only uses windows with zero missing data** across all 4 signals at all timesteps. No forward-fill, no mask channels. Forces the model to learn actual physiological dynamics.

Uses **chunk-level splits** — each continuous recording segment (gap detection at >30 min between records) is independently assigned to train/val/test, preventing windows from spanning temporal discontinuities.

## Results (chunk-level splits, 3,383 test windows)

| Metric | TFT | iTransformer |
|--------|-----|--------------|
| Overall MAE | 4.42 | **4.13** |
| Overall RMSE | 7.36 | **6.93** |
| Calibration (target 80%) | 72.6% | **78.8%** |
| Best epoch | 12 | 19 |

### Per-Vital MAE
| Signal | TFT | iTransformer |
|--------|-----|--------------|
| mean_bp | 7.75 mmHg | **7.38 mmHg** |
| pulse | 6.19 bpm | **5.66 bpm** |
| spo2 | 1.21 % | **1.15 %** |
| resp_rate | 2.55 /min | **2.32 /min** |

### Per-Vital Calibration (target: 80%)
| Signal | TFT | iTransformer |
|--------|-----|--------------|
| mean_bp | 72.6% | **78.3%** |
| pulse | 71.5% | **78.3%** |
| spo2 | 74.4% | **80.7%** |
| resp_rate | 71.6% | **77.9%** |

### Per-Vital Correlation
| Signal | TFT | iTransformer |
|--------|-----|--------------|
| mean_bp | 0.767 | **0.788** |
| pulse | 0.848 | **0.870** |
| spo2 | 0.676 | **0.708** |
| resp_rate | 0.768 | **0.802** |

## What's Different from Phase 4.1

| Aspect | Phase 4.1 | Phase 4.2 |
|--------|-----------|-----------|
| Missing data | Forward-fill + mask input | Discard (no NaN allowed) |
| Input features | 9 (4 vitals + 4 masks + 1 time) | 5 (4 vitals + 1 time) |
| Split unit | Chunk (continuous segment) | Chunk (continuous segment) |
| Training windows | 174,995 | 28,687 (16.3% kept) |
| Data quality | Mixed real + imputed | 100% real measurements |
| Selection bias | All patients | Patients with full monitoring |

## Quick Start

```bash
cd /gpfs/home/dk5565/forecasting/phase42

# Data prep (filters complete windows from phase41 chunk data)
cd tft && sbatch prepare_data.sh

# Train models
cd tft && sbatch train.sh
cd iTransformer && sbatch train.sh
```

## Data Paths

| What | Path |
|------|------|
| Source per-chunk .npy files | `/gpfs/scratch/dk5565/phase41_data/` (shared with phase41) |
| Processed tensors | `/gpfs/scratch/dk5565/phase42_data/processed/` |

## Key Findings

- **iTransformer dominates TFT** across all metrics consistently
- **iTransformer calibration near target** — 78.8% overall (SpO2 hits 80.7%)
- **TFT under-calibrates** (72.6%) — prediction intervals too narrow for real physiological variability
- **16.3% keep rate** from Phase 4.1 windows (complete data across all 4 signals is stringent)
- **Chunk-level splits** prevent temporal leakage from discontinuous record concatenation while maintaining similar performance to prior patient-level splits
