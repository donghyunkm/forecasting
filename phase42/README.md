# Phase 4.2 — Vital Sign Forecasting with Complete Windows Only

Same task as Phase 4.1 but **only uses windows with zero missing data** across all 4 signals at all timesteps. No forward-fill, no mask channels. Forces the model to learn actual physiological dynamics.

## Results

| Metric | TFT | iTransformer |
|--------|-----|--------------|
| Overall MAE | 4.40 | **4.09** |
| Overall RMSE | 7.05 | **6.59** |
| Calibration (target 80%) | 70.4% | **76.4%** |

### Per-Vital MAE
| Signal | TFT | iTransformer |
|--------|-----|--------------|
| mean_bp | 7.45 mmHg | **6.99 mmHg** |
| pulse | 6.21 bpm | **5.68 bpm** |
| spo2 | 1.33 % | **1.24 %** |
| resp_rate | 2.62 /min | **2.46 /min** |

### Phase 4.1 vs 4.2 (iTransformer)
| Metric | Phase 4.1 | Phase 4.2 | Notes |
|--------|-----------|-----------|-------|
| MAE | 4.29 | **4.09** | Better point forecasts |
| RMSE | 7.10 | **6.59** | Better on outliers |
| Calibration | **80.8%** | 76.4% | More variability to capture |
| Correlation | 0.760 | **0.780** | Tracks dynamics better |

### Uncertainty Quantification (Calibration & Prediction Intervals)

Both models predict 10th, 50th, 90th percentiles. The 10th–90th interval = 80% prediction band.

| Signal | TFT Calibration | iTrans Calibration | Target |
|--------|-----------------|--------------------| -------|
| mean_bp | 72.1% | **77.0%** | 80% |
| pulse | 65.5% | **76.0%** | 80% |
| spo2 | 71.7% | **77.2%** | 80% |
| resp_rate | 72.3% | **75.3%** | 80% |

- Both models under-calibrate on complete-window data (intervals too narrow)
- Real physiological data has more variability than forward-filled data → harder to bound
- iTransformer closer to target (76.4% vs TFT's 70.4%)
- TFT pulse calibration especially poor (65.5%) — fails to capture HR variability

## What's Different from Phase 4.1

| Aspect | Phase 4.1 | Phase 4.2 |
|--------|-----------|-----------|
| Missing data | Forward-fill + mask input | Discard (no NaN allowed) |
| Input features | 9 (4 vitals + 4 masks + 1 time) | 5 (4 vitals + 1 time) |
| Training windows | 216,908 | 29,791 (14% kept) |
| Data quality | Mixed real + imputed | 100% real measurements |
| Selection bias | All patients | Patients with full monitoring |

## Quick Start

```bash
cd /gpfs/home/dk5565/forecasting/phase42

# Data prep (filters complete windows from phase41 data, ~13 sec)
cd tft && sbatch prepare_data.sh

# Train models
cd tft && sbatch train.sh
cd iTransformer && sbatch train.sh
```

## Data Paths

| What | Path |
|------|------|
| Source .npy files | `/gpfs/scratch/dk5565/phase41_data/` (shared with phase41) |
| Processed tensors | `/gpfs/scratch/dk5565/phase42_data/processed/` |

## Key Findings

- MAE improved ~5% by removing forward-filled data
- Correlation improved — model tracks real physiological fluctuations better
- Calibration dropped — real data has more variability than flat forward-filled segments, intervals need to be wider
- 14% keep rate (37K from 267K windows) still provides sufficient training data
- iTransformer continues to outperform TFT on all metrics
