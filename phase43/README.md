# Phase 4.3 — Multi-Scale Vital Sign Forecasting (1h / 3h / 6h)

Trains **separate models for 3 forecast horizons** to test whether shorter prediction windows better capture physiological dynamics. Same complete-window data as Phase 4.2 (no missing values).

## Results

### Overall (iTransformer vs TFT)

| Horizon | Model | MAE | RMSE | Calibration |
|---------|-------|-----|------|-------------|
| 1h | TFT | 3.10 | 5.38 | 70.0% |
| 1h | **iTransformer** | **2.84** | **5.00** | **75.5%** |
| 3h | TFT | 3.87 | 6.44 | 71.1% |
| 3h | **iTransformer** | **3.57** | **6.01** | **78.0%** |
| 6h | TFT | 4.29 | 6.97 | 73.3% |
| 6h | **iTransformer** | **3.99** | **6.55** | **78.5%** |

### iTransformer Per-Vital MAE by Horizon

| Signal | 1h | 3h | 6h |
|--------|-----|-----|-----|
| mean_bp | 5.35 mmHg | 6.54 mmHg | 7.16 mmHg |
| pulse | 3.45 bpm | 4.60 bpm | 5.32 bpm |
| spo2 | 0.87 % | 1.07 % | 1.19 % |
| resp_rate | 1.70 /min | 2.06 /min | 2.31 /min |

### iTransformer Correlation by Horizon

| Signal | 1h | 3h | 6h |
|--------|-----|-----|-----|
| mean_bp | 0.888 | 0.843 | 0.817 |
| pulse | **0.934** | 0.889 | 0.855 |
| spo2 | 0.812 | 0.741 | 0.702 |
| resp_rate | 0.891 | 0.845 | 0.808 |

## Key Findings

- **1h forecast MAE is 29% lower than 6h** — shorter horizons capture dynamics much better
- **Pulse correlation at 1h: 0.934** — model tracks beat-to-beat variability well
- **iTransformer wins at all horizons** (consistent across all phases)
- **Accuracy decay is non-linear:** biggest drop is 1h→3h, then 3h→6h is smaller
- **Calibration improves with longer horizons** — short-term fluctuations are harder to bound

## Horizon Table

| Horizon | Output Steps | Train windows | Val | Test |
|---------|-------------|---------------|-----|------|
| 1h      | 4           | 36,421        | 4,834 | 5,272 |
| 3h      | 12          | 33,369        | 4,465 | 4,853 |
| 6h      | 24          | 29,418        | 3,966 | 4,331 |

All horizons use 75-step (18.75h) input. Complete windows only (no NaN).

## Quick Start

```bash
cd /gpfs/home/dk5565/forecasting/phase43

# 1. Prepare data (~15 sec)
cd tft && sbatch prepare_data.sh

# 2. Train models (loops over 1h/3h/6h)
cd tft && sbatch train.sh
cd iTransformer && sbatch train.sh
```

Individual: `python train.py --epochs 100 --horizon 1h`

## Data Paths

| What | Path |
|------|------|
| 1h processed | `/gpfs/scratch/dk5565/phase43_data/processed/horizon_4/` |
| 3h processed | `/gpfs/scratch/dk5565/phase43_data/processed/horizon_12/` |
| 6h processed | `/gpfs/scratch/dk5565/phase43_data/processed/horizon_24/` |
