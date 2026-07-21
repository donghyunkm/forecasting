# Phase 4.3 — Multi-Scale Vital Sign Forecasting (1h/3h/6h Horizons)

## Overview

Same task as Phase 4.2 (complete windows only, no missing data) but trains **SEPARATE models** for 3 forecast horizons:
- **1 hour** (4 steps)
- **3 hours** (12 steps)
- **6 hours** (24 steps)

**Hypothesis:** Shorter horizons should better capture physiological fluctuations, while longer horizons test the model's ability to maintain accuracy over extended periods.

## Key Design

- Same 75-step (18.75h) input for all horizons
- Different output lengths: 4/12/24 steps
- Separate model trained per horizon (not one model doing all three)
- Same completeness filter as Phase 4.2 (no NaN allowed in any window)
- Data source: `/gpfs/scratch/dk5565/phase41_data/` (raw .npy)
- Processed to: `/gpfs/scratch/dk5565/phase43_data/processed/horizon_{4,12,24}/`
- Both TFT and iTransformer architectures

## Directory Structure

```
phase43/
├── tft/
│   ├── model.py, preprocess.py, prepare_data.py
│   ├── train.py, test.py, plot_predictions.py, plot_full_forecast.py
│   ├── prepare_data.sh, train.sh
├── iTransformer/
│   ├── model.py, preprocess.py
│   ├── train.py, test.py, plot_predictions.py, plot_full_forecast.py
│   ├── train.sh
├── CLAUDE.md, README.md, .gitignore
```

## Horizon Details

| Horizon | Output steps | Window size | Train windows | Val windows | Test windows |
|---------|-------------|-------------|---------------|-------------|--------------|
| 1h      | 4           | 79          | 36,421        | 4,834       | 5,272        |
| 3h      | 12          | 87          | 33,369        | 4,465       | 4,853        |
| 6h      | 24          | 99          | 29,418        | 3,966       | 4,331        |

## Model Configurations

Same as Phase 4.2 per model type — only `pred_len` changes.

### TFT
- `state_size=240`
- `num_historical_numeric=5`
- Learning rate: `1e-3`
- `pred_len`: 4 / 12 / 24 depending on horizon

### iTransformer
- `d_model=256`
- `n_vars=4`
- `use_mask_input=False`
- Learning rate: `1e-4`
- `pred_len`: 4 / 12 / 24 depending on horizon

## Pipeline

```bash
# 1. Prepare data for all 3 horizons
cd tft && sbatch prepare_data.sh

# 2. Train TFT for 1h, 3h, 6h
cd tft && sbatch train.sh

# 3. Train iTransformer for 1h, 3h, 6h
cd iTransformer && sbatch train.sh
```

All scripts accept `--horizon {1h,3h,6h}`.

## Output Structure

```
tft/outputs/tft_1h_epochs_100/
tft/outputs/tft_3h_epochs_100/
tft/outputs/tft_6h_epochs_100/
iTransformer/outputs/iTransformer_1h_epochs_100/
iTransformer/outputs/iTransformer_3h_epochs_100/
iTransformer/outputs/iTransformer_6h_epochs_100/
```

## SLURM Configuration

### Data preparation (prepare_data.sh)
- Partition: cpu_medium
- Resources: 8 CPUs, 64GB RAM, 2hr limit
- Runtime: ~15 seconds (filters from phase41 data)

### Model training (train.sh)
- Partition: gpu4_medium
- Resources: 1 GPU, 8 CPUs, 64GB RAM, 6hr limit
- Runs all 3 horizons sequentially: train → test → plot for each

### Environment
- Conda env: CSDI (Python 3.11, PyTorch 2.13)
- Conda activation: `source /gpfs/share/apps/anaconda3/gpu/2023.09/etc/profile.d/conda.sh && conda activate CSDI`

## Known Considerations

- Shorter horizons (1h) produce more training windows since the sliding window is smaller
- All three horizons use identical input length (75 steps / 18.75h) for fair comparison
- Models are completely independent — no weight sharing between horizons
- Results should be compared across horizons to validate the multi-scale hypothesis
- Memory usage is similar across horizons since batch size and input length are constant
- The 6h horizon is directly comparable to Phase 4.2 (same window size of 99)

## Results

### Overall Metrics by Horizon

| Horizon | Model | MAE | RMSE | Calibration | Best Epoch |
|---------|-------|-----|------|-------------|------------|
| 1h | TFT | 3.10 | 5.38 | 70.0% | 16 |
| 1h | **iTransformer** | **2.84** | **5.00** | **75.5%** | 21 |
| 3h | TFT | 3.87 | 6.44 | 71.1% | 16 |
| 3h | **iTransformer** | **3.57** | **6.01** | **78.0%** | 23 |
| 6h | TFT | 4.29 | 6.97 | 73.3% | 12 |
| 6h | **iTransformer** | **3.99** | **6.55** | **78.5%** | 21 |

### Per-Vital MAE (iTransformer, median quantile)
| Signal | 1h | 3h | 6h |
|--------|-----|-----|-----|
| mean_bp | 5.35 mmHg | 6.54 mmHg | 7.16 mmHg |
| pulse | 3.45 bpm | 4.60 bpm | 5.32 bpm |
| spo2 | 0.87 % | 1.07 % | 1.19 % |
| resp_rate | 1.70 /min | 2.06 /min | 2.31 /min |

### Per-Vital Correlation (iTransformer)
| Signal | 1h | 3h | 6h |
|--------|-----|-----|-----|
| mean_bp | 0.888 | 0.843 | 0.817 |
| pulse | 0.934 | 0.889 | 0.855 |
| spo2 | 0.812 | 0.741 | 0.702 |
| resp_rate | 0.891 | 0.845 | 0.808 |

### Per-Vital Calibration (iTransformer, target: 80%)
| Signal | 1h | 3h | 6h |
|--------|-----|-----|-----|
| mean_bp | 75.0% | 78.0% | 78.5% |
| pulse | 74.9% | 77.6% | 78.8% |
| spo2 | 76.8% | 79.0% | 78.9% |
| resp_rate | 75.1% | 77.4% | 77.8% |

### Key Findings
- **Shorter horizons dramatically improve accuracy:** 1h MAE (2.84) is 29% lower than 6h MAE (3.99)
- **Correlation much higher at 1h:** pulse r=0.934, resp r=0.891 — model captures fluctuations well at short horizons
- **iTransformer outperforms TFT at all horizons** (consistent with Phase 4.1/4.2)
- **Calibration paradox:** 1h calibration (75.5%) is worse than 6h (78.5%) — short-term fluctuations are harder to bound with prediction intervals
- **Accuracy decay is non-linear:** 1h→3h drops 26% in MAE, 3h→6h drops only 12%
- **6h results comparable to Phase 4.2:** iTransformer 6h MAE=3.99 vs Phase 4.2 MAE=4.09 (similar, slight difference due to 24 vs 25 step output)

## Git
- .gitignore excludes: data/, checkpoints/, outputs/, logs/, __pycache__/, *.npy
