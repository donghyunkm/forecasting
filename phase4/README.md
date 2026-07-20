# TFT-multi: Simultaneous Vital Sign Forecasting (Phase 4)

Temporal Fusion Transformer for simultaneous multi-variate vital sign trajectory forecasting using MIMIC-III charted clinical data. Based on the TFT-multi paper: [https://arxiv.org/abs/2409.15586](https://arxiv.org/abs/2409.15586)

Model and training code adapted from: [https://github.com/rosie068/TFT-multi](https://github.com/rosie068/TFT-multi)

## Overview

Given 75 hours of 5 vital sign measurements from ICU charting, simultaneously predict quantile trajectories (10th, 50th, 90th percentiles) for all 5 vital signs over the next 25 hours.

| Property | Value |
|----------|-------|
| Model | Temporal Fusion Transformer (TFT-multi) |
| Input | 75 hours × 11 features (5 vitals + 5 masks + time) |
| Output | 25 hours × 5 vital signs × 3 quantiles |
| Params | ~7.9M |
| Quantiles | 10th, 50th, 90th percentile |
| Loss | Quantile loss with masking |

## Vital Signs

Extracted from MIMIC-III `CHARTEVENTS` table, hourly-binned:

| Vital Sign | Description | Unit |
|------------|-------------|------|
| Mean BP | Mean arterial pressure (direct or computed MAP) | mmHg |
| Pulse | Heart rate | bpm |
| SpO2 | Oxygen saturation | % |
| Respiratory Rate | Breaths per minute | insp/min |
| Temperature | Body temperature | °C |

Missing values are forward-filled; a binary mask indicates real vs imputed values and is fed to the model as additional input features.

## Architecture (TFT-multi)

The model code is taken directly from [rosie068/TFT-multi](https://github.com/rosie068/TFT-multi) (adapted from [PlaytikaOSS/tft-torch](https://github.com/PlaytikaOSS/tft-torch)):

1. **Variable Selection Networks** — learn which inputs are most relevant at each time step
2. **Gated Residual Networks (GRN)** — nonlinear processing with skip connections
3. **LSTM Encoder** — capture temporal dependencies (past → future)
4. **Interpretable Multi-Head Attention** — long-range dependencies over encoded sequence
5. **Quantile Output** — predict 10th/50th/90th percentiles for all vital signs simultaneously

### Configuration

| Hyperparameter | Value |
|---------------|-------|
| State size | 240 |
| Dropout | 0.3 |
| LSTM layers | 2 |
| Attention heads | 2 |
| Learning rate | 1e-3 |
| Optimizer | Adam |
| Grad clipping | 100 |
| Batch size | 64 |
| Epochs | 100 |
| Early stopping | Patience 20 (on train loss) |

## Project Structure

```
phase4/
├── tft/
│   ├── model.py              # TFT-multi model (from rosie068/TFT-multi)
│   ├── visualization.py      # Visualization functions (from rosie068/TFT-multi)
│   ├── download_data.py      # Extract vital signs from CHARTEVENTS.csv.gz
│   ├── preprocess.py         # Data loading, windowing, normalization
│   ├── train.py              # Training loop with quantile loss + masking
│   ├── test.py               # Evaluation with masked metrics
│   ├── plot_predictions.py   # Generate forecast plots
│   ├── run_pipeline.py       # End-to-end orchestrator
│   └── main.sh               # SLURM script
├── README.md
├── CLAUDE.md
└── .gitignore
```

## Data

- **Source:** `/gpfs/data/eh3828lab/datasets/mimic_clinical/CHARTEVENTS.csv.gz`
- **Storage:** `/gpfs/scratch/dk5565/phase4_data/`
- **ICU stays:** 100 (with LOS >= 48 hours)
- **Resolution:** Hourly bins (mean of measurements within each hour)
- **Missingness:** Forward-filled with binary mask indicators
- **Windows:** 23,289 samples (75h input + 25h output, stride=12h)

### Data Pipeline

1. Load eligible ICU stays from `ICUSTAYS.csv.gz` (LOS >= 48h)
2. Stream `CHARTEVENTS.csv.gz` with pandas chunked reading (~4 min for 330M rows)
3. Bin into hourly intervals (mean aggregation)
4. Filter outliers using physiological bounds
5. Forward-fill missing values, generate binary mask
6. Save per-stay `.npy` files + `metadata.json`

### Data Split (Patient-Level, No Leakage)

Split is performed at the **ICU stay level** — all sliding windows from the same patient go to the same split. This prevents data leakage from overlapping windows.

- **Train:** 80 stays (~18.6K windows)
- **Val:** 10 stays (~2.3K windows)
- **Test:** 10 stays (~2.3K windows)

## Quick Start

```bash
# Run full pipeline
cd tft
python run_pipeline.py --epochs 100

# Or submit as SLURM job
sbatch main.sh
```

### Individual Steps

```bash
# 1. Extract vital signs from CHARTEVENTS (~4 min)
python download_data.py --num-patients 100

# 2. Train TFT-multi model (~11 min on GPU)
python train.py --epochs 100

# 3. Evaluate on test set
python test.py --epochs 100

# 4. Generate plots
python plot_predictions.py --epochs 100
```

## Metrics

| Metric | Description |
|--------|-------------|
| MAE (per quantile) | Mean absolute error on real values only |
| MAPE (per quantile) | Mean absolute percentage error on real values only |
| Correlation | Pearson r per vital sign (median predictions) |
| Calibration | % of true values within 10th-90th quantile bounds (target: 80%) |
| Q-loss | Quantile loss with masking (training objective) |
| Per-step MAE | Error degradation over forecast horizon |

## Outputs

```
outputs/tft_epochs_100/
├── test_predictions.npy          # (N, 25, 5, 3) - 5 vitals × 3 quantiles
├── test_targets.npy              # (N, 25, 5)
├── test_masks.npy                # (N, 25, 5)
├── test_metrics.json             # All metrics (MAE, MAPE, calibration, correlation)
├── training_curves.png
├── plot_forecast_sample_{i}.png  # Time series with confidence bands
├── plot_scatter_per_vital.png    # Predicted vs actual (50th pct)
├── plot_error_by_step.png        # MAE over forecast horizon
└── plot_calibration_summary.png  # Per-vital calibration bar chart
```

## Reference

```bibtex
@article{zhang2024tft,
  title={TFT-multi: simultaneous forecasting of vital sign trajectories in the ICU},
  author={Zhang, Rosie and others},
  journal={arXiv preprint arXiv:2409.15586},
  year={2024}
}
```

## Requirements

- Python 3.11+
- PyTorch 2.0+
- omegaconf
- numpy, pandas
- matplotlib
