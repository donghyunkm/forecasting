# MIMIC-III Multivariate Waveform Forecasting (Phase 3)

Forecast physiological waveforms (II, PLETH, RESP, ABP) using aggregated 6-minute interval features. Uses all 4 signals as input to predict 4 aggregated features of each target signal. Two architectures: LSTM and Diffusion (DDPM).

## Overview

Given 7.5 hours (75 intervals × 6 min) of aggregated physiological signal features, predict the next 2.5 hours (25 intervals) of a target signal's 4 aggregated features (mean, std, min, max). Each interval is represented by 4 statistical features for each of the 4 signals.

| Model | Description | Epochs | Params |
|-------|-------------|--------|--------|
| **LSTM** | Bidirectional LSTM (hidden=128) with temporal attention | 100 | ~764K |
| **Diffusion** | Conditional DDPM with 1D U-Net + Transformer condition encoder | 100 | ~3.5M |

Both produce **4 models** — one per target signal (II, PLETH, RESP, ABP).

## Results (Target: II, 100 patients)

| Metric | LSTM | Diffusion |
|--------|------|-----------|
| MAE (normalized) | **0.759** | 1.075 |
| RMSE (normalized) | **1.821** | 2.260 |
| Correlation | **0.872** | 0.768 |
| Best epoch | 18 | 35 |

### Per-Feature MAE (normalized)

| Feature | LSTM | Diffusion |
|---------|------|-----------|
| mean | **0.100** | 0.148 |
| std | **0.054** | 0.079 |
| min | **0.388** | 0.506 |
| max | **0.163** | 0.252 |

### Per-Feature Correlation

| Feature | LSTM | Diffusion |
|---------|------|-----------|
| mean | **0.880** | 0.546 |
| std | **0.825** | 0.464 |
| min | **0.845** | 0.416 |
| max | **0.710** | 0.434 |

## Project Structure

```
phase3/
├── lstm/
│   ├── download_data.py     # Load waveforms from local GPFS
│   ├── preprocess.py        # Resample to 6-min, aggregate features, create datasets
│   ├── model.py             # LSTM forecasting model
│   ├── test.py              # Evaluate with best checkpoint
│   ├── plot_predictions.py  # Visualize forecasts
│   ├── run_pipeline.py      # End-to-end orchestrator
│   └── main.sh              # SLURM script
├── diffusion/
│   ├── download_data.py     # Load waveforms from local GPFS
│   ├── preprocess.py        # Same resampling/aggregation
│   ├── model.py             # Conditional DDPM forecaster
│   ├── test.py              # Reverse diffusion sampling + evaluation
│   ├── plot_predictions.py  # Visualize forecasts
│   ├── run_pipeline.py      # End-to-end orchestrator
│   └── main.sh              # SLURM script
├── check_patients.py        # Survey patient data availability
├── valid_patients.json      # Pre-scanned: 885 valid patients, 26,772 hrs
├── README.md
├── CLAUDE.md
├── requirements.txt
└── .gitignore
```

## Data

- **Source:** `/gpfs/data/eh3828lab/datasets/mimic3_waveforms_matched/`
- **Storage:** `/gpfs/scratch/dk5565/phase3_data/` (raw + resampled cache)
- **Format:** WFDB (PhysioNet standard) at 125 Hz
- **Patients:** 100 (885 valid pre-scanned, sorted by data volume)
- **Signals:** II, PLETH, RESP, ABP (all 4 channels)
- **No truncation** — full segments read

### Resampling Strategy

Raw 125 Hz waveforms are resampled into **6-minute intervals** with 4 aggregated features per signal:

| Feature | Description |
|---------|-------------|
| mean | Average value over interval |
| std | Standard deviation (variability) |
| min | Minimum value (trough) |
| max | Maximum value (peak) |

This yields **16 features per interval** (4 signals × 4 features).

## Task Formulation

- **Input:** 75 time steps × 16 features (7.5 hours of all signals)
- **Output:** 25 time steps × 4 features of target signal (2.5 hours)
- **Models:** 4 per architecture (one for each target: II, PLETH, RESP, ABP)
- **Stride:** 25 intervals (2.5 hours) between consecutive windows

## Quick Start

```bash
# Run LSTM pipeline for II target
cd lstm
python run_pipeline.py --target II --epochs 100

# Run for all targets
for TARGET in II PLETH RESP ABP; do
    python run_pipeline.py --target $TARGET --epochs 100
done
```

### SLURM Submission

```bash
# Submit all 8 jobs (4 targets × 2 models)
cd /gpfs/home/dk5565/forecasting/phase3
for TARGET in II PLETH RESP ABP; do
    sbatch lstm/main.sh $TARGET
    sbatch diffusion/main.sh $TARGET
done
```

### Individual Steps

```bash
# 1. Load data from GPFS (cached as .npy, skips if data/ exists)
python download_data.py --num-patients 100

# 2. Train model for a specific target
python model.py --target II --epochs 100

# 3. Test with best checkpoint
python test.py --target II --epochs 100

# 4. Plot predictions
python plot_predictions.py --target II --epochs 100
```

## Data Leakage Prevention

1. **Patient-level split:** Entire patients assigned to train/val/test — no patient appears in multiple splits
2. **Normalization isolation:** Z-score statistics computed from training patients only
3. **Chunk independence:** No window spans NaN-gap chunk boundaries
4. **Strict ordering:** Split → Normalize → Window

### Split (80/10/10 by data volume)

With 100 patients:

| Split | Ratio | Purpose |
|-------|-------|---------|
| Train | 80% | Model training |
| Val | 10% | Early stopping / checkpoint selection |
| Test | 10% | Final evaluation |

## Model Architectures

### LSTM
- Input projection: Linear(16 → 128) + ReLU
- 2-layer Bidirectional LSTM (hidden=128)
- Temporal attention mechanism over 75 time steps
- FC decoder: 256 → 128 → 128 → 100 (reshaped to 25 × 4)

### Diffusion (DDPM)
- **Condition encoder:** MLP (16 → 128) + 3-layer Transformer (4 heads)
  - Produces 75 context tokens of dimension 128
- **Denoiser:** 1D U-Net operating on 100 flattened values (25 steps × 4 features)
  - Levels: 100 → 50 → 25 (bottleneck) → 50 → 100
  - Cross-attention to condition tokens at each level
- **Schedule:** T=200 steps, linear β ∈ [1e-4, 0.02]
- **Output:** 100 values reshaped to (25, 4)

## Memory Optimization

Both models use a streaming data pipeline to handle large datasets within 128GB RAM:
- **Download:** Processes and saves one patient at a time (peak ~4GB per patient)
- **Preprocessing:** Loads raw data per-patient via mmap, resamples, caches to disk, frees memory
- **Resampled cache:** ~28MB total (vs 400GB raw), reused across both models

## Metrics

| Metric | Description |
|--------|-------------|
| MAE / RMSE | Overall (normalized and original units) |
| Per-feature MAE/RMSE/r | Individual metrics for each of the 4 features |
| Per-step MAE | Error degradation over forecast horizon |
| Correlation | Pearson r between predicted and actual |

## Outputs

```
outputs/{TARGET}_epochs_{N}/
├── test_predictions.npy          # (N, 25, 4) in original units
├── test_targets.npy              # (N, 25, 4) in original units
├── test_metrics.json             # All metrics
├── training_curves.png
├── plot_forecast_sample_{i}.png  # 4-panel plot (one per feature)
├── plot_scatter_per_feature.png
├── plot_error_by_step.png
└── plot_feature_summary.png
```

## CLI Arguments

All scripts accept:
- `--target {II,PLETH,RESP,ABP}` — Target signal to forecast (default: II)
- `--epochs N` — Number of training epochs (default: 100)
- `--num-patients N` — Number of patients to load (download_data.py only, default: 5)

## Requirements

- Python 3.10+
- PyTorch 2.0+
- wfdb (for WFDB file reading)
- numpy (signal processing)
- matplotlib (visualization)
