# MIMIC-III Waveform Forecasting

Time-series forecasting of physiological waveforms (ABP, PLETH, ECG Lead II) from the [MIMIC-III Waveform Database Matched](https://physionet.org/content/mimic3wdb-matched/1.0/) using deep learning models.

## Overview

Given 1 second (125 samples) of all 3 physiological signals, predict the next 0.2 seconds (25 samples) of each signal. Two model architectures are implemented:

| Model | Description | Parameters per signal |
|-------|-------------|---------------------|
| **LSTM** | 2-layer LSTM (hidden=64), input_size=3 | ~52K |
| **Diffusion (DDPM)** | Conditional denoising diffusion, LSTM condition encoder + MLP denoiser | ~349K |

Both use all 3 signals as input (cross-signal information) and train a separate model per target signal.

## Project Structure

```
forecasting/
├── lstm/                    # LSTM baseline
│   ├── download_data.py     # Download waveforms from PhysioNet
│   ├── preprocess.py        # Z-score normalization, sliding window dataset
│   ├── model.py             # LSTM model, training loop
│   ├── test.py              # Evaluate with best checkpoint
│   ├── plot_predictions.py  # Visualize predictions vs ground truth
│   ├── run_pipeline.py      # End-to-end orchestrator
│   ├── main.sh              # SLURM sbatch script (L40S GPU)
│   └── requirements.txt
├── diffusion/               # Diffusion model
│   ├── download_data.py     # Download waveforms from PhysioNet
│   ├── preprocess.py        # Z-score normalization, sliding window dataset
│   ├── model.py             # DDPM model, training loop (accepts --epochs)
│   ├── test.py              # Generate forecasts via reverse diffusion
│   ├── plot_predictions.py  # Visualize predictions vs ground truth
│   ├── run_pipeline.py      # End-to-end orchestrator
│   ├── main.sh              # SLURM sbatch script (L40S GPU)
│   └── requirements.txt
└── .gitignore
```

## Data

- **Source:** [mimic3wdb-matched/1.0](https://physionet.org/content/mimic3wdb-matched/1.0/) on PhysioNet (publicly accessible)
- **Patients:** 2 patients (p000160, p000188), 10 minutes each
- **Signals:**
  - **ABP** — Arterial Blood Pressure (mmHg)
  - **PLETH** — Photoplethysmogram (PPG)
  - **II** — ECG Lead II (mV)
- **Sampling rate:** 125 Hz
- **Total samples:** 150,000 (75,000 per patient)

## Quick Start

```bash
# Install dependencies
pip install -r lstm/requirements.txt

# Run LSTM pipeline (download → train → test → plot)
cd lstm
python run_pipeline.py

# Run Diffusion pipeline
cd diffusion
python run_pipeline.py
```

### Individual steps

```bash
# 1. Download data (cached locally, only runs once)
python download_data.py

# 2. Train models
python model.py                # LSTM (20 epochs default)
python model.py --epochs 200   # Diffusion (accepts --epochs argument)

# 3. Test with best checkpoint (saved at min validation loss)
python test.py

# 4. Plot predictions
python plot_predictions.py
```

### SLURM submission (diffusion)

```bash
sbatch diffusion/main.sh   # Submits 200-epoch training on L40S GPU
```

## Task Formulation

- **Input:** 125 time steps × 3 signals (1 second of ABP + PLETH + II)
- **Output:** 25 time steps of 1 target signal (0.2 second forecast)
- **Normalization:** Z-score per signal (statistics from training data only)
- **Checkpoint selection:** Best model saved at minimum validation loss

## Preprocessing & Data Split

### Contiguous Temporal Split (No Data Leakage)

Each patient's time series is split **chronologically** before sliding window creation:

| Block | Samples per patient | Purpose |
|-------|-------------------|---------|
| First 70% | 52,500 | Training |
| Next 15% | 11,250 | Validation |
| Last 15% | 11,250 | Test |

Sliding windows (stride=1) are created **independently within each block**, then concatenated across patients:

| Split | Windows per patient | Total (2 patients) |
|-------|-------------------|-------------------|
| Train | 52,351 | 104,702 |
| Val | 11,101 | 22,202 |
| Test | 11,101 | 22,202 |

### Data Leakage Prevention

1. **Temporal isolation** — The raw time series is partitioned into contiguous blocks before any windowing. No sliding window can span across split boundaries, ensuring zero sample overlap between train/val/test.
2. **Normalization isolation** — Z-score mean and std are computed from training blocks only, then applied to val/test. No future information leaks through statistics.

## Results

### 20 Epochs

| Signal | LSTM MAE | LSTM RMSE | Diffusion MAE | Diffusion RMSE |
|--------|----------|-----------|---------------|----------------|
| ABP    | 2.44 mmHg | 4.15 mmHg | 3.30 mmHg | 10.09 mmHg |
| PLETH  | 0.078    | 0.151     | 0.128         | 0.291          |
| II     | 0.022 mV | 0.046 mV | 0.036 mV | 0.076 mV |

### 100 Epochs (Diffusion)

| Signal | Diffusion MAE (20 ep) | Diffusion MAE (100 ep) | Diffusion RMSE (100 ep) | Improvement |
|--------|-----------------------|------------------------|-------------------------|-------------|
| ABP    | 3.30 mmHg             | 3.21 mmHg              | 10.79 mmHg              | 3%          |
| PLETH  | 0.128                 | 0.130                  | 0.285                   | -2%         |
| II     | 0.036 mV              | 0.033 mV              | 0.067 mV               | 8%          |

Best checkpoints selected at minimum validation loss (ABP: epoch 46, PLETH: epoch 71, II: epoch 29).

### Summary

The LSTM remains the strongest model overall. The diffusion model shows modest improvement from 20 to 100 epochs, with ABP and II showing 3–8% MAE reduction. Diffusion models benefit from longer training and stable sampling (predicted x₀ clipping during reverse diffusion).

### Diffusion Sampling Stability Fix

The standard DDPM reverse process (`p_sample`) is numerically unstable without output clipping. With a well-trained model (200 epochs), the denoiser produces sharper noise predictions that, when divided by the small `sqrt(alpha_bar_t)` at high timesteps, amplify errors exponentially across the 200 denoising steps — causing predictions to diverge to infinity.

**Symptoms:** MAE of 10^17+ (effectively infinity/NaN) despite good training loss.

**Fix applied in `diffusion/model.py`:**
1. **Predict x₀ clipping** — During each reverse step, explicitly compute the predicted clean signal x₀ and clip it to ±6 standard deviations (matching z-normalized data range)
2. **Recompute posterior mean from clipped x₀** — Use the proper DDPM posterior formula rather than the simplified one-step formula
3. **Gradient clipping during training** — Added `clip_grad_norm_(max_norm=1.0)` to prevent the loss spikes observed at epochs 122, 162, 153

This is standard practice in DDPM implementations (analogous to Ho et al. clipping to [-1, 1] for images). Without it, a weaker model (20 epochs) stays stable by accident, but a stronger model (200 epochs) explodes during sampling.

## Outputs Generated

After running `test.py` and `plot_predictions.py`:

- `outputs/test_predictions_{abp,pleth,ii}.npy` — Model predictions (original scale)
- `outputs/test_targets_{abp,pleth,ii}.npy` — Ground truth (original scale)
- `outputs/test_metrics.json` — MSE, MAE, RMSE for all signals
- `outputs/plot_overlay_samples.png` — Per-window prediction vs ground truth
- `outputs/plot_continuous_waveform.png` — Concatenated waveform comparison
- `outputs/plot_error_analysis.png` — MAE by forecast step
- `outputs/plot_scatter.png` — Predicted vs actual scatter plots

## Requirements

- Python 3.10+
- PyTorch 2.0+
- wfdb (for PhysioNet data access)
- numpy, matplotlib, scipy
