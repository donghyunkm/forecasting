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
- **Split:** 70% train / 15% validation / 15% test
- **Normalization:** Z-score per signal
- **Checkpoint selection:** Best model saved at minimum validation loss

## Results (20 epochs)

| Signal | LSTM MAE | LSTM RMSE | Diffusion MAE | Diffusion RMSE |
|--------|----------|-----------|---------------|----------------|
| ABP    | 1.46 mmHg | 4.37 mmHg | 3.88 mmHg | 14.26 mmHg |
| PLETH  | 0.054    | 0.110     | 0.129         | 0.334          |
| II     | 0.019 mV | 0.038 mV | 0.035 mV | 0.073 mV |

The LSTM outperforms at 20 epochs. Diffusion models typically require more training (100–200+ epochs) and benefit from cosine noise schedules and DDIM sampling.

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
