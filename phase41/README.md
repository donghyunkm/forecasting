# Phase 4.1 — Vital Sign Forecasting from MIMIC-III Waveform Numerics

Simultaneous 4-vital-sign trajectory forecasting using MIMIC-III Waveform Database Matched Subset. Predicts 6.25-hour quantile trajectories (10th, 50th, 90th percentiles) from 18.75 hours of input at 15-minute resolution (75 input steps → 25 output steps).

Two models compared:
- **TFT** — Temporal Fusion Transformer (~7.3M params)
- **iTransformer** — Inverted Transformer (~1.6M params) ← **best performer**

## Results

| Metric | TFT | iTransformer |
|--------|-----------|--------------|
| Overall MAE | 4.46 | **4.29** |
| Overall RMSE | 7.37 | **7.10** |
| Calibration (target 80%) | 76.7% | **80.8%** |
| Parameters | 7.3M | 1.6M |
| Train time/epoch | ~4 min | ~49 sec |

### Per-Vital MAE (median)
| Signal | TFT | iTransformer |
|--------|-----------|--------------|
| mean_bp | 7.60 mmHg | **7.36 mmHg** |
| pulse | 6.39 bpm | **6.09 bpm** |
| spo2 | 1.46 % | **1.42 %** |
| resp_rate | 2.53 /min | **2.45 /min** |

### Uncertainty Quantification (Calibration & Prediction Intervals)

Both models output 3 quantiles per vital (10th, 50th, 90th percentile). The 10th–90th interval forms an 80% prediction band.

| Signal | TFT Calibration | iTrans Calibration | Target |
|--------|-----------------|--------------------| -------|
| mean_bp | 78.0% | **80.8%** | 80% |
| pulse | 77.9% | **80.3%** | 80% |
| spo2 | 73.6% | **81.6%** | 80% |
| resp_rate | 77.1% | **80.3%** | 80% |

- **iTransformer: 80.8% overall** — near-perfect calibration (trustworthy confidence bands)
- **TFT: 76.7% overall** — under-calibrated (overconfident, intervals too narrow)

## Phase 4 vs Phase 4.1

| Aspect | Phase 4 | Phase 4.1 |
|--------|---------|-----------|
| Data source | CHARTEVENTS (nurse-charted) | WFDB numerics (bedside monitor) |
| Resolution | 1 hour | 15 minutes |
| Signals | 5 (incl. temperature) | 4 (no temperature) |
| Input | 75 hours (75 steps) | 18.75 hours (75 steps) |
| Output | 25 hours (25 steps) | 6.25 hours (25 steps) |
| Patients | 100 | 7,941 |
| Training windows | 23,289 | 216,908 |

## Vital Signs

| Signal | Description | Coverage |
|--------|-------------|----------|
| mean_bp | Mean arterial pressure (mmHg) | 51.1% |
| pulse | Heart rate (bpm) | 99.0% |
| spo2 | Oxygen saturation (%) | 52.9% |
| respiratory_rate | Breaths per minute | 95.3% |

## Available Signals in Numerics (for future phases)

Beyond the 4 signals currently used, MIMIC-III waveform numerics records also contain:

| Signal | Description | Notes |
|--------|-------------|-------|
| ABPSys / ABPDias | Arterial BP systolic/diastolic | Currently collapsed to MAP; could forecast separately |
| NBPSys / NBPDias | Non-invasive BP systolic/diastolic | Intermittent (cuff-based) |
| PAPSys / PAPDias / PAPMean | Pulmonary artery pressure | Subset with PA catheters |
| CVP | Central venous pressure | Subset with central lines |
| CO | Cardiac output | Rare; thermodilution measurements |
| ST III, ST V, etc. | ECG ST segment levels | Ischemia indicator |
| PVC Rate | Premature ventricular contractions/min | Arrhythmia burden |

## Quick Start

```bash
cd /gpfs/home/dk5565/forecasting/phase41

# Step 1: Prepare data (run once, ~22 min on CPU)
cd tft && sbatch prepare_data.sh

# Step 2a: Train TFT (GPU)
cd tft && sbatch train.sh

# Step 2b: Train iTransformer (GPU)
cd iTransformer && sbatch train.sh
```

## Directory Structure

```
phase41/
├── tft/                      # TFT implementation
│   ├── model.py              # TFT architecture (from rosie068/TFT-multi)
│   ├── download_data.py      # Extract vitals from WFDB numerics
│   ├── preprocess.py         # Windowing, normalization, DataLoader
│   ├── prepare_data.py       # One-time: extract + preprocess + save .pt
│   ├── train.py              # Training
│   ├── test.py               # Evaluation
│   ├── plot_predictions.py   # Visualizations
│   ├── prepare_data.sh       # SLURM: data prep (CPU)
│   └── train.sh              # SLURM: train + eval (GPU)
├── iTransformer/             # iTransformer implementation
│   ├── model.py              # iTransformer architecture
│   ├── preprocess.py         # Fast data loading from .pt files
│   ├── train.py              # Training
│   ├── test.py               # Evaluation
│   ├── plot_predictions.py   # Visualizations
│   └── train.sh              # SLURM: train + eval (GPU)
├── CLAUDE.md
├── README.md
└── .gitignore
```

## Data Paths

| What | Path |
|------|------|
| Raw WFDB records | `/gpfs/data/eh3828lab/datasets/mimic3_waveforms_matched/` |
| Extracted .npy files | `/gpfs/scratch/dk5565/phase41_data/` |
| Pre-processed tensors | `/gpfs/scratch/dk5565/phase41_data/processed/` |
| TFT outputs | `tft/outputs/tft_epochs_100/` |
| iTransformer outputs | `iTransformer/outputs/iTransformer_epochs_100/` |

## Requirements

- Python 3.11 (conda env: CSDI)
- PyTorch 2.13
- wfdb
- omegaconf (TFT only)
- numpy
- matplotlib

## References

- TFT-multi: Liu et al., "TFT-multi: simultaneous forecasting of vital sign trajectories in the ICU", 2024. [arXiv:2409.15586](https://arxiv.org/abs/2409.15586)
- iTransformer: Liu et al., "iTransformer: Inverted Transformers Are Effective for Time Series Forecasting", ICLR 2024. [arXiv:2310.06625](https://arxiv.org/abs/2310.06625)
