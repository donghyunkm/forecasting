# MIMIC-III Heart Rate Prediction from Waveforms (Phase 2)

Predict heart rate (BPM) from physiological waveforms (II, PLETH, RESP, ABP) using deep learning models. Data is read from the local MIMIC-III Waveform Database Matched stored on GPFS.

## Overview

Given 5 minutes (37,500 samples at 125 Hz) of 4 physiological signals, predict the heart rate (BPM) that would be observed in the next 1 minute. Heart rate is derived from the PLETH signal using peak detection with strict quality filtering. Two model architectures are implemented:

| Model | Description | Parameters |
|-------|-------------|-----------|
| **LSTM** | 2-layer LSTM (hidden=64) + MLP head, input_size=4 | ~37K |
| **Diffusion (DDPM)** | CNN-compressed condition encoder (250× reduction) + BiLSTM + MLP denoiser | ~500K |

Both use all 4 signals as input (cross-signal information) and predict a single HR scalar.

## Project Structure

```
phase2/
├── lstm/                    # LSTM model
│   ├── download_data.py     # Load waveforms from local GPFS (multi-segment, max extraction)
│   ├── preprocess.py        # Derive HR from PLETH, create datasets (chunk-aware)
│   ├── model.py             # LSTM model for HR prediction
│   ├── test.py              # Evaluate with best checkpoint
│   ├── plot_predictions.py  # Visualize HR predictions
│   ├── run_pipeline.py      # End-to-end orchestrator
│   ├── main.sh              # SLURM sbatch script
│   └── requirements.txt
├── diffusion/               # Diffusion model
│   ├── download_data.py     # Load waveforms from local GPFS
│   ├── preprocess.py        # Derive HR from PLETH, create datasets
│   ├── model.py             # DDPM with CNN-compressed conditioning
│   ├── test.py              # Generate HR via reverse diffusion
│   ├── plot_predictions.py  # Visualize HR predictions
│   ├── run_pipeline.py      # End-to-end orchestrator
│   ├── main.sh              # SLURM sbatch script
│   └── requirements.txt
├── README.md
├── CLAUDE.md
└── .gitignore
```

## Data

- **Source:** `/gpfs/data/eh3828lab/globus/ICU/mimic3_waveforms_matched/` (local GPFS, read-only)
- **Format:** WFDB (PhysioNet standard) — `.hea` header + `.dat` binary files
- **Patients:** 5 (configurable via `--num-patients`)
- **Duration:** All available clean data per patient (typically 8–50+ hours each)
- **Signals:** II, PLETH, RESP, ABP at 125 Hz
- **Multi-segment:** Extracts ALL valid segments per patient, splits around NaN regions into clean chunks

### Data Extraction Strategy

Unlike a fixed-duration approach, this pipeline:
1. Finds all segments per patient containing II+PLETH+RESP+ABP
2. Reads each segment fully, splits into NaN-free chunks
3. Keeps all chunks ≥ 45,000 samples (6 minutes — minimum for one full window)
4. No per-patient cap — uses everything available
5. Trims edges naturally (chunks that don't fit a full window are discarded)

### Source Data Directory Organization

```
/gpfs/data/eh3828lab/globus/ICU/mimic3_waveforms_matched/
├── p00/ through p09/              # Patient group directories (~1000+ patients each)
│   └── pXXXXXX/                   # Individual patient directory
│       ├── RECORDS                # List of all records for this patient
│       ├── NNNNNNN_layout.hea    # Layout header (all available signal names)
│       ├── NNNNNNN_XXXX.hea      # Segment headers
│       ├── NNNNNNN_XXXX.dat      # Segment waveform data (16-bit binary)
│       └── ...
├── RECORDS                        # All patient directory paths (10,282 patients)
└── RECORDS-waveforms              # All waveform recording paths (22,317 recordings)
```

### Patient Discovery (Optimized)

1. Read layout `.hea` (1 file per patient) — instant skip if missing any of the 4 signals
2. Scan segment headers for those with all 4 signals
3. Read segment data, split into NaN-free chunks
4. ~50% of patients have all 4 signals; finding 5 with substantial data takes ~5 seconds

### Signal Availability (from 500-patient survey)

| Signal | Availability |
|--------|-------------|
| II (ECG Lead II) | 96.0% |
| PLETH (Pulse Oximeter) | 70.2% |
| RESP (Respiration) | 55.2% |
| ABP (Arterial BP) | 54.8% |

## Heart Rate Computation & Quality Filtering

For each target window (1 minute of PLETH):
1. Detect systolic peaks with minimum distance 0.3s and adaptive prominence
2. **Require ≥5 peaks** (rejects signal loss)
3. Compute inter-peak intervals (IPI)
4. Filter physiologically implausible intervals (0.3s–2.0s)
5. **Require ≥80% of intervals to be valid** (rejects noisy detection)
6. **Require coefficient of variation < 0.5** (rejects irregular/artifact-corrupted windows)
7. HR = 60 / mean(valid IPI)
8. Discard if HR outside [30, 200] BPM

## Quick Start

```bash
# Install dependencies
pip install -r lstm/requirements.txt

# Run LSTM pipeline (load data → train → test → plot)
cd lstm
python run_pipeline.py

# Run Diffusion pipeline
cd diffusion
python run_pipeline.py
```

### Individual steps

```bash
# 1. Load data from GPFS (cached locally as .npy, skips if exists)
python download_data.py --num-patients 5

# 2. Train model
python model.py --epochs 20 --input-length 37500 --target-length 7500

# 3. Test with best checkpoint
python test.py --epochs 20 --input-length 37500 --target-length 7500

# 4. Plot predictions
python plot_predictions.py --epochs 20 --input-length 37500 --target-length 7500
```

### SLURM submission

```bash
sbatch lstm/main.sh       # LSTM: 20 epochs on L40S GPU
sbatch diffusion/main.sh  # Diffusion: 100 epochs on L40S GPU
```

## Task Formulation

- **Input:** 37,500 time steps × 4 signals (5 minutes of II + PLETH + RESP + ABP)
- **Output:** 1 scalar — heart rate in BPM (normalized during training)
- **Stride:** 3,750 samples (30 seconds) between consecutive windows
- **Signal normalization:** Z-score per channel (statistics from training data only)
- **HR normalization:** Z-score (mean/std from training HR values only)
- **Checkpoint selection:** Best model saved at minimum validation loss

## Preprocessing & Data Split

### Chunk-Aware Contiguous Temporal Split (No Data Leakage)

Each **chunk** (independent NaN-free segment) is split chronologically:

| Block | Fraction | Purpose |
|-------|----------|---------|
| First 70% | Training | |
| Next 15% | Validation | |
| Last 15% | Test | |

Only blocks large enough to fit one full window (45,000 samples = 6 min) are kept.

### Data Leakage Prevention

1. **Temporal isolation** — Each chunk split independently; no window spans chunk or split boundaries
2. **Signal normalization isolation** — Z-score mean/std computed from training blocks only
3. **HR normalization isolation** — HR mean/std computed from training HR values only
4. **Chunk independence** — Chunks from different segments/time periods are never mixed within a window

## Model Architectures

### LSTM
- 2-layer LSTM, input_size=4, hidden=64
- FC head: Linear(64→32) → ReLU → Linear(32→1)
- Processes raw 37,500-step sequence directly

### Diffusion (DDPM) with CNN Compression
- **Condition encoder:** Strided 1D CNN compresses 37,500 → 150 tokens (250× reduction)
  - Conv1d(4→32, k=15, s=5) → Conv1d(32→64, k=11, s=5) → Conv1d(64→128, k=7, s=5) → Conv1d(128→128, k=5, s=2)
  - BatchNorm + SiLU after each layer
- **Temporal modeling:** Bidirectional LSTM (2 layers, hidden=128) on 150 compressed tokens
- **Denoiser:** MLP conditioned on [noisy_hr + condition_vector + time_embedding]
- **Diffusion:** T=200 steps, linear beta schedule [1e-4, 0.02], x₀ clipping at ±6σ

## Results

### Data Summary (5 patients)

| Patient | Chunks | Duration |
|---------|--------|----------|
| p056796 | 5 | 8.4 hrs |
| p048056 | 18 | 47.8 hrs |
| p057905 | 17 | 61.3 hrs |
| p080744 | 19 | 55.3 hrs |
| p012799 | 22 | 48.5 hrs |
| **Total** | **81** | **221.2 hrs** |

Dataset splits (stride=30s, input=5min, target=1min):

| Split | Samples |
|-------|---------|
| Train | 13,562 |
| Val | 2,391 |
| Test | 2,317 |

### Diffusion Model (100 epochs, best at epoch 33)

| Metric | Value |
|--------|-------|
| MAE | 2.77 BPM |
| RMSE | 5.36 BPM |
| Within ±5 BPM | 86.7% |
| Within ±10 BPM | 95.3% |
| HR mean (training) | 73.2 BPM |
| HR std (training) | 18.0 BPM |

## Metrics

| Metric | Description |
|--------|-------------|
| MAE (BPM) | Mean absolute error in beats per minute |
| RMSE (BPM) | Root mean square error in BPM |
| Within ±5 BPM | Percentage of predictions within 5 BPM of ground truth |
| Within ±10 BPM | Percentage of predictions within 10 BPM of ground truth |

## Outputs Generated

```
outputs/in{X}_tgt{Y}_epochs_{N}/
├── test_predictions_hr.npy      # Model predictions (BPM)
├── test_targets_hr.npy          # Ground truth HR (BPM)
├── test_metrics.json            # All metrics
├── training_curves.png          # Loss curves
├── plot_hr_timeseries.png       # Time series comparison
├── plot_hr_scatter.png          # Predicted vs actual scatter
├── plot_hr_errors.png           # Error distribution histograms
└── plot_hr_bland_altman.png     # Bland-Altman agreement plot
```

## CLI Arguments

All scripts accept:
- `--epochs N` — Number of training epochs (default: 20 for LSTM, 100 for diffusion)
- `--input-length N` — Input window in samples (default: 7500, main.sh uses 37500)
- `--target-length N` — Target window in samples (default: 7500)
- `--num-patients N` — Number of patients to load (download_data.py only, default: 5)

## Requirements

- Python 3.10+
- PyTorch 2.0+
- wfdb (for WFDB file reading)
- numpy, matplotlib, scipy (for peak detection)
