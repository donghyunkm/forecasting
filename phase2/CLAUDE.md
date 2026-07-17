# Project Memory — Phase 2 (Heart Rate Prediction)

## Project Overview
Heart rate prediction from physiological waveforms (MIMIC-III Waveform Database Matched). Given 5 minutes (37,500 samples at 125Hz) of 4 signals (II, PLETH, RESP, ABP), predict the heart rate (BPM) derived from the next 1 minute of the PLETH signal.

## Directory Structure
```
/gpfs/home/dk5565/forecasting/phase2/
├── lstm/          # LSTM model for HR prediction
├── diffusion/     # DDPM model with CNN-compressed conditioning
├── README.md
├── CLAUDE.md      # This file
└── .gitignore
```

## Signals (Input — 4 channels)
- **II** (index 0) — ECG Lead II (mV) — 96% availability
- **PLETH** (index 1) — Photoplethysmogram (a.u.) — 70% availability — **used for HR derivation**
- **RESP** (index 2) — Respiration (pm) — 55% availability
- **ABP** (index 3) — Arterial Blood Pressure (mmHg) — 55% availability

## Heart Rate Derivation (Strict Quality Filtering)
- Source signal: PLETH (index 1)
- Method: `scipy.signal.find_peaks` with:
  - `distance`: 0.3s minimum (37 samples) → max 200 BPM
  - `prominence`: 30% of signal range (adaptive)
- Quality filters:
  1. **≥5 peaks required** (rejects signal loss)
  2. **≥80% of IPI must be physiologically valid** (rejects noisy detection)
  3. **Coefficient of variation < 0.5** (rejects irregular/artifact windows)
  4. HR must be within [30, 200] BPM

## Data
- Source: /gpfs/data/eh3828lab/globus/ICU/mimic3_waveforms_matched/ (local GPFS, read-only)
- Format: WFDB (.hea + .dat), read via wfdb.rdrecord(full_path, ...)
- **5 patients** (configurable via --num-patients)
- **All available clean data** per patient (no duration cap, typically 8–50+ hours each)
- **Multi-segment extraction:** all valid segments per patient, split around NaN regions
- Chunks < 45,000 samples discarded (can't fit one full window)
- download_data.py caches to data/ as .npy with chunk_boundaries in metadata.json

### Patient Discovery (Optimized)
1. Read layout .hea → instant skip if missing II/PLETH/RESP/ABP
2. Scan segment headers for those with all 4 signals
3. Read segment data, split into NaN-free chunks
4. Finding 5 patients takes ~5 seconds

## Pipeline
```
python download_data.py --num-patients 5       # Load from GPFS, cache as .npy
python model.py --epochs N --input-length L --target-length T
python test.py --epochs N --input-length L --target-length T
python plot_predictions.py --epochs N --input-length L --target-length T
python run_pipeline.py --epochs N --input-length L --target-length T
```

## Architecture
- **LSTM:** 2-layer LSTM, input_size=4, hidden=64, output=1 (HR). FC head: 64→32→1 with ReLU.
- **Diffusion:** Conditional DDPM with CNN-compressed condition encoder:
  - CNN: Conv1d stride [5,5,5,2] compresses 37,500 → 150 tokens (250× compression)
  - 4→32→64→128→128 channels, BatchNorm+SiLU after each layer
  - BiLSTM: 2 layers, hidden=128 on compressed 150-token sequence
  - Denoiser MLP: [noisy_hr, cond_vec, time_emb] → predicted noise
  - T=200 steps, linear beta [1e-4, 0.02], x₀ clipping ±6σ

## Key Parameters
| Parameter | Value |
|-----------|-------|
| Input window | 37,500 samples (5 min) — configurable via --input-length |
| Target window | 7,500 samples (1 min) for HR computation — configurable via --target-length |
| Stride | 3,750 samples (30 seconds) between windows |
| Output | 1 scalar (HR in BPM, z-normalized) |
| Sampling rate | 125 Hz |
| Num patients | 5 (configurable, no per-patient cap) |
| Signals | II, PLETH, RESP, ABP (4 channels) |
| HR bounds | [30, 200] BPM |
| HR quality filters | ≥5 peaks, ≥80% valid IPI, CV < 0.5 |
| LSTM hidden | 64, 2 layers |
| Diffusion T | 200 steps |
| CNN compression | 250× (37,500 → 150 tokens) |
| Learning rate | 0.001 (Adam) |
| Batch size | 64 |
| Default epochs | 20 (LSTM), 100 (diffusion) |
| Min chunk size | 45,000 samples (6 min) |

## Normalization
- **Signals:** Z-score per channel (mean/std from training data only)
- **HR targets:** Z-score (mean/std from training HR values only)
- Both stored in checkpoint `norm_params` dict

## Data Split (Chunk-Aware, No Data Leakage)
Each **chunk** (independent NaN-free segment) is split 70/15/15 chronologically.
Only blocks ≥ 45,000 samples (one full window) are kept.

### Data Leakage Prevention
1. Temporal isolation — each chunk split independently, no window crosses boundaries
2. Signal normalization isolation — z-score from training blocks only
3. HR normalization isolation — HR mean/std from training HR values only
4. Chunk independence — chunks from different segments never mixed in a window

## Metrics
- MAE / RMSE in BPM
- Within ±5 BPM percentage
- Within ±10 BPM percentage
- Bland-Altman plot (bias and limits of agreement)

## Output Directory Structure
```
lstm/
├── checkpoints/in37500_tgt7500_epochs_20/best_model_hr.pt
└── outputs/in37500_tgt7500_epochs_20/{test_metrics.json, *.npy, *.png}

diffusion/
├── checkpoints/in37500_tgt7500_epochs_100/best_model_hr.pt
└── outputs/in37500_tgt7500_epochs_100/{test_metrics.json, *.npy, *.png}
```

## SLURM
- LSTM: `sbatch lstm/main.sh` (20 epochs, L40S GPU)
- Diffusion: `sbatch diffusion/main.sh` (100 epochs, L40S GPU)
- Partition: gl40s_short, 32GB RAM, 4 CPUs, 12hr limit
- Conda env: CSDI (Python 3.11, PyTorch 2.x)
- Conda activation: `source /gpfs/share/apps/anaconda3/gpu/2023.09/etc/profile.d/conda.sh && conda activate CSDI`

## Git
- .gitignore excludes: data/, checkpoints/, outputs/, logs/, __pycache__/
