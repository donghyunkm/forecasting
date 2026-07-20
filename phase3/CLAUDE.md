# Project Memory — Phase 3 (Multivariate Waveform Forecasting)

## Project Overview
Multivariate waveform forecasting from physiological signals (MIMIC-III Waveform Database Matched). Given 75 time points (7.5 hours) of aggregated features from all 4 signals (II, PLETH, RESP, ABP), predict the next 25 time points (2.5 hours) of a target signal's 6 aggregated features. Each 6-minute interval is represented by 6 aggregated features per signal.

## Directory Structure
```
/gpfs/home/dk5565/forecasting/phase3/
├── lstm/              # LSTM model for waveform forecasting
├── diffusion/         # DDPM model for waveform forecasting
├── check_patients.py  # Survey script for patient availability
├── valid_patients.json # Cached list of 885 valid patients
├── README.md
├── CLAUDE.md          # This file
├── requirements.txt
└── .gitignore
```

## Signals (4 channels)
- **II** (index 0) — ECG Lead II (mV)
- **PLETH** (index 1) — Photoplethysmogram (a.u.)
- **RESP** (index 2) — Respiration (pm)
- **ABP** (index 3) — Arterial Blood Pressure (mmHg)

## Feature Aggregation (per 6-min interval, per signal)
Each 6-minute interval (45,000 raw samples at 125 Hz) is reduced to 6 features:
1. **mean** — average value
2. **std** — standard deviation (variability)
3. **min** — minimum value (trough)
4. **max** — maximum value (peak)
5. **skewness** — asymmetry of distribution
6. **kurtosis** — tail heaviness (excess kurtosis, Fisher)

Total features per interval: 4 signals × 6 features = **24 features**

## Task Formulation
- **4 separate models** (one per target signal: II, PLETH, RESP, ABP)
- **Input:** 75 intervals × 24 features (all 4 signals, all 6 features)
- **Output:** 25 intervals × 6 features of the target signal (mean, std, min, max, skew, kurtosis)
- **Each model uses all 4 signals as input** (cross-signal information)
- **Output shape:** (25, 6) per sample — all aggregated features of the target signal

## Data
- Source: /gpfs/data/eh3828lab/datasets/mimic3_waveforms_matched/ (local GPFS, read-only)
- Format: WFDB (.hea + .dat), read via wfdb.rdrecord()
- **10 patients** (configurable via --num-patients)
- **No truncation** — full segments read (MAX_READ_SAMPLES = None)
- Min chunk size: 45,000 samples (6 min = 1 interval)
- Multi-segment extraction: all valid segments and all NaN-free chunks per patient
- Pre-scanned patient list: valid_patients.json (885 patients, 26,772 hours total)

### Patient Discovery (Optimized)
1. Uses cached `valid_patients.json` for instant patient discovery (sorted by data volume)
2. Layout headers checked on disk (not just RECORDS file) for signal availability
3. Falls back to scanning RECORDS file if cache not found

## Pipeline
```bash
# Per-model (target signal specified via CLI)
python download_data.py --num-patients 10
python model.py --target II --epochs 50
python test.py --target II --epochs 50
python plot_predictions.py --target II --epochs 50
python run_pipeline.py --target II --epochs 50
```

## Architecture

### LSTM
- Input projection: Linear(24 → 128) + ReLU + Dropout
- 2-layer Bidirectional LSTM, hidden=128
- Temporal attention: weighted average over 75 time steps
- FC decoder: Linear(256 → 128) → ReLU → Linear(128 → 128) → ReLU → Linear(128 → 150)
- Output reshaped: 150 → (25, 6)
- Optimizer: Adam, LR=0.001, weight_decay=1e-5
- Scheduler: ReduceLROnPlateau (factor=0.5, patience=10)
- Default epochs: 50
- Parameters: ~764K

### Diffusion (DDPM)
- **Condition encoder:** MLP projection (24 → 128) + 3-layer Transformer with positional embeddings
  - Output: 75 context tokens of dim 128
- **Denoiser:** 1D U-Net with cross-attention
  - Operates on flattened 150 values (25 steps × 6 features)
  - Levels: 150 → 75 → 38 (bottleneck) → 75 → 150
  - Channels: 128 → 192 → 192 (bottleneck)
  - Each level: ResidualBlock1D + CrossAttention1D
  - Bottleneck: ResBlock + SelfAttn + CrossAttn + ResBlock
  - Skip connections between encoder/decoder levels
- **Diffusion:** T=200 steps, linear beta [1e-4, 0.02], x₀ clipping ±6σ
- **Output:** reshaped from 150 → (25, 6)
- Optimizer: AdamW, LR=0.0002, weight_decay=1e-4
- Scheduler: CosineAnnealingLR
- Default epochs: 100
- Parameters: ~3.5M

## Key Parameters
| Parameter | Value |
|-----------|-------|
| Resampling interval | 6 minutes (45,000 samples) |
| Features per interval | 6 (mean, std, min, max, skew, kurtosis) |
| Input window | 75 intervals (7.5 hours) |
| Output window | 25 intervals (2.5 hours) |
| Total input features | 24 (4 signals × 6 features) |
| Output target | 1 signal's 6 features, 25 steps (shape: 25×6=150) |
| Window stride | 1 interval (6 min) |
| Sampling rate (raw) | 125 Hz |
| Num patients | 10 (configurable) |
| Min chunk size | 45,000 samples (6 min) |
| LSTM hidden | 128, 2 layers, bidirectional |
| Diffusion T | 200 steps |
| Batch size | 64 |
| Default epochs | 50 (LSTM), 100 (diffusion) |

## Normalization
- **Features:** Z-score per (signal, feature) pair — mean/std from training patients only
- **Target:** Same normalization (target signal's features already normalized as part of the input)
- Both stored in checkpoint `norm_params` dict

## Data Split (Patient-Level, No Data Leakage)
1. Load raw waveform chunks from GPFS, grouped by patient
2. Split entire **patients** into train/val/test (80/10/10 by data volume)
3. Resample each chunk to 6-min intervals with aggregated features
4. Compute normalization from training patients only
5. Apply normalization to all data
6. Create windowed datasets within each chunk (no cross-boundary windows)

### Data Leakage Prevention
1. **Patient-level split** — no patient appears in multiple splits (completely different individuals)
2. **Normalization isolation** — z-score per (signal, feature) from training patients only
3. **Chunk independence** — no window crosses NaN-gap chunk boundaries
4. **Minimum 3 patients required** — at least 1 per split guaranteed

## Metrics
- MAE / RMSE (normalized and original units)
- Correlation (Pearson r)
- Per-feature MAE/RMSE/r (for each of the 6 aggregated features)
- Per-step MAE/RMSE (error vs forecast horizon)

## Output Directory Structure
```
lstm/
├── checkpoints/{TARGET}_epochs_{N}/best_model.pt
└── outputs/{TARGET}_epochs_{N}/{test_metrics.json, *.npy, *.png}

diffusion/
├── checkpoints/{TARGET}_epochs_{N}/best_model.pt
└── outputs/{TARGET}_epochs_{N}/{test_metrics.json, *.npy, *.png}
```

## SLURM
```bash
# Submit all 8 jobs (4 targets × 2 models)
cd /gpfs/home/dk5565/forecasting/phase3
for TARGET in II PLETH RESP ABP; do
    sbatch lstm/main.sh $TARGET
    sbatch diffusion/main.sh $TARGET
done
```
- Partition: gl40s_short, 32GB RAM, 4 CPUs, L40S GPU, 12hr limit
- Conda env: CSDI (Python 3.11, PyTorch 2.x)
- Conda activation: `source /gpfs/share/apps/anaconda3/gpu/2023.09/etc/profile.d/conda.sh && conda activate CSDI`

## Utilities
- `check_patients.py` — Survey all patients for data availability
  - `python check_patients.py --min-hours 10 --save valid_patients.json`
  - Results cached in `valid_patients.json` (885 patients, used by download_data.py)

## Git
- .gitignore excludes: data/, checkpoints/, outputs/, logs/, __pycache__/
