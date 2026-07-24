# Project Memory — Phase 6 (Correlation-Only Forecasting)

## Project Overview
Forecasting waveform-derived correlation vectors from their own history. Uses MIMIC-III waveform-derived data from `data_m3_120s_prediction` at 2.5-minute resolution. Correlations are both input AND output (no vital signs in the pipeline). Point predictions with Huber loss in Fisher z-space.

Two models implemented:
1. **TFT** — Temporal Fusion Transformer (7.0M params)
2. **iTransformer** — Inverted Transformer (1.6M params)

### Key Properties
- **Input:** 48 steps (2h) × 8 channels (7 correlations + 1 time position)
- **Output:** 12 steps (30min) × 7 correlations (point predictions)
- **Loss:** Huber (δ=1.0) in Fisher z-space
- **Transform:** clip ±0.9999 → arctanh (Fisher z) → z-score normalize
- **Resolution:** 2.5 min (150s)
- **Data source:** `data_m3_120s_prediction/corr_features_focused.npy` (564,596 windows, 0% NaN)
- **Patient split:** seed=42, 70/15/15 (zero overlap)

### Motivation
- Phase 5 showed correlations provide only ~1.3% benefit for vital sign forecasting
- This phase investigates whether correlations are themselves forecastable
- Higher resolution (2.5 min) captures faster hemodynamic dynamics
- If correlations can be accurately predicted, they could serve as early warning signals

## Directory Structure
```
/gpfs/home/dk5565/forecasting/phase6/
├── prepare_data.py            # Build correlation-only sequences from data_m3_120s_prediction
├── prepare_data.sh            # SLURM: data preparation (CPU, 32GB)
├── tft/                       # TFT model
│   ├── model.py               # TFT architecture (rosie068/TFT-multi)
│   ├── preprocess.py          # Data loading from .pt files
│   ├── train.py               # Training loop with Huber loss
│   ├── test.py                # Evaluation + plots
│   └── train.sh               # SLURM: training + eval (GPU)
├── iTransformer/              # iTransformer model
│   ├── model.py               # iTransformer architecture
│   ├── preprocess.py          # Data loading from .pt files
│   ├── train.py               # Training with Huber loss
│   ├── test.py                # Evaluation + plots
│   └── train.sh               # SLURM: training + eval (GPU)
├── phase6_data/processed/     # Tensors + metadata
│   ├── train_data.pt, val_data.pt, test_data.pt
│   ├── norm_params.json       # Fisher z-space means/stds
│   └── split_info.json        # Patient lists + config
├── README.md
├── CLAUDE.md                  # This file
└── .gitignore
```

## Feature Vector

### Input (8-dim)
| Index | Feature | Description |
|-------|---------|-------------|
| 0 | PLETH_ACDC × PLETH_amp | PPG AC/DC coupling integrity |
| 1 | ABP_area × ABP_tau | Stroke volume vs vascular resistance |
| 2 | ABP_area × ShockIdx | BP vs shock index |
| 3 | PLETH_amp × ShockIdx | Perfusion vs shock |
| 4 | PLETH_ACDC × ShockIdx | Perfusion index vs shock |
| 5 | ShockIdx × ABP_tau | Shock vs vascular resistance |
| 6 | PLETH_ACDC × ABP_tau | Perfusion vs resistance |
| 7 | Time position | Linear [0, 0.75] for history, [0.76, 1.0] for future |

### Output (7-dim)
Same 7 correlations (indices 0-6) predicted 12 steps (30 min) into the future. Point predictions, no uncertainty.

## Data Source & Processing

### Source
- **Base data:** `/gpfs/data/eh3828lab/mimic_derived_data/data_m3_120s_prediction/`
  - `corr_features_focused.npy` — (564596, 7) all correlation values, zero NaN
  - `patient_ids.npy` — (564596,) patient identifiers
  - `seg_names.npy` — (564596,) segment identifiers
  - `window_times.npy` — (564596,) timestamps in seconds
  - `block_start_times.npy` — (564596,) block start times for continuity grouping
- **Processed tensors:** `/gpfs/home/dk5565/forecasting/phase6/phase6_data/processed/`

### Data Preparation (prepare_data.py)
1. Load arrays from `data_m3_120s_prediction` (564,596 windows × 7 correlations)
2. Group windows into continuous segments using (seg_name, block_start_time)
3. Sort each segment by window_time, verify uniform 150s spacing
4. Keep segments ≥ 60 windows (all 4,092 qualify)
5. Form sliding windows: 60 steps (48 input + 12 output), stride=12
6. **No NaN filtering needed** — corr_features_focused has 0% NaN
7. Split by patient (70/15/15, seed=42)
8. **Fisher z-transform:** clip to ±0.9999 → arctanh → z-score normalize using training set stats
9. Add time position feature as 8th input channel
10. Save tensors to .pt files

### Continuous Segments
- 4,092 continuous time series (all ≥ 60 windows)
- Avg length: 138 windows (345 min / ~5.75 hours)
- Max: 145 windows (capped by upstream extraction)

### Window Parameters
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Input window | 48 steps (2 hours) | Good coverage within segment constraints |
| Output window | 12 steps (30 min) | Clinically actionable forecast horizon |
| Total window | 60 steps (2.5 hours) | Fits within all segments (min=65) |
| Stride | 12 steps (30 min) | stride = forecast → no target overlap between consecutive windows |
| Data point spacing | 2.5 minutes (150s) | Native resolution of data_m3_120s_prediction |

### Data Scale
| Split | Patients | Windows |
|-------|----------|---------|
| Train | 1,010 | 20,879 |
| Val | 216 | 4,444 |
| Test | 217 | 4,763 |
| **Total** | **1,443** | **30,086** |

### Normalization
- **Transform:** Fisher z (arctanh) — maps bounded [-1, 1] correlations to unbounded space
- **Why:** correlations near ±1 are compressed in raw space; Fisher z stretches them out, making the distribution more Gaussian-like
- **Clip:** ±0.9999 before arctanh (5.6% of values are exactly +1.0)
- **Z-score:** computed on training set in Fisher z-space
- **Inverse:** tanh(z * std + mean) → back to [-1, 1]

## Task Formulation
- **Single model** predicts all 7 correlations simultaneously
- **Input:** 48 steps × 8 features (7 correlations + 1 time position)
- **Output:** 12 steps × 7 correlations (point predictions)
- **Loss:** Huber loss (δ=1.0) in Fisher z-space — robust to outliers from extreme correlations
- **Forecast targets:** All 7 Pearson correlations

## Architecture (TFT)

### Model Config
```python
data_props = {
    'num_historical_numeric': 8,      # 7 corr + 1 time
    'num_static_numeric': 1,          # placeholder
    'num_future_numeric': 1,          # time position (known into future)
    'num_feature_predicted': 7,       # predict all 7 correlations
}
configuration = {
    'model': {'state_size': 240, 'dropout': 0.3, 'lstm_layers': 2,
              'attention_heads': 2, 'output_quantiles': [0.5]},  # single output per feature
    'task_type': 'regression',
    'target_window_start': None,
}
```
Note: `output_quantiles=[0.5]` is a structural config for the TFT base model — it gives 1 output per feature = point prediction. The output dict key is `predicted_quantiles` but contains direct point predictions.

### Model Input/Output (batch dict)
```
Input:
  static_feats_numeric:  (batch, 1)       — placeholder
  historical_ts_numeric: (batch, 48, 8)   — past correlations + time
  future_ts_numeric:     (batch, 12, 1)   — future time position

Output:
  predicted_quantiles:   (batch, 12, 7)   — point predictions
  target:                (batch, 12, 7)   — future correlation values
```

## Architecture (iTransformer)

### iTransformer Config
```python
{
    'seq_len': 48,          # input steps
    'pred_len': 12,         # output steps
    'n_vars': 7,            # 7 correlations (input variates, excluding time)
    'n_input_vars': 8,      # + 1 time position
    'n_output_vars': 7,     # predict all 7 correlations
    'd_model': 256,
    'n_heads': 4,
    'd_ff': 512,
    'n_layers': 3,
    'dropout': 0.1,
}
```

### Key Design
- Each variate's full 48-step history is treated as a single token
- Attention is across variates (not across time)
- All correlation tokens (indices 0-6) are used for output projection
- Time token (index 7) participates in attention but has no output head
- Output is (B, 12, 7) — one value per correlation per future timestep

## Training
- **Optimizer:** Adam
- **TFT:** LR=1e-3, grad clip max_norm=100
- **iTransformer:** LR=1e-4, grad clip max_norm=1.0, cosine annealing scheduler
- **Early stopping:** Patience 20 on val loss
- **Batch size:** 64
- **Epochs:** 100 (default)
- **Loss:** Huber (δ=1.0)

## Metrics (computed in original correlation space [-1, 1])
- Per-correlation: MAE, RMSE, MAPE, Pearson r
- Overall: MAE, RMSE
- Note: MAPE uses |target| > 0.05 threshold since correlations can be near zero

## Plots (generated by test.py)
- `training_curves.png` — train/val loss over epochs
- `sample_forecasts.png` — full 2h input history + 30min forecast (solid=true, dashed=predicted)
- `error_by_horizon.png` — MAE at each of 12 forecast steps per correlation
- `scatter_pred_vs_actual.png` — scatter plots per correlation
- `bland_altman.png` — Bland-Altman agreement plots (bias + ±1.96 SD limits)
- `metrics_bar.png` — bar charts of per-correlation MAE and Pearson r

## SLURM

### Data preparation
```bash
cd /gpfs/home/dk5565/forecasting/phase6
sbatch prepare_data.sh          # cpu_short, 2h, 32GB
```

### Model training
```bash
cd tft && sbatch train.sh       # gpu4_medium, 6h
cd ../iTransformer && sbatch train.sh  # gpu4_medium, 6h
```

### Environment
- `source /gpfs/share/apps/anaconda3/gpu/2023.09/etc/profile.d/conda.sh && conda activate CSDI`

## Key Paths
- **Source data:** `/gpfs/data/eh3828lab/mimic_derived_data/data_m3_120s_prediction/`
- **Processed tensors:** `/gpfs/home/dk5565/forecasting/phase6/phase6_data/processed/`

## Data Leakage Prevention
- **Split by patient** — all segments of a patient are in the same split (train OR val OR test)
- **Zero patient overlap** between any two splits (verified)
- **Each segment belongs to exactly one patient** (verified)
- **Stride = forecast length** — consecutive windows have 0% target overlap within split

## Git
- .gitignore excludes: phase6_data/, checkpoints/, outputs/, logs/, __pycache__/, *.npy, *.pt, *.pyc
