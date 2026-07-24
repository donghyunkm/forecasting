# Project Memory — Phase 6.1 (Correlation + Physio Feature Forecasting)

## Project Overview
Extends Phase 6 by adding physiological feature statistics from X_stats.npy to the model input.
Still forecasting waveform-derived correlation vectors, but now the model also sees 38 summarized
physiological features (mean + std of 19 features over 109 sub-windows within each 120s context).

Two models implemented:
1. **TFT** — Temporal Fusion Transformer
2. **iTransformer** — Inverted Transformer

### Key Properties
- **Input:** 48 steps (2h) × 46 channels (7 correlations + 38 physio stats + 1 time position)
- **Output:** 12 steps (30min) × 7 correlations (point predictions)
- **Loss:** Huber (δ=1.0) in Fisher z-space
- **Transform:** Correlations: clip ±0.9999 → arctanh (Fisher z) → z-score. Physio: z-score → NaN imputed with 0
- **Resolution:** 2.5 min (150s)
- **Data source:** `data_m3_120s_prediction/corr_features_focused.npy` + `X_stats.npy`
- **Patient split:** seed=42, 70/15/15 (zero overlap)

### What's New vs Phase 6
- **+38 input channels** from X_stats.npy (19 mean + 19 std per 2.5-min window)
- Input width: 8 → 46 channels
- TFT: VariableSelectionNetwork now selects from 46 variables (was 8)
- iTransformer: attention across 46 variate tokens (was 8), only 7 produce output
- Hypothesis: physiological features (HR, BP, perfusion, etc.) provide causal context that improves correlation forecasting

## Directory Structure
```
/gpfs/home/dk5565/forecasting/phase61/
├── prepare_data.py            # Build sequences from data_m3_120s_prediction + X_stats
├── prepare_data.sh            # SLURM: data preparation (CPU, 64GB)
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
├── phase61_data/processed/    # Tensors + metadata
│   ├── train_data.pt, val_data.pt, test_data.pt
│   ├── norm_params.json       # Normalization params (corr Fisher z + physio z-score)
│   └── split_info.json        # Patient lists + config
├── README.md
├── CLAUDE.md                  # This file
└── .gitignore
```

## Feature Vector

### Input (46-dim)
| Index | Feature | Description |
|-------|---------|-------------|
| 0 | PLETH_ACDC × PLETH_amp | PPG AC/DC coupling integrity |
| 1 | ABP_area × ABP_tau | Stroke volume vs vascular resistance |
| 2 | ABP_area × ShockIdx | BP vs shock index |
| 3 | PLETH_amp × ShockIdx | Perfusion vs shock |
| 4 | PLETH_ACDC × ShockIdx | Perfusion index vs shock |
| 5 | ShockIdx × ABP_tau | Shock vs vascular resistance |
| 6 | PLETH_ACDC × ABP_tau | Perfusion vs resistance |
| 7–25 | Physio mean (19 features) | Mean of each feature over 109 sub-windows |
| 26–44 | Physio std (19 features) | Std of each feature over 109 sub-windows |
| 45 | Time position | Linear [0, 0.75] for history, [0.76, 1.0] for future |

### Physiological Features (19 from X_stats)
| Index | Name | Unit | Description |
|-------|------|------|-------------|
| 0 | HR | bpm | Heart rate |
| 1 | RR | br/min | Respiratory rate |
| 2 | SBP | mmHg | Systolic blood pressure |
| 3 | DBP | mmHg | Diastolic blood pressure |
| 4 | PP | mmHg | Pulse pressure |
| 5 | MAP | mmHg | Mean arterial pressure |
| 6 | ABP_area | mmHg·s | ABP waveform area (stroke volume proxy) |
| 7 | PLETH_ACDC | ratio | Pleth AC/DC ratio |
| 8 | PLETH_amp | a.u. | Pleth amplitude |
| 9 | ECG_Ramp | mV | ECG R-wave amplitude |
| 10 | HRV_RMSSD | ms | Heart rate variability (RMSSD) |
| 11 | HR_range | bpm | Heart rate range in sub-window |
| 12 | ShockIdx | – | Shock index (HR/SBP) |
| 13 | PPV | % | Pulse pressure variation |
| 14 | PVI | % | Pleth variability index |
| 15 | PTT | ms | Pulse transit time |
| 16 | dPdt_max | mmHg/s | Max rate of pressure rise |
| 17 | ABP_tau | s | Exponential decay time constant |
| 18 | RESP_amp | a.u. | Respiratory amplitude |

### Output (7-dim)
Same 7 correlations (indices 0-6) predicted 12 steps (30 min) into the future. Point predictions, no uncertainty.

## Data Source & Processing

### Source
- **Base data:** `/gpfs/data/eh3828lab/mimic_derived_data/data_m3_120s_prediction/`
  - `corr_features_focused.npy` — (564596, 7) all correlation values, zero NaN
  - `X_stats.npy` — (564596, 19, 109) physiological features × 109 sub-windows
  - `patient_ids.npy` — (564596,) patient identifiers
  - `seg_names.npy` — (564596,) segment identifiers
  - `window_times.npy` — (564596,) timestamps in seconds
  - `block_start_times.npy` — (564596,) block start times for continuity grouping
- **Processed tensors:** `/gpfs/home/dk5565/forecasting/phase61/phase61_data/processed/`

### X_stats Summarization
- X_stats shape: (564596, 19, 109) — 19 features computed in 109 sliding sub-windows
- Sub-windows: 120s window, 10s stride within the 20-min (1200s) context → 109 positions
- Summarization: nanmean + nanstd over 109 sub-windows → (564596, 38) per data point
- Result: 19 mean values + 19 std values = 38 features per 2.5-min window
- NaN handling: features with all-NaN sub-windows get NaN mean/std, imputed to 0 after z-score

### Data Preparation (prepare_data.py)
1. Load arrays from `data_m3_120s_prediction` (564,596 windows × 7 correlations + 19×109 physio)
2. Summarize X_stats: (N, 19, 109) → (N, 38) via nanmean + nanstd over 109 sub-windows
3. Group windows into continuous segments using (seg_name, block_start_time)
4. Sort each segment by window_time, verify uniform 150s spacing
5. Keep segments ≥ 60 windows
6. Form sliding windows: 60 steps (48 input + 12 output), stride=12
7. **NaN filter:** reject windows with >50% NaN in physio features
8. Split by patient (70/15/15, seed=42)
9. **Correlation normalization:** clip to ±0.9999 → arctanh (Fisher z) → z-score (training stats)
10. **Physio normalization:** z-score normalize using training stats → impute NaN with 0
11. Concatenate: 7 corr + 38 physio + 1 time = 46 input channels
12. Save tensors to .pt files

### Window Parameters
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Input window | 48 steps (2 hours) | Good coverage within segment constraints |
| Output window | 12 steps (30 min) | Clinically actionable forecast horizon |
| Total window | 60 steps (2.5 hours) | Fits within all segments (min=65) |
| Stride | 12 steps (30 min) | stride = forecast → no target overlap between consecutive windows |
| Data point spacing | 2.5 minutes (150s) | Native resolution of data_m3_120s_prediction |

### Normalization
- **Correlations:** clip ±0.9999 → arctanh (Fisher z) → z-score (training set stats)
- **Physio features:** z-score normalize (training set stats) → NaN imputed with 0.0
- **NaN imputation:** 0 in z-score space = population mean — conservative default for missing data
- **Inverse (for output):** tanh(z * std + mean) → back to correlation space [-1, 1]

## Task Formulation
- **Single model** predicts all 7 correlations simultaneously
- **Input:** 48 steps × 46 features (7 correlations + 38 physio stats + 1 time position)
- **Output:** 12 steps × 7 correlations (point predictions)
- **Loss:** Huber loss (δ=1.0) in Fisher z-space — robust to outliers from extreme correlations
- **Forecast targets:** All 7 Pearson correlations

## Architecture (TFT)

### Model Config
```python
data_props = {
    'num_historical_numeric': 46,     # 7 corr + 38 physio + 1 time
    'num_static_numeric': 1,          # placeholder
    'num_future_numeric': 1,          # time position (known into future)
    'num_feature_predicted': 7,       # predict all 7 correlations
}
configuration = {
    'model': {'state_size': 240, 'dropout': 0.3, 'lstm_layers': 2,
              'attention_heads': 2, 'output_quantiles': [0.5]},
    'task_type': 'regression',
    'target_window_start': None,
}
```

### Model Input/Output (batch dict)
```
Input:
  static_feats_numeric:  (batch, 1)        — placeholder
  historical_ts_numeric: (batch, 48, 46)   — past correlations + physio + time
  future_ts_numeric:     (batch, 12, 1)    — future time position

Output:
  predicted_quantiles:   (batch, 12, 7)    — point predictions
  target:                (batch, 12, 7)    — future correlation values
```

## Architecture (iTransformer)

### iTransformer Config
```python
{
    'seq_len': 48,          # input steps
    'pred_len': 12,         # output steps
    'n_vars': 7,            # 7 correlations (output variates)
    'n_input_vars': 46,     # 7 corr + 38 physio + 1 time
    'n_output_vars': 7,     # predict all 7 correlations
    'd_model': 256,
    'n_heads': 4,
    'd_ff': 512,
    'n_layers': 3,
    'dropout': 0.1,
}
```

### Key Design
- Each variate's full 48-step history is treated as a single token (46 tokens total)
- Attention is across variates (not across time)
- Only correlation tokens (indices 0-6) are used for output projection
- Physio tokens (indices 7-44) and time token (index 45) participate in attention but have no output head
- Output is (B, 12, 7) — one value per correlation per future timestep
- Physio features inform the model about the underlying physiological state driving correlation changes

## Training
- **Optimizer:** Adam
- **TFT:** LR=1e-3, grad clip max_norm=100
- **iTransformer:** LR=1e-4, grad clip max_norm=1.0, cosine annealing scheduler
- **Early stopping:** Patience 20 on val loss
- **Batch size:** 64
- **Epochs:** 100 (default)
- **Loss:** Huber (δ=1.0)

## Results

### Training Summary
| Model | Best Epoch | Best Val Loss | Total Epochs |
|-------|-----------|---------------|--------------|
| TFT | 11 | 0.2740 | 31 (early stopped) |
| iTransformer | 12 | 0.2689 | 32 (early stopped) |

### Overall Test Performance
| Model | MAE | RMSE | Params |
|-------|-----|------|--------|
| **iTransformer** | **0.273** | **0.384** | 1.6M |
| TFT | 0.279 | 0.389 | 18.6M |

### Comparison to Phase 6 (correlation-only, 8 input channels)

| Model | Phase 6 MAE | Phase 6.1 MAE | Δ | Phase 6 r (avg) | Phase 6.1 r (avg) | Δ |
|-------|-------------|---------------|---|-----------------|-------------------|---|
| iTransformer | 0.274 | 0.273 | −0.4% | 0.592 | 0.597 | +0.8% |
| TFT | 0.277 | 0.279 | +0.7% | 0.591 | 0.590 | −0.2% |

### Per-Correlation Comparison (iTransformer)
| Correlation | Phase 6 MAE | Phase 6.1 MAE | Phase 6 r | Phase 6.1 r |
|-------------|-------------|---------------|-----------|-------------|
| PLETH_ACDC × PLETH_amp | 0.064 | 0.062 | 0.625 | 0.633 |
| ABP_area × ABP_tau | 0.313 | 0.311 | 0.694 | 0.698 |
| ABP_area × ShockIdx | 0.250 | 0.247 | 0.684 | 0.694 |
| PLETH_amp × ShockIdx | 0.317 | 0.316 | 0.509 | 0.515 |
| PLETH_ACDC × ShockIdx | 0.321 | 0.320 | 0.539 | 0.544 |
| ShockIdx × ABP_tau | 0.333 | 0.332 | 0.587 | 0.587 |
| PLETH_ACDC × ABP_tau | 0.320 | 0.320 | 0.510 | 0.511 |

### Per-Correlation Comparison (TFT)
| Correlation | Phase 6 MAE | Phase 6.1 MAE | Phase 6 r | Phase 6.1 r |
|-------------|-------------|---------------|-----------|-------------|
| PLETH_ACDC × PLETH_amp | 0.063 | 0.061 | 0.645 | 0.652 |
| ABP_area × ABP_tau | 0.317 | 0.328 | 0.688 | 0.677 |
| ABP_area × ShockIdx | 0.251 | 0.255 | 0.681 | 0.685 |
| PLETH_amp × ShockIdx | 0.322 | 0.323 | 0.500 | 0.499 |
| PLETH_ACDC × ShockIdx | 0.325 | 0.327 | 0.533 | 0.531 |
| ShockIdx × ABP_tau | 0.336 | 0.337 | 0.581 | 0.576 |
| PLETH_ACDC × ABP_tau | 0.324 | 0.323 | 0.507 | 0.508 |

### Key Findings
1. **Adding 38 physiological features provides negligible improvement** — overall MAE changes by <1%
2. iTransformer shows a marginal gain (−0.4% MAE, +0.8% avg r); TFT shows a marginal loss (+0.7% MAE)
3. TFT with 46 inputs (18.6M params) overfits slightly vs 8 inputs (7.0M params) — the VSN has 46 GRNs to learn with limited data
4. **Conclusion:** Correlations already encode the relevant physiological dynamics. Adding the raw source features (HR, BP, etc.) does not provide additional predictive power for forecasting correlation trajectories. The correlations are a sufficient summary of the underlying state for their own forecasting.

## Metrics (computed in original correlation space [-1, 1])
- Per-correlation: MAE, RMSE, MAPE, Pearson r
- Overall: MAE, RMSE
- Note: MAPE uses |target| > 0.05 threshold since correlations can be near zero

## Plots (generated by test.py)
- `training_curves.png` — train/val loss over epochs
- `sample_forecasts_*.png` — full 2h input history + 30min forecast (one per correlation)
- `error_by_horizon.png` — MAE at each of 12 forecast steps per correlation
- `scatter_pred_vs_actual.png` — scatter plots per correlation
- `bland_altman.png` — Bland-Altman agreement plots (bias + ±1.96 SD limits)
- `metrics_bar.png` — bar charts of per-correlation MAE and Pearson r

## SLURM

### Data preparation
```bash
cd /gpfs/home/dk5565/forecasting/phase61
sbatch prepare_data.sh          # cpu_short, 2h, 64GB
```

### Model training
```bash
cd tft && sbatch --dependency=afterok:<data_job_id> train.sh       # gpu4_medium, 6h
cd ../iTransformer && sbatch --dependency=afterok:<data_job_id> train.sh  # gpu4_medium, 6h
```

### Job History
| Job ID | Task | Status |
|--------|------|--------|
| 25795818 | Data preparation | ✓ Completed |
| 25795824 | TFT train + test | ✓ Completed |
| 25795825 | iTransformer train + test | ✓ Completed |

### Environment
- `source /gpfs/share/apps/anaconda3/gpu/2023.09/etc/profile.d/conda.sh && conda activate CSDI`

## Key Paths
- **Source data:** `/gpfs/data/eh3828lab/mimic_derived_data/data_m3_120s_prediction/`
- **Processed tensors:** `/gpfs/home/dk5565/forecasting/phase61/phase61_data/processed/`

## Data Leakage Prevention
- **Split by patient** — all segments of a patient are in the same split (train OR val OR test)
- **Zero patient overlap** between any two splits (verified)
- **Each segment belongs to exactly one patient** (verified)
- **Stride = forecast length** — consecutive windows have 0% target overlap within split

## Git
- .gitignore excludes: phase61_data/, checkpoints/, outputs/, logs/, __pycache__/, *.npy, *.pt, *.pyc
