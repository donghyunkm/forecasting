# Project Memory — Phase 4.2 (Vital Sign Forecasting — Complete Windows Only, No Missing Data)

## Project Overview
Simultaneous multi-variate vital sign trajectory forecasting using MIMIC-III Waveform Database Matched Subset. Same task as Phase 4.1 (predict 25 future steps from 75 historical steps at 15-min resolution), but **only using windows where ALL 4 vital signs have real measurements at EVERY timestep**. No forward-fill, no masks.

Two models implemented and compared:
1. **TFT** — Temporal Fusion Transformer (from [rosie068/TFT-multi](https://github.com/rosie068/TFT-multi), based on [arXiv:2409.15586](https://arxiv.org/abs/2409.15586))
2. **iTransformer** — Inverted Transformer (from [thuml/iTransformer](https://github.com/thuml/iTransformer), ICLR 2024, [arXiv:2310.06625](https://arxiv.org/abs/2310.06625))

### Key Difference from Phase 4.1
Phase 4.1 uses forward-fill imputation + binary masks for missing data (input: 75×9 = 4 vitals + 4 masks + 1 time). Phase 4.2 **discards all windows with any missing data** — only keeps windows where all 100 steps × 4 signals have real measurements. Input is (75, 5) = 4 vitals + 1 time position. No mask channels needed.

### Motivation
- In Phase 4.1, models tended to predict flat/mean values
- Hypothesis: forward-filled data teaches the model that "staying flat" is correct
- By only using complete windows, the model sees genuine physiological variability
- Trade-off: fewer training samples but higher quality signal

## Directory Structure
```
/gpfs/home/dk5565/forecasting/phase42/
├── tft/
│   ├── model.py              # TFT model (from rosie068/TFT-multi)
│   ├── visualization.py      # Visualization functions (from rosie068/TFT-multi)
│   ├── preprocess.py         # Data loading, windowing, normalization (complete windows only)
│   ├── prepare_data.py       # Filter + preprocess + save tensors
│   ├── train.py              # Training loop with quantile loss (no masking needed)
│   ├── test.py               # Evaluation metrics (all values are real)
│   ├── plot_predictions.py   # Generate forecast plots
│   ├── prepare_data.sh       # SLURM script for data preparation (CPU)
│   └── train.sh              # SLURM script for training + eval (GPU)
├── iTransformer/
│   ├── model.py              # iTransformer architecture
│   ├── preprocess.py         # Fast data loading from pre-saved .pt files
│   ├── train.py              # Training with quantile loss (no masking)
│   ├── test.py               # Evaluation (same metrics as TFT)
│   ├── plot_predictions.py   # Visualization plots
│   └── train.sh              # SLURM script for training + eval (GPU)
├── README.md
├── CLAUDE.md                  # This file
└── .gitignore
```

## Vital Signs (4 channels)
| Index | Signal | Description | Unit |
|-------|--------|-------------|------|
| 0 | mean_bp | Mean arterial pressure | mmHg |
| 1 | pulse | Heart rate/Pulse | bpm |
| 2 | spo2 | Oxygen saturation | % |
| 3 | respiratory_rate | Breaths per minute | insp/min |

## Data Source & Processing

### Source
- **Raw .npy files:** Reuses Phase 4.1 per-chunk data from `/gpfs/scratch/dk5565/phase41_data/`
- **Processed tensors:** `/gpfs/scratch/dk5565/phase42_data/processed/`

### Filtering Strategy
1. Load per-chunk `.npy` files (shape: N×4, NaN for missing). Each chunk is a continuous recording segment (gap detection >30 min between records).
2. Create sliding windows (stride=12, window=100 steps) within each chunk
3. **Filter:** Only keep windows with **zero NaN** across all 100 steps × 4 signals
4. Split at chunk level (80/10/10, seed=42) — each chunk independently assigned to train/val/test
5. Z-score normalize using statistics from training chunks' data
6. No forward-fill needed — all values are real measurements
7. No mask channels — input is just vitals + time position

### Expected Data Characteristics
- **FEWER windows** than Phase 4.1 (266,823 windows) since many windows have gaps
- Especially fewer windows for patients with intermittent BP or SpO2 monitoring
- Higher quality: every value is a genuine physiological measurement
- No imputation artifacts that could bias the model toward flat predictions

### Input Features (Phase 4.2 vs Phase 4.1)
| Feature | Phase 4.1 | Phase 4.2 |
|---------|-----------|-----------|
| Vitals (channels 0-3) | Z-normed, forward-filled | Z-normed, all real |
| Masks (channels 4-7) | Binary (1=real, 0=imputed) | **NOT USED** |
| Time position | Channel 8 | Channel 4 |
| **Total input features** | **9** | **5** |

## Task Formulation
- **Single model** predicts all 4 vital signs simultaneously
- **Input:** 75 steps (18.75 hours) × 5 features (4 vitals + 1 time position)
- **Output:** 25 steps (6.25 hours) × 4 vitals × 3 quantiles (10th, 50th, 90th percentile)
- **Loss:** Quantile loss (no masking needed — all target values are real)
- **Quantile regression:** Predicts prediction intervals, not just point estimates

## Architecture (TFT)

### Model Config
```python
data_props = {
    'num_historical_numeric': 5,      # 4 vitals + 1 time (was 9 in Phase 4.1)
    'num_static_numeric': 1,          # placeholder
    'num_future_numeric': 1,          # time position (known into future)
    'num_feature_predicted': 4,       # predict all 4 vitals
}
configuration = {
    'model': {'state_size': 240, 'dropout': 0.3, 'lstm_layers': 2,
              'attention_heads': 2, 'output_quantiles': [0.1, 0.5, 0.9]},
    'optimization': {'learning_rate': 1e-3, 'max_grad_norm': 100},
    'task_type': 'regression',
    'target_window_start': None,
}
```

### Model Input/Output (batch dict)
```
Input:
  static_feats_numeric:  (batch, 1)      — placeholder (always 0)
  historical_ts_numeric: (batch, 75, 5)  — past vitals + time position
  future_ts_numeric:     (batch, 25, 1)  — future time position

Output:
  predicted_quantiles:   (batch, 25, 12) — reshaped to (batch, 25, 4, 3)
  target:                (batch, 25, 4)  — future vital sign values (all real)
```

### Input Features Detail
Historical (75 steps × 5 features):
- Channels 0-3: Z-score normalized vitals (all real measurements, no imputation)
- Channel 4: Time position (linearly spaced 0→0.74)

Future (25 steps × 1 feature):
- Channel 0: Time position (0.76→1.0)

## Architecture (iTransformer)

### iTransformer Config
```python
{
    'seq_len': 75,          # input steps
    'pred_len': 25,         # output steps
    'n_vars': 4,            # number of vitals
    'use_mask_input': False, # no masks in Phase 4.2
    'n_input_vars': 5,      # 4 vitals + 1 time position
    'n_output_vars': 4,     # predict 4 vitals
    'n_quantiles': 3,       # 10th, 50th, 90th
    'd_model': 256,
    'n_heads': 4,
    'd_ff': 512,
    'n_layers': 3,
    'dropout': 0.1,
}
# Optimizer: Adam, LR=1e-4, grad_clip=1.0, patience=20
```

## Training

### Loss Function
```python
# Quantile loss — NO masking needed (all values are real):
errors = targets.unsqueeze(-1) - outputs  # (batch, 25, 4, 3)
losses = max((q-1)*errors, q*errors)      # pinball loss
q_loss = losses.sum(dim=-1).sum(dim=-1).mean(dim=-1).mean()
```

### Training Loop
- Adam optimizer
- TFT: LR=1e-3, grad clipping max_norm=100
- iTransformer: LR=1e-4, grad clipping max_norm=1.0
- Early stopping: patience=20 on val loss
- Batch size: 64
- Default: 100 epochs
- Best model checkpoint saved on lowest val loss

## Metrics (test.py)

### Per-Vital (all values contribute — no masking needed):
- **MAE per quantile** (q0.1, q0.5, q0.9)
- **MAPE per quantile** (q0.1, q0.5, q0.9)
- **Calibration:** % of true values within 10th–90th prediction interval (target: 80%)
- **Correlation:** Pearson r on median predictions

### Overall:
- Overall MAE (median predictions)
- Overall RMSE (median predictions)
- Overall calibration (mean across vitals)

## Key Parameters
| Parameter | Value |
|-----------|-------|
| Input window | 75 steps (18.75 hours) |
| Output window | 25 steps (6.25 hours) |
| Window stride | 12 steps (3 hours) |
| Resolution | 15-minute intervals |
| Historical features | 5 (4 vitals + 1 time) |
| Future features | 1 (time position) |
| Output | 4 vitals × 3 quantiles = 12 |
| TFT state size | 240 |
| TFT LSTM layers | 2 |
| TFT attention heads | 2 |
| TFT dropout | 0.3 |
| iTransformer d_model | 256 |
| iTransformer n_layers | 3 |
| iTransformer n_heads | 4 |
| iTransformer dropout | 0.1 |
| Batch size | 64 |
| Early stopping | Patience 20 (val loss) |
| Default epochs | 100 |

## Output Directory Structure
```
tft/
├── checkpoints/tft_epochs_100/best_model.pt
└── outputs/tft_epochs_100/
    ├── test_predictions.npy       # (N, 25, 4, 3)
    ├── test_targets.npy           # (N, 25, 4)
    ├── test_metrics.json
    ├── training_curves.png
    ├── plot_forecast_sample_{i}.png
    ├── plot_scatter_per_vital.png
    ├── plot_error_by_step.png
    └── plot_calibration_summary.png

iTransformer/
├── checkpoints/itransformer_epochs_100/best_model.pt
└── outputs/itransformer_epochs_100/
    ├── test_predictions.npy
    ├── test_targets.npy
    ├── test_metrics.json
    ├── training_curves.png
    └── plot_*.png
```

## Pipeline

### Recommended workflow (two-stage):
```bash
# Stage 1: Prepare data (run once, CPU job)
cd tft && sbatch prepare_data.sh

# Stage 2: Train TFT (GPU job)
cd tft && sbatch train.sh

# Stage 3: Train iTransformer (GPU job)
cd iTransformer && sbatch train.sh
```

### Individual steps:
```bash
python prepare_data.py               # Filter complete windows + save tensors
python train.py --epochs 100         # Train on GPU
python test.py --epochs 100          # Evaluate
python plot_predictions.py --epochs 100  # Generate plots
```

## SLURM

### Data preparation (prepare_data.sh)
- Partition: cpu_medium
- Resources: 8 CPUs, 64GB RAM, 6hr limit
- Job: Load Phase 4.1 raw .npy → filter complete windows → save tensors

### Model training (train.sh)
- Partition: gpu4_medium
- Resources: 1 GPU, 8 CPUs, 64GB RAM, 6hr limit
- Runs: train.py → test.py → plot_predictions.py

### Environment
- Conda env: CSDI (Python 3.11, PyTorch 2.13)
- Conda activation: `source /gpfs/share/apps/anaconda3/gpu/2023.09/etc/profile.d/conda.sh && conda activate CSDI`

## Known Considerations
1. **Significantly fewer windows:** Requiring complete data across all 4 signals × 100 steps is stringent; ~16% keep rate from Phase 4.1 windows
2. **Selection bias:** Complete windows are biased toward patients with continuous monitoring (arterial lines, continuous SpO2) — typically sicker ICU patients
3. **No masking needed:** All values are real, so loss and metrics are computed on all predictions without any masking logic
4. **Simpler model input:** 5 features instead of 9 — the model doesn't need to learn to ignore/weight masked values
5. **Fewer parameters informing the model:** Without masks telling the model "this was imputed," the model has less context — but the data it sees is higher quality
6. **Shared raw data:** Per-chunk .npy files live in Phase 4.1's scratch directory; Phase 4.2 only stores processed tensors separately
7. **Chunk-level split:** Each continuous recording segment is independently assigned to train/val/test (not patient-level). Gap detection (>30 min between records) ensures no window spans a temporal discontinuity.

## Phase 4.1 vs 4.2 Comparison Summary
| Aspect | Phase 4.1 | Phase 4.2 |
|--------|-----------|-----------|
| Missing data handling | Forward-fill + masks | Discard windows with any NaN |
| Input features | 9 (4 vitals + 4 masks + 1 time) | 5 (4 vitals + 1 time) |
| Split unit | Chunk (continuous recording segment) | Chunk (continuous recording segment) |
| Training windows | 174,995 | 28,687 (16.3% keep rate) |
| Val windows | 22,081 | 3,608 |
| Test windows | 21,221 | 3,383 |
| Data quality | Mixed (real + imputed) | High (all real measurements) |
| Loss function | Quantile loss with masking | Quantile loss (mask all 1s) |

## Results

### Head-to-Head: Phase 4.2 (test set, 3,383 windows, chunk-level splits)

| Metric | TFT | iTransformer | Winner |
|--------|-----|--------------|--------|
| Overall MAE (median) | 4.42 | **4.13** | iTransformer |
| Overall RMSE (median) | 7.36 | **6.93** | iTransformer |
| Overall Calibration | 72.6% | **78.8%** | iTransformer |
| Best epoch | 12 | 19 | — |

### Per-Vital MAE (median quantile)
| Signal | TFT | iTransformer |
|--------|-----|--------------|
| mean_bp | 7.75 mmHg | **7.38 mmHg** |
| pulse | 6.19 bpm | **5.66 bpm** |
| spo2 | 1.21 % | **1.15 %** |
| resp_rate | 2.55 /min | **2.32 /min** |

### Per-Vital Calibration (target: 80%)
| Signal | TFT | iTransformer |
|--------|-----|--------------|
| mean_bp | 72.6% | **78.3%** |
| pulse | 71.5% | **78.3%** |
| spo2 | 74.4% | **80.7%** |
| resp_rate | 71.6% | **77.9%** |

### Per-Vital Correlation
| Signal | TFT | iTransformer |
|--------|-----|--------------|
| mean_bp | 0.767 | **0.788** |
| pulse | 0.848 | **0.870** |
| spo2 | 0.676 | **0.708** |
| resp_rate | 0.768 | **0.802** |

### Key Findings
- **iTransformer dominates TFT** across all metrics (consistent with prior runs)
- **iTransformer calibration near target** — 78.8% overall (SpO2 achieves 80.7%)
- **TFT calibration degraded** (72.6%) — prediction intervals too narrow
- **Chunk-level splits** prevent temporal leakage from discontinuous record concatenation
- Results are consistent with prior patient-level split runs, confirming that the chunk-level split fix does not materially change model performance while improving data integrity

## Git
- .gitignore excludes: data/, checkpoints/, outputs/, logs/, __pycache__/, *.npy, *.pyc, .ipynb_checkpoints/
