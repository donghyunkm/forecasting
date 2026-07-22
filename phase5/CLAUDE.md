# Project Memory — Phase 5 (Vital Sign Forecasting with Correlation Features)

## Project Overview
Multi-variate vital sign trajectory forecasting using MIMIC-III data, enhanced with waveform-derived correlation features from the mimicEran pipeline. Predicts 24 future steps (2 hours) from 72 historical steps (6 hours) at 5-min resolution for 4 vital signs.

Two models implemented and compared:
1. **TFT** — Temporal Fusion Transformer
2. **iTransformer** — Inverted Transformer (ICLR 2024)

### Key Difference from Phase 4.2
- **New data source:** mimicEran merged output (11-dim vectors at 5-min stride) instead of per-patient .npy files at 15-min resolution
- **Additional input features:** 7 correlation features from waveform analysis (capture hemodynamic coupling dynamics)
- **Higher temporal resolution:** 5-min steps (vs 15-min in Phase 4.2) → more data points, finer dynamics
- **Input features:** 12 (7 correlations + 4 vitals + 1 time position) vs 5 in Phase 4.2
- **Output:** Still 4 vitals × 3 quantiles (forecasting targets unchanged)

### Motivation
- Phase 4.2 achieved good results forecasting vitals from vitals alone
- Waveform-derived correlations capture cardiovascular coupling dynamics (perfusion × shock, vascular resistance × stroke volume) that may predict future vital sign changes
- The 7 correlation features provide physiological context that pure vital sign history cannot

## Directory Structure
```
/gpfs/home/dk5565/forecasting/phase5/
├── tft/
│   ├── model.py              # TFT model (same architecture as Phase 4.2)
│   ├── preprocess.py         # Data loading from mimicEran merged output
│   ├── prepare_data.py       # Build sequences, filter complete windows, save tensors
│   ├── train.py              # Training loop with quantile loss
│   ├── test.py               # Evaluation metrics
│   ├── plot_predictions.py   # Generate forecast plots
│   ├── prepare_data.sh       # SLURM script for data preparation (CPU)
│   └── train.sh              # SLURM script for training + eval (GPU)
├── iTransformer/
│   ├── model.py              # iTransformer architecture
│   ├── preprocess.py         # Fast data loading from pre-saved .pt files
│   ├── train.py              # Training with quantile loss
│   ├── test.py               # Evaluation
│   ├── plot_predictions.py   # Visualization plots
│   └── train.sh              # SLURM script for training + eval (GPU)
├── README.md
├── CLAUDE.md                 # This file
└── .gitignore
```

## Feature Vector (11-dim input from mimicEran)

| Index | Feature | Source | Description |
|-------|---------|--------|-------------|
| 0 | PLETH_ACDC × PLETH_amp | waveform corr | PPG AC/DC coupling integrity |
| 1 | ABP_area × ABP_tau | waveform corr | Stroke volume vs vascular resistance |
| 2 | ABP_area × ShockIdx | waveform corr | BP vs shock index |
| 3 | PLETH_amp × ShockIdx | waveform corr | Perfusion vs shock |
| 4 | PLETH_ACDC × ShockIdx | waveform corr | Perfusion index vs shock |
| 5 | ShockIdx × ABP_tau | waveform corr | Shock vs vascular resistance |
| 6 | PLETH_ACDC × ABP_tau | waveform corr | Perfusion vs resistance |
| 7 | ABPMean | numerics | Mean arterial blood pressure (mmHg) |
| 8 | PULSE | numerics | Pulse rate (bpm) |
| 9 | SpO2 | numerics | Oxygen saturation (%) |
| 10 | RESP | numerics | Respiratory rate (breaths/min) |

## Data Source & Processing

### Source
- **Raw merged data:** `/gpfs/home/dk5565/mimicEran/output/merged/`
  - `features.npy` — (N, 11) all windows
  - `patient_ids.npy` — (N,) patient identifiers
  - `seg_names.npy` — (N,) segment identifiers
  - `window_times.npy` — (N,) timestamps (seconds)
- **Processed tensors:** `/gpfs/home/dk5565/forecasting/phase5/phase5_data/processed/`

### Processing Pipeline
1. Load merged arrays from mimicEran output
2. Group windows by (patient_id, seg_name) → continuous time series per segment
3. Sort each segment by window_time
4. Verify temporal continuity (5-min stride = 300s between consecutive windows)
5. Form sliding windows: 96 steps total (72 input + 24 output), stride=12
6. **Filter:** Only keep windows with zero NaN across all 96 steps × 11 features
7. Split by patient (80/10/10, seed=42)
8. Z-score normalize all 11 features using training set statistics
9. Save tensors to .pt files

### Window Parameters
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Input window | 72 steps (6 hours) | Sufficient history for vital sign dynamics |
| Output window | 24 steps (2 hours) | Clinically useful forecast horizon |
| Total window | 96 steps (8 hours) | Requires 8h continuous monitoring |
| Stride | 12 steps (1 hour) | Balance between sample count and overlap |
| Resolution | 5 minutes | Native mimicEran output stride |

### Input Features
| Feature | Channels | Description |
|---------|----------|-------------|
| Correlations (0-6) | 7 | Waveform-derived Pearson correlations |
| Vitals (7-10) | 4 | ABPMean, PULSE, SpO2, RESP |
| Time position | 1 | Linearly spaced [0, 1] |
| **Total input** | **12** | |

### Output
- 24 steps × 4 vitals × 3 quantiles (10th, 50th, 90th)
- Only vital signs are forecasted (correlations are input-only features)

## Task Formulation
- **Single model** predicts all 4 vital signs simultaneously (not one model per vital)
- **Input:** 72 steps × 12 features (7 correlations + 4 vitals + 1 time position)
- **Output:** 24 steps × 4 vitals × 3 quantiles
- **Loss:** Quantile loss (pinball loss, no masking needed — all complete windows)
- **Forecast targets:** ABPMean, PULSE, SpO2, RESP only (indices 7-10 from original 11-dim vector)

## Architecture (TFT)

### Model Config
```python
data_props = {
    'num_historical_numeric': 12,     # 7 corr + 4 vitals + 1 time
    'num_static_numeric': 1,          # placeholder
    'num_future_numeric': 1,          # time position (known into future)
    'num_feature_predicted': 4,       # predict 4 vitals only
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
  static_feats_numeric:  (batch, 1)       — placeholder
  historical_ts_numeric: (batch, 72, 12)  — past correlations + vitals + time
  future_ts_numeric:     (batch, 24, 1)   — future time position

Output:
  predicted_quantiles:   (batch, 24, 12)  — reshaped to (batch, 24, 4, 3)
  target:                (batch, 24, 4)   — future vital sign values
```

## Architecture (iTransformer)

### iTransformer Config
```python
{
    'seq_len': 72,          # input steps
    'pred_len': 24,         # output steps
    'n_vars': 11,           # 7 corr + 4 vitals (input variates)
    'n_input_vars': 12,     # + 1 time position
    'n_output_vars': 4,     # predict 4 vitals only
    'n_quantiles': 3,       # 10th, 50th, 90th
    'd_model': 256,
    'n_heads': 4,
    'd_ff': 512,
    'n_layers': 3,
    'dropout': 0.1,
}
```

## Training
- **Optimizer:** Adam
- **TFT:** LR=1e-3, grad clip max_norm=100
- **iTransformer:** LR=1e-4, grad clip max_norm=1.0
- **Early stopping:** Patience 20 on val loss
- **Batch size:** 64
- **Epochs:** 100 (default)
- **Loss:** Quantile loss (pinball), no masking

## Metrics (same as Phase 4.2)
- Per-vital MAE, MAPE per quantile (q0.1, q0.5, q0.9)
- Calibration: % of true values within 10th–90th prediction interval (target: 80%)
- Correlation: Pearson r on median predictions
- Overall MAE, RMSE on median predictions

## SLURM

### Data preparation (prepare_data.sh)
- Partition: cpu_medium (or cpu_short)
- Resources: 8 CPUs, 32GB RAM
- Job: Load mimicEran merged → build sequences → filter complete → save tensors

### Model training (train.sh)
- Partition: gpu4_medium
- Resources: 1 GPU, 8 CPUs, 64GB RAM, 6hr limit
- Runs: train.py → test.py → plot_predictions.py

### Environment
- Conda env: CSDI (Python 3.11, PyTorch)
- Activation: `source /gpfs/share/apps/anaconda3/gpu/2023.09/etc/profile.d/conda.sh && conda activate CSDI`

## Key Paths
- **mimicEran merged data:** `/gpfs/home/dk5565/mimicEran/output/merged/`
- **Processed tensors:** `/gpfs/home/dk5565/forecasting/phase5/phase5_data/processed/`
- **Phase 4.2 (comparison):** `/gpfs/home/dk5565/forecasting/phase42/`

## Expected Data Scale
From mimicEran analysis (~1.35M total windows, 73% fully complete):
- ~900 patients with ≥1 valid 8h continuous sequence
- ~145k forecast samples (8h sequences with all vitals present)
- Train: ~100k, Val: ~22k, Test: ~22k (70/15/15 by patient)

## Results

### Current Run (42/50 extraction parts merged)
- Train: 10,917 windows (562 patients)
- Val: 2,318 windows (120 patients)
- Test: 2,191 windows (121 patients)

### Overall Performance

| Model | MAE | RMSE |
|-------|-----|------|
| iTransformer | **3.18** | **5.37** |
| TFT | 3.28 | 5.48 |

### iTransformer Per-Vital

| Vital | MAE | RMSE | Correlation | Calibration |
|-------|-----|------|-------------|-------------|
| ABPMean | 5.68 mmHg | 8.04 | 0.850 | 77.9% |
| PULSE | 4.11 bpm | 6.42 | 0.930 | 79.7% |
| SpO2 | 0.89% | 2.41 | 0.795 | 78.7% |
| RESP | 2.05 br/min | 3.47 | 0.812 | 78.0% |

### TFT Per-Vital

| Vital | MAE | RMSE | Correlation | Calibration |
|-------|-----|------|-------------|-------------|
| ABPMean | 5.86 mmHg | 8.23 | 0.844 | 80.2% |
| PULSE | 4.25 bpm | 6.58 | 0.928 | 78.2% |
| SpO2 | 0.93% | 2.53 | 0.790 | 79.9% |
| RESP | 2.08 br/min | 3.54 | 0.812 | 80.3% |

### Comparison with Phase 4.2

| Metric | Phase 4.2 (iTransformer) | Phase 5 (iTransformer) | Change |
|--------|--------------------------|------------------------|--------|
| Overall MAE | 4.09 | **3.18** | -22% |
| Mean BP correlation | 0.791 | **0.850** | +7.5% |
| Pulse correlation | 0.862 | **0.930** | +7.9% |
| SpO2 correlation | 0.704 | **0.795** | +12.9% |
| Resp correlation | 0.764 | **0.812** | +6.3% |
| Mean calibration | 76.4% | **78.6%** | +2.2pp |

**Key findings:**
- Correlation features provide substantial predictive value (+6–13% correlation improvement)
- Calibration improved across all vitals (especially TFT: 70%→80%)
- Both models benefit equally from the richer input
- iTransformer still slightly outperforms TFT overall

**⚠️ Important caveat:** Phase 5 has a shorter forecast horizon (2h vs 6.25h) AND higher temporal resolution (5-min vs 15-min), so the comparison is NOT apples-to-apples. Lower MAE and higher correlations are partially expected from the easier task alone. A controlled experiment on the same data/horizon is still needed to isolate the contribution of correlation features.

### TODO
- [ ] Run ablation: train Phase 5 architecture with vitals-only input (drop correlations) to measure the true added value of waveform features at the same horizon/resolution
- [ ] Re-run with full merged data (all 50 extraction parts) for more training samples

## Git
- .gitignore excludes: phase5_data/, checkpoints/, outputs/, logs/, __pycache__/, *.npy, *.pt, *.pyc
