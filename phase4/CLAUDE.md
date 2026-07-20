# Project Memory — Phase 4 (TFT-multi: Simultaneous Vital Sign Forecasting)

## Project Overview
Temporal Fusion Transformer (TFT-multi) for simultaneous multi-variate vital sign trajectory forecasting using MIMIC-III charted clinical data. Given 75 hours of 5 vital sign measurements from ICU charting, simultaneously predict quantile trajectories (10th, 50th, 90th percentiles) for all 5 vital signs over the next 25 hours.

Based on: [TFT-multi: simultaneous forecasting of vital sign trajectories in the ICU](https://arxiv.org/abs/2409.15586)
Code from: [rosie068/TFT-multi](https://github.com/rosie068/TFT-multi)

## Directory Structure
```
/gpfs/home/dk5565/forecasting/phase4/
├── tft/
│   ├── model.py              # TFT-multi model (exact from rosie068/TFT-multi)
│   ├── visualization.py      # Visualization functions (exact from rosie068/TFT-multi)
│   ├── download_data.py      # Extract vital signs from CHARTEVENTS.csv.gz
│   ├── preprocess.py         # Data loading, windowing, normalization, patient-level split
│   ├── train.py              # Training loop with quantile loss + masking
│   ├── test.py               # Evaluation with masked metrics (MAE, MAPE, calibration)
│   ├── plot_predictions.py   # Generate forecast plots
│   ├── run_pipeline.py       # End-to-end orchestrator
│   └── main.sh               # SLURM script
├── README.md
├── CLAUDE.md                  # This file
└── .gitignore
```

## Vital Signs (5 channels)
| Index | Signal | Description | Unit | ITEMIDs (examples) |
|-------|--------|-------------|------|-------------------|
| 0 | mean_bp | Mean arterial pressure (direct or MAP=DBP+1/3*(SBP-DBP)) | mmHg | 456, 52, 220052, 220181 |
| 1 | pulse | Heart rate | bpm | 211, 220045 |
| 2 | spo2 | Oxygen saturation | % | 646, 220277 |
| 3 | respiratory_rate | Breaths per minute | insp/min | 618, 220210 |
| 4 | temperature | Body temperature (Fahrenheit converted to Celsius) | °C | 676-679, 223761-223762 |

## Data Source & Extraction

### Source
- **CHARTEVENTS:** `/gpfs/data/eh3828lab/datasets/mimic_clinical/CHARTEVENTS.csv.gz` (4.0 GB, ~330M rows)
- **ICUSTAYS:** `/gpfs/data/eh3828lab/datasets/mimic_clinical/ICUSTAYS.csv.gz`
- **Storage:** `/gpfs/scratch/dk5565/phase4_data/`

### CHARTEVENTS Row Format
```
ROW_ID, SUBJECT_ID, HADM_ID, ICUSTAY_ID, ITEMID, CHARTTIME, STORETIME, CGID, VALUE, VALUENUM, VALUEUOM, WARNING, ERROR, RESULTSTATUS, STOPPED
```
Each row = one measurement of one vital sign for one patient at one time.

### Extraction Pipeline (download_data.py)
1. Load eligible ICU stays from ICUSTAYS.csv.gz (LOS ≥ 48h) → ~32K stays
2. Select top 300 candidates (longest LOS), target 100 final stays
3. Stream CHARTEVENTS.csv.gz with pandas chunked reading (2M rows/chunk, ~4 min)
   - Filter by ITEMID (55 vital sign codes) and ICUSTAY_ID (300 candidates)
   - Only read 4 columns: ICUSTAY_ID, ITEMID, CHARTTIME, VALUENUM
4. Compute hour offset from ICU admission: `hour = int((charttime - intime).total_seconds() / 3600)`
5. Hourly bin: mean aggregation of multiple measurements per hour
6. Convert Fahrenheit temperatures to Celsius (ITEMIDs 678, 679, 223761, 3652, 3654)
7. Compute MAP from SBP/DBP when direct mean BP unavailable: `MAP = DBP + 1/3*(SBP-DBP)`
8. Filter outliers by physiological bounds
9. Quality check: ≥30% heart rate coverage required
10. Save per-stay `.npy` files (shape: num_hours × 5) + `metadata.json`

### Physiological Bounds (outlier filtering)
- Heart rate: 20–250 bpm
- Mean BP: 20–200 mmHg
- SBP: 40–300 mmHg, DBP: 10–200 mmHg
- SpO2: 50–100%
- Respiratory rate: 4–60 breaths/min
- Temperature: 30–42°C

### Data Statistics
- ICU stays: 100 (LOS ≥ 48h, sorted by longest)
- Total hours: 288,787
- Average LOS: 2,888 hours (~120 days)
- Windows: 23,289 (75h input + 25h output, stride=12h)

## Task Formulation
- **Single model** predicts all 5 vital signs simultaneously
- **Input:** 75 hours × 11 features (5 vitals + 5 missingness masks + 1 time position)
- **Output:** 25 hours × 5 vitals × 3 quantiles (10th, 50th, 90th percentile)
- **Loss:** Quantile loss with masking (only real values contribute)
- **Quantile regression:** Predicts prediction intervals, not just point estimates

## Architecture (TFT-multi)
Exact implementation from rosie068/TFT-multi (adapted from PlaytikaOSS/tft-torch):

1. **InputChannelEmbedding** — project each input variable to state_size via learned linear layers
2. **VariableSelectionNetwork** — learn which inputs are most relevant (GRN + softmax weights)
3. **Static covariate encoders** — 4 GRNs producing context vectors for selection, enrichment, LSTM init
4. **Temporal processing** — separate past/future LSTMs with gated skip connection
5. **Static enrichment** — GRN enriches temporal features with static context
6. **InterpretableMultiHeadAttention** — causal self-attention over encoded sequence
7. **Position-wise feed-forward** — GRN + gated skip connection
8. **Output layer** — Linear(state_size → num_signals × num_quantiles)

### Model Config
```python
data_props = {
    'num_historical_numeric': 11,    # 5 vitals + 5 masks + 1 time
    'num_static_numeric': 1,          # placeholder
    'num_future_numeric': 1,          # time position (known into future)
    'num_feature_predicted': 5,       # predict all 5 vitals
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
  static_feats_numeric:  (batch, 1)      — placeholder
  historical_ts_numeric: (batch, 75, 11)  — past vitals + masks + time
  future_ts_numeric:     (batch, 25, 1)   — future time position

Output:
  predicted_quantiles:   (batch, 25, 15)  — reshaped to (batch, 25, 5, 3)
  target:                (batch, 25, 5)   — future vital sign values
  target_mask:           (batch, 25, 5)   — 1=real, 0=imputed
```

## Training (train.py)

### Loss Function (from TFT-multi notebook)
```python
# Quantile loss with masking:
errors = targets.unsqueeze(-1) - outputs  # (batch, 25, 5, 3)
errors[..., i, j] *= masks[..., i]       # zero out imputed values
losses = max((q-1)*errors, q*errors)      # pinball loss
q_loss = losses.sum(dim=-1).sum(dim=-1).mean(dim=-1).mean()
```

### Training Loop
- Adam optimizer, LR=1e-3
- Gradient clipping: max_norm=100
- Early stopping: patience=20 on **training loss** (matching their notebook)
- No learning rate scheduler (matching their notebook)
- Batch size: 64
- Default: 100 epochs

## Preprocessing (preprocess.py)

### Forward-Fill + Mask
1. Load per-stay .npy (num_hours × 5, NaN for missing)
2. Create binary mask: 1 where real measurement exists, 0 where NaN
3. Forward-fill NaN values (backfill leading edge from first valid value)
4. Sliding window: stride=12h, window=100h (75+25)

### Normalization
- Z-score per vital sign: (x - mean) / std
- Computed from **training patients' real values only** (masked mean/std)
- Applied to both historical vitals and targets
- Masks and time features left unnormalized

### Data Split (Patient-Level, No Leakage)
- Split at the **ICU stay level** (not window level)
- All windows from one stay go to the same split
- 80% train / 10% val / 10% test (by number of stays)
- Random permutation with seed=42

## Metrics (test.py)

### Per-Vital (computed only on real/masked values):
- **MAE per quantile** (q0.1, q0.5, q0.9)
- **MAPE per quantile** (q0.1, q0.5, q0.9)
- **Calibration:** % of true values within 10th–90th prediction interval (target: 80%)
- **Correlation:** Pearson r on median predictions

### Overall:
- Overall MAE (median predictions, masked)
- Overall RMSE (median predictions, masked)
- Overall calibration (mean across vitals)

## Key Parameters
| Parameter | Value |
|-----------|-------|
| Input window | 75 hours |
| Output window | 25 hours |
| Window stride | 12 hours |
| Resolution | Hourly (mean aggregation) |
| Num patients | 100 ICU stays |
| Total windows | 23,289 |
| Historical features | 11 (5 vitals + 5 masks + 1 time) |
| Future features | 1 (time position) |
| Output | 5 vitals × 3 quantiles = 15 |
| State size | 240 |
| LSTM layers | 2 |
| Attention heads | 2 |
| Dropout | 0.3 |
| Batch size | 64 |
| Learning rate | 1e-3 |
| Grad clipping | 100 |
| Early stopping | Patience 20 (train loss) |
| Default epochs | 100 |
| Model params | ~7.9M |

## Output Directory Structure
```
tft/
├── checkpoints/tft_epochs_100/best_model.pt
└── outputs/tft_epochs_100/
    ├── test_predictions.npy       # (N, 25, 5, 3)
    ├── test_targets.npy           # (N, 25, 5)
    ├── test_masks.npy             # (N, 25, 5)
    ├── test_metrics.json
    ├── training_curves.png
    ├── plot_forecast_sample_{i}.png
    ├── plot_scatter_per_vital.png
    ├── plot_error_by_step.png
    └── plot_calibration_summary.png
```

## Pipeline
```bash
# Full pipeline
cd tft
python run_pipeline.py --epochs 100

# Individual steps
python download_data.py --num-patients 100   # ~4 min
python train.py --epochs 100                  # ~11 min on GPU
python test.py --epochs 100
python plot_predictions.py --epochs 100

# SLURM submission
sbatch main.sh
```

## SLURM
- Partition: gpu4_medium
- Resources: 1 GPU, 8 CPUs, 64GB RAM, 3hr limit
- Conda env: CSDI (Python 3.11, PyTorch 2.13)
- Conda activation: `source /gpfs/share/apps/anaconda3/gpu/2023.09/etc/profile.d/conda.sh && conda activate CSDI`

## Known Considerations
1. **Temperature ITEMIDs:** Some Fahrenheit-flagged ITEMIDs may occasionally store Celsius; these get filtered by bounds (conservative data loss, not corruption)
2. **Forward-fill over long gaps:** Stale values for rarely-measured vitals (esp. temperature every 4-8h); mask correctly marks these as imputed
3. **No ERROR column filtering:** CHARTEVENTS ERROR=1 rows not excluded (rare, bounds filter catches extremes)
4. **Hour alignment:** Hours are relative to ICU admission time (floor), not clock-aligned
5. **ICUSTAY_ID = per-admission:** Same person readmitted twice = two separate stays (standard practice)

## Git
- .gitignore excludes: data/, checkpoints/, outputs/, logs/, __pycache__/, *.npy
