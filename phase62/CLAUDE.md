# Project Memory — Phase 6.2 (Cluster Label Forecasting)

## Project Overview
Forecasts future physiological cluster assignments from correlation history.
Same input as Phase 6 (7 correlations + time position), but the target is now
discrete cluster labels (7 classes, integers 0-6) instead of continuous correlations.

Two models implemented:
1. **TFT** — Temporal Fusion Transformer (7.0M params)
2. **iTransformer** — Inverted Transformer (2.1M params)

### Key Properties
- **Task:** Multi-step classification (predict cluster label at each of 12 future steps)
- **Input:** 48 steps (2h) × 8 channels (7 correlations + 1 time position)
- **Output:** 12 steps (30min) × 7 classes (logits for each cluster)
- **Loss:** Weighted cross-entropy (class weights from inverse frequency)
- **Transform:** Input correlations: clip ±0.9999 → arctanh (Fisher z) → z-score normalize
- **Resolution:** 2.5 min (150s)
- **Data source:** `data_m3_120s_prediction/corr_features_focused.npy` + `cluster_labels.npy`
- **Patient split:** seed=42, 70/15/15 (zero overlap)

### Motivation
- Phase 6 showed correlations are moderately forecastable (r ≈ 0.5–0.7)
- Clusters represent discrete physiological regimes derived from the correlations
- Predicting which regime the patient will be in (rather than exact correlation values) may be more clinically actionable
- Classification may be easier than regression if clusters are stable over time

## Directory Structure
```
/gpfs/home/dk5565/forecasting/phase62/
├── prepare_data.py            # Build sequences from data_m3_120s_prediction
├── prepare_data.sh            # SLURM: data preparation (CPU, 32GB)
├── tft/                       # TFT model
│   ├── model.py               # TFT with classification head
│   ├── preprocess.py          # Data loading from .pt files
│   ├── train.py               # Training with weighted cross-entropy
│   ├── test.py                # Evaluation (accuracy, F1, confusion matrix)
│   └── train.sh               # SLURM: training + eval (GPU)
├── iTransformer/              # iTransformer model
│   ├── model.py               # iTransformer with classification head
│   ├── preprocess.py          # Data loading from .pt files
│   ├── train.py               # Training with weighted cross-entropy
│   ├── test.py                # Evaluation (accuracy, F1, confusion matrix)
│   └── train.sh               # SLURM: training + eval (GPU)
├── phase62_data/processed/    # Tensors + metadata
│   ├── train_data.pt, val_data.pt, test_data.pt
│   ├── norm_params.json       # Fisher z-space means/stds + class_weights
│   └── split_info.json        # Patient lists + config
├── README.md
├── CLAUDE.md                  # This file
└── .gitignore
```

## Cluster Labels

### Distribution (full dataset: 564,596 windows)
| Cluster | Count | Percentage |
|---------|-------|-----------|
| 0 | 120,409 | 21.3% |
| 1 | 78,879 | 14.0% |
| 2 | 21,230 | 3.8% |
| 3 | 76,906 | 13.6% |
| 4 | 141,420 | 25.0% |
| 5 | 79,227 | 14.0% |
| 6 | 46,525 | 8.2% |

- **Imbalanced:** Cluster 4 (25%) is 6.7× more common than Cluster 2 (3.8%)
- **Class weights:** Inverse frequency weighting applied to cross-entropy loss

## Feature Vector

### Input (8-dim) — same as Phase 6
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

### Output
- 12 future time steps × 7-class probability distribution (via softmax on logits)
- Each step independently classified (no explicit transition constraints)

## Data Preparation (prepare_data.py)
1. Load correlations (564596, 7) and cluster_labels (564596,) — both zero NaN
2. Group windows into continuous segments using (seg_name, block_start_time)
3. Sort each segment by window_time, verify uniform 150s spacing
4. Keep segments ≥ 60 windows
5. Form sliding windows: 60 steps (48 input + 12 output), stride=12
6. Split by patient (70/15/15, seed=42)
7. Compute Fisher z normalization from training set
8. Compute class weights (inverse frequency) from training forecast labels
9. Normalize input correlations; target remains integer labels
10. Save tensors + norm_params.json (includes class_weights)

### Window Parameters
| Parameter | Value |
|-----------|-------|
| Input window | 48 steps (2 hours) |
| Output window | 12 steps (30 min) |
| Total window | 60 steps (2.5 hours) |
| Stride | 12 steps (30 min) |
| Resolution | 2.5 min (150s) |

## Architecture (TFT)

### Model Config
```python
data_props = {
    'num_historical_numeric': 8,      # 7 corr + 1 time
    'num_static_numeric': 1,          # placeholder
    'num_future_numeric': 1,          # time position
    'num_classes': 7,                 # 7 cluster classes
}
configuration = {
    'model': {'state_size': 240, 'dropout': 0.3, 'lstm_layers': 2, 'attention_heads': 2},
    'task_type': 'classification',
}
```

### Output
- `logits`: (B, 12, 7) — raw class scores per time step
- Loss applied as: `CrossEntropyLoss(logits.reshape(B*12, 7), target.reshape(B*12))`

## Architecture (iTransformer)

### Design
- Each variate's 48-step history → single token (8 tokens total)
- Attention across 8 variate tokens
- Pool all tokens → flatten → MLP → (pred_len × num_classes) = (12 × 7) = 84 outputs
- Reshape to (B, 12, 7) logits

### Config
```python
{
    'seq_len': 48, 'pred_len': 12, 'n_input_vars': 8,
    'num_classes': 7, 'd_model': 256, 'n_heads': 4,
    'd_ff': 512, 'n_layers': 3, 'dropout': 0.1,
}
```

## Training
- **Loss:** Weighted cross-entropy (class weights from training set inverse frequency)
- **TFT:** Adam, LR=1e-3, grad clip=100, patience=20
- **iTransformer:** Adam, LR=1e-4, cosine annealing, grad clip=1.0, patience=20
- **Batch size:** 64
- **Epochs:** 100 (with early stopping)

## Metrics
- **Overall:** Accuracy, Macro F1, Weighted F1
- **Per-class:** Precision, Recall, F1, Support
- **Per-horizon:** Accuracy at each of 12 forecast steps
- **Confusion matrix:** 7×7, normalized by true label

## Results

### Training Summary
| Model | Best Epoch | Best Val Loss | Total Epochs |
|-------|-----------|---------------|--------------|
| TFT | 8 | 1.399 | 28 (early stopped) |
| iTransformer | 5 | 1.408 | 25 (early stopped) |

### Overall Test Performance
| Model | Accuracy | Macro F1 | Weighted F1 | Params |
|-------|----------|----------|-------------|--------|
| **TFT** | **0.4515** | **0.4435** | **0.4537** | 7.0M |
| iTransformer | 0.4476 | 0.4369 | 0.4480 | 2.1M |

Random baseline: 1/7 = 14.3%

### Per-Class Results (TFT)
| Cluster | Precision | Recall | F1 | Support |
|---------|-----------|--------|-----|---------|
| 0 | 0.558 | 0.409 | 0.472 | 11,903 |
| 1 | 0.422 | 0.426 | 0.424 | 7,556 |
| 2 | 0.282 | 0.670 | 0.397 | 2,677 |
| 3 | 0.394 | 0.444 | 0.418 | 7,070 |
| 4 | 0.555 | 0.443 | 0.493 | 14,820 |
| 5 | 0.410 | 0.376 | 0.392 | 7,587 |
| 6 | 0.438 | 0.608 | 0.509 | 5,543 |

### Per-Class Results (iTransformer)
| Cluster | Precision | Recall | F1 | Support |
|---------|-----------|--------|-----|---------|
| 0 | 0.520 | 0.436 | 0.474 | 11,903 |
| 1 | 0.438 | 0.386 | 0.410 | 7,556 |
| 2 | 0.279 | 0.612 | 0.384 | 2,677 |
| 3 | 0.417 | 0.416 | 0.416 | 7,070 |
| 4 | 0.569 | 0.422 | 0.485 | 14,820 |
| 5 | 0.407 | 0.359 | 0.382 | 7,587 |
| 6 | 0.395 | 0.707 | 0.507 | 5,543 |

### Key Findings
1. **~45% accuracy (3.1× random)** — correlations provide meaningful signal for cluster prediction
2. Both models perform nearly identically; TFT has a slight edge
3. **Heavy overfitting** — train acc climbs to 55-60% while val plateaus at ~43-45%
4. **Cluster 6 best predicted** (F1 ~0.51) — likely a well-separated physiological regime
5. **Cluster 2 over-predicted** — high recall (0.67) but low precision (0.28) due to class weighting on the rarest cluster
6. Clusters 0 and 4 (the two largest) have the highest precision — the model is reasonably confident when predicting common states
7. **Task is inherently difficult** — cluster transitions likely depend on factors beyond 2h of correlation history alone

## Plots (generated by test.py)
- `training_curves.png` — loss + accuracy over epochs
- `confusion_matrix.png` — normalized confusion matrix heatmap
- `accuracy_by_horizon.png` — accuracy at each forecast step (2.5–30 min)
- `per_class_f1.png` — bar chart of F1 per cluster

## SLURM

### Data preparation
```bash
cd /gpfs/home/dk5565/forecasting/phase62
sbatch prepare_data.sh          # cpu_short, 2h, 32GB
```

### Model training
```bash
cd tft && sbatch --dependency=afterok:<data_job> train.sh       # gpu4_medium, 6h
cd ../iTransformer && sbatch --dependency=afterok:<data_job> train.sh  # gpu4_medium, 6h
```

### Job History
| Job ID | Task | Status |
|--------|------|--------|
| 25796649 | Data preparation | ✓ Completed |
| 25796650 | TFT train + test | ✓ Completed |
| 25796651 | iTransformer train + test | ✓ Completed |

### Notes
- `sklearn` and `seaborn` needed to be installed in the CSDI env for test.py
- Training completed in SLURM; test.py re-run manually after installing deps

### Environment
- `source /gpfs/share/apps/anaconda3/gpu/2023.09/etc/profile.d/conda.sh && conda activate CSDI`

## Key Paths
- **Source data:** `/gpfs/data/eh3828lab/mimic_derived_data/data_m3_120s_prediction/`
- **Processed tensors:** `/gpfs/home/dk5565/forecasting/phase62/phase62_data/processed/`

## Data Leakage Prevention
- **Split by patient** — zero patient overlap between splits
- **Normalization** — computed from training set only
- **Class weights** — computed from training forecast labels only
- **Stride = forecast length** — no target overlap between consecutive windows

## Git
- .gitignore excludes: phase62_data/, checkpoints/, outputs/, logs/, __pycache__/, *.npy, *.pt, *.pyc

## v2 Models — With Label History Input

### Motivation
The v1 models achieve ~45% accuracy but never see the past cluster labels. Since clusters are highly autocorrelated (patients tend to stay in the same regime), providing past labels as input should dramatically improve forecasting.

### Design
- **Input:** 48 steps × 9 channels (7 correlations + 1 time + 1 cluster label history)
- **Label encoding:** cluster labels (0-6) normalized to [0, 1] by dividing by 6
- **Output:** Same as v1 — 12 steps × 7 classes
- **Data:** `phase62_data/processed_v2/` (same patient split, just wider input tensor)

### Architecture
| | TFT v2 | iTransformer v2 |
|---|--------|-----------------|
| Input | (48, 9) | (48, 9) |
| Params | 7.3M | 2.2M |
| Config | `num_historical_numeric: 9` | `n_input_vars: 9` |

### Directory Structure
```
phase62/
├── prepare_data_v2.py       # Adds label history to input
├── prepare_data_v2.sh       # SLURM: data prep v2
├── phase62_data/processed_v2/  # v2 tensors
├── tft_v2/                  # TFT with label history
│   ├── model.py, preprocess.py, train.py, test.py, train.sh
├── iTransformer_v2/         # iTransformer with label history
│   ├── model.py, preprocess.py, train.py, test.py, train.sh
```

### Job History (v2)
| Job ID | Task | Status |
|--------|------|--------|
| 25796909 | Data preparation v2 | ✓ Completed |
| 25796912 | TFT v2 train + test | ✓ Completed |
| 25796913 | iTransformer v2 train + test | ✓ Completed |

### Results (v2)

#### Training Summary
| Model | Best Epoch | Best Val Loss |
|-------|-----------|---------------|
| TFT v2 | 7 | 1.397 |
| iTransformer v2 | 3 | 1.406 |

#### Overall Test Performance
| Model | Accuracy | Macro F1 | Weighted F1 |
|-------|----------|----------|-------------|
| TFT v2 | 0.4544 | 0.4468 | 0.4558 |
| iTransformer v2 | 0.4520 | 0.4439 | 0.4525 |
| TFT v1 (baseline) | 0.4515 | 0.4435 | 0.4537 |
| iTransformer v1 (baseline) | 0.4476 | 0.4369 | 0.4480 |

#### Key Finding: Accuracy by Horizon (TFT v2)
| Horizon | v1 Accuracy | v2 Accuracy | Δ |
|---------|-------------|-------------|---|
| 2.5 min (step 1) | ~45% | **75.1%** | +30% |
| 5.0 min (step 2) | ~45% | **62.2%** | +17% |
| 7.5 min (step 3) | ~45% | **53.1%** | +8% |
| 10.0 min (step 4) | ~45% | 46.3% | +1% |
| 15.0 min (step 6) | ~45% | 40.2% | −5% |
| 30.0 min (step 12) | ~45% | 36.2% | −9% |

#### Interpretation
- **Label history dominates short-term prediction** — the model learns cluster persistence (≈75% at 2.5 min)
- **Overall accuracy barely improves** because it's averaged across all 12 steps, and long-horizon accuracy *decreases* (model over-relies on persistence, hurting when transitions occur)
- At longer horizons (>10 min), the model performs *worse* than v1 — it predicts "stay in same cluster" too aggressively
- The strong horizon gradient (75% → 36%) confirms clusters are autocorrelated but transitions are hard to anticipate beyond ~10 minutes

## TODO
- [ ] Extract data from scratch to have longer horizons (no cap on segment length) — current segments are capped at 145 windows (~6h) by upstream extraction; removing this cap would allow longer input histories and forecast horizons
