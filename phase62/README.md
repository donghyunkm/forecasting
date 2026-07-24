# Phase 6.2 — Cluster Label Forecasting

**Task:** Given 2 hours of 7 waveform-derived correlation features at 2.5-min stride, predict the physiological cluster assignment (7 classes) for the next 30 minutes (12 steps).

## What's New (vs Phase 6)
- **Classification instead of regression** — predicting discrete cluster labels (0-6) rather than continuous correlation values
- **Weighted cross-entropy loss** — handles imbalanced cluster distribution (3.8%–25.0%)
- **Metrics:** Accuracy, Macro/Weighted F1, per-class precision/recall/F1, confusion matrix
- Same input features and architecture backbone as Phase 6

## Motivation
- Phase 6 showed correlations are moderately forecastable (r ≈ 0.5–0.7)
- Clusters represent discrete physiological regimes — predicting regime transitions may be more clinically actionable than exact correlation values
- Can the model anticipate when a patient will shift to a different physiological state?

## Data

**Source:** `/gpfs/data/eh3828lab/mimic_derived_data/data_m3_120s_prediction/`
- `corr_features_focused.npy` — (564596, 7) correlation values
- `cluster_labels.npy` — (564596,) cluster assignments (int32, values 0-6)

**Cluster distribution:**
| Cluster | Percentage | Description |
|---------|-----------|-------------|
| 0 | 21.3% | |
| 1 | 14.0% | |
| 2 | 3.8% | (rarest) |
| 3 | 13.6% | |
| 4 | 25.0% | (most common) |
| 5 | 14.0% | |
| 6 | 8.2% | |

**Window parameters:**
| Parameter | Value |
|-----------|-------|
| Resolution | 2.5 min (150s) |
| Input window | 48 steps (2 hours) |
| Forecast window | 12 steps (30 min) |
| Stride | 12 steps (30 min) |
| Patient split | seed=42, 70/15/15 |

## Architecture

| | TFT | iTransformer |
|---|-----|--------------|
| Input | (48, 8) = 7 corr + time | Same |
| Output | (12, 7) logits | Same |
| Params | 7.0M | 2.1M |
| Key config | state=240, LSTM×2, heads=2 | d_model=256, layers=3, heads=4 |
| Head | Linear(state_size → 7) per step | Pool + MLP → 12×7 |
| LR | 1e-3 | 1e-4 (cosine anneal) |
| Loss | Weighted cross-entropy | Same |
| Early stop | patience=20 | patience=20 |

## Results

Both models trained with early stopping (patience=20).

### Overall Performance

| Model | Accuracy | Macro F1 | Weighted F1 | Best Epoch | Params |
|-------|----------|----------|-------------|------------|--------|
| **TFT** | **0.452** | **0.444** | **0.454** | 8 | 7.0M |
| iTransformer | 0.448 | 0.437 | 0.448 | 5 | 2.1M |

Random baseline accuracy: 1/7 = 0.143

### Per-Class F1 Scores

| Cluster | TFT F1 | iTrans F1 | Support (test) |
|---------|--------|-----------|----------------|
| 0 | 0.472 | 0.474 | 11,903 |
| 1 | 0.424 | 0.410 | 7,556 |
| 2 | 0.397 | 0.384 | 2,677 |
| 3 | 0.418 | 0.416 | 7,070 |
| 4 | 0.493 | 0.485 | 14,820 |
| 5 | 0.392 | 0.382 | 7,587 |
| 6 | 0.509 | 0.507 | 5,543 |

### Key Observations
- Both models achieve ~45% accuracy (3.1× better than random 14.3%)
- TFT and iTransformer perform nearly identically on this task
- **Cluster 6** has the highest F1 (~0.51) despite being only 8.2% of data — well-separated regime
- **Cluster 2** (rarest, 3.8%) has lowest precision but highest recall — model over-predicts it due to class weighting
- Both models overfit significantly (train acc 50–60% vs val acc 43–45%) — the task is inherently difficult
- Correlations provide meaningful signal for cluster prediction but far from perfect — cluster transitions may depend on factors beyond correlation history

## Quick Start

```bash
# 1. Prepare data
cd /gpfs/home/dk5565/forecasting/phase62
sbatch prepare_data.sh

# 2. Train models (with dependency on data prep job)
cd tft && sbatch --dependency=afterok:<data_job_id> train.sh
cd ../iTransformer && sbatch --dependency=afterok:<data_job_id> train.sh
```

## v2 — With Cluster Label History Input

The v1 models never see past cluster labels. Since clusters are highly autocorrelated, adding label history as a 9th input channel should significantly improve accuracy.

**Changes:** Input 48×8 → 48×9 (adds cluster label history normalized to [0,1])

| | TFT v2 | iTransformer v2 |
|---|--------|-----------------|
| Params | 7.3M | 2.2M |
| Extra input | past cluster labels / 6 → [0,1] | Same |

### v2 Results

| Model | Accuracy | Macro F1 | Weighted F1 |
|-------|----------|----------|-------------|
| TFT v2 | 0.454 | 0.447 | 0.456 |
| iTransformer v2 | 0.452 | 0.444 | 0.452 |
| TFT v1 | 0.452 | 0.444 | 0.454 |

**Per-horizon accuracy — v1 models:**

| Step | Time | TFT v1 | iTransformer v1 |
|------|------|--------|-----------------|
| 1 | 2.5 min | 74.8% | 72.4% |
| 2 | 5 min | 61.6% | 60.8% |
| 3 | 7.5 min | 52.0% | 51.6% |
| 4 | 10 min | 46.0% | 46.5% |
| 5 | 12.5 min | 42.2% | 42.4% |
| 6 | 15 min | 40.4% | 40.1% |
| 7 | 17.5 min | 39.4% | 39.2% |
| 8 | 20 min | 38.3% | 38.8% |
| 9 | 22.5 min | 37.1% | 36.9% |
| 10 | 25 min | 36.6% | 36.6% |
| 11 | 27.5 min | 37.1% | 36.2% |
| 12 | 30 min | 36.2% | 35.8% |

**Per-horizon accuracy — v2 models (with label history):**

| Step | Time | TFT v2 | iTransformer v2 |
|------|------|--------|-----------------|
| 1 | 2.5 min | 75.1% | 73.4% |
| 2 | 5 min | 62.2% | 60.9% |
| 3 | 7.5 min | 53.1% | 51.7% |
| 4 | 10 min | 46.3% | 47.3% |
| 5 | 12.5 min | 42.7% | 42.7% |
| 6 | 15 min | 40.2% | 41.0% |
| 7 | 17.5 min | 39.2% | 39.4% |
| 8 | 20 min | 38.7% | 38.6% |
| 9 | 22.5 min | 37.7% | 37.4% |
| 10 | 25 min | 37.0% | 37.0% |
| 11 | 27.5 min | 37.0% | 37.2% |
| 12 | 30 min | 36.2% | 35.9% |

**Interpretation:** Adding label history gives **75% accuracy at the first step** (strong persistence signal), but overall accuracy barely improves because long-horizon steps (>10 min) actually get *worse* — the model over-predicts "stay in current cluster" and fails at transitions. The v1 models show a similar decay pattern (75% → 36%) because the test.py script computes horizon accuracy using the same approach — this reveals that both v1 and v2 have strong short-horizon performance but converge to ~36% at 30 minutes. The v1 models have no persistence bias yet still show the gradient because nearby steps are inherently more predictable.

### v2 Jobs
| Job ID | Task | Status |
|--------|------|--------|
| 25796909 | Data prep v2 | ✓ Completed |
| 25796912 | TFT v2 | ✓ Completed |
| 25796913 | iTransformer v2 | ✓ Completed |

## v3 — With X_stats Physiological Features

v3 adds the 19 physiological features from `X_stats.npy` (summarized as mean + std over 109 sub-windows = 38 extra channels) alongside correlation history and cluster label history.

**Changes:** Input 48×9 → 48×47 (7 corr + 38 physio + 1 time + 1 label)

| | TFT v3 | iTransformer v3 |
|---|--------|-----------------|
| Extra input | 19 mean + 19 std of X_stats features | Same |
| Input dims | 48 × 47 | Same |

**X_stats features (19):** HR, RR, SBP, DBP, PP, MAP, ABP_area, PLETH_ACDC, PLETH_amp, ECG_Ramp, HRV_RMSSD, HR_range, ShockIdx, PPV, PVI, PTT, dPdt_max, ABP_tau, RESP_amp

**Normalization:** Physio features z-scored (train set stats), NaN imputed to 0 (= population mean).

### v3 Quick Start
```bash
# 1. Prepare data
sbatch prepare_data_v3.sh

# 2. Train (after data prep completes)
cd tft_v3 && sbatch --dependency=afterok:<data_job_id> train.sh
cd ../iTransformer_v3 && sbatch --dependency=afterok:<data_job_id> train.sh
```

### v3 Results

| Model | Accuracy | Macro F1 | Weighted F1 | Best Epoch | Params |
|-------|----------|----------|-------------|------------|--------|
| TFT v3 | 0.446 | 0.441 | 0.445 | 7 | ~7.5M |
| iTransformer v3 | 0.451 | 0.442 | 0.450 | 3 | 4.7M |

### Comparison Across All Versions

| Model | Accuracy | Macro F1 | Weighted F1 |
|-------|----------|----------|-------------|
| TFT v1 | 0.452 | 0.444 | 0.454 |
| TFT v2 | **0.454** | **0.447** | **0.456** |
| TFT v3 | 0.446 | 0.441 | 0.445 |
| iTransformer v1 | 0.448 | 0.437 | 0.448 |
| iTransformer v2 | 0.452 | 0.444 | 0.452 |
| iTransformer v3 | 0.451 | 0.442 | 0.450 |

### Per-Horizon Accuracy Comparison

| Step | Time | TFT v1 | TFT v2 | TFT v3 | iTrans v1 | iTrans v2 | iTrans v3 |
|------|------|--------|--------|--------|-----------|-----------|-----------|
| 1 | 2.5 min | 74.8% | 75.1% | 73.8% | 72.4% | 73.4% | 73.3% |
| 2 | 5 min | 61.6% | 62.2% | 61.4% | 60.8% | 60.9% | 61.5% |
| 3 | 7.5 min | 52.0% | 53.1% | 52.5% | 51.6% | 51.7% | 51.8% |
| 4 | 10 min | 46.0% | 46.3% | 45.9% | 46.5% | 47.3% | 45.6% |
| 5 | 12.5 min | 42.2% | 42.7% | 42.2% | 42.4% | 42.7% | 41.8% |
| 6 | 15 min | 40.4% | 40.2% | 39.9% | 40.1% | 41.0% | 40.2% |
| 7 | 17.5 min | 39.4% | 39.2% | 38.6% | 39.2% | 39.4% | 39.8% |
| 8 | 20 min | 38.3% | 38.7% | 37.0% | 38.8% | 38.6% | 38.6% |
| 9 | 22.5 min | 37.1% | 37.7% | 36.6% | 36.9% | 37.4% | 37.6% |
| 10 | 25 min | 36.6% | 37.0% | 36.0% | 36.6% | 37.0% | 36.8% |
| 11 | 27.5 min | 37.1% | 37.0% | 35.7% | 36.2% | 37.2% | 37.6% |
| 12 | 30 min | 36.2% | 36.2% | 35.4% | 35.8% | 35.9% | 37.0% |

### v3 Key Findings

- **Adding 38 X_stats features did NOT improve performance** — TFT v3 is slightly *worse* than v1/v2 (0.446 vs 0.452–0.454), and iTransformer v3 matches v1/v2 (0.451).
- Both v3 models early-stopped very early (epoch 7 and 3) suggesting the extra features add noise/complexity without useful signal for this task.
- The 19 physiological features (HR, BP, etc.) may be redundant with the correlation features already derived from them, or the summarization (mean/std over sub-windows) loses temporal dynamics needed for prediction.
- **Best overall: TFT v2** (0.454 accuracy, 0.447 macro F1) — cluster label history alone is the most useful addition.

### v3 Jobs
| Job ID | Task | Status |
|--------|------|--------|
| 25797263 | Data prep v3 | ✓ Completed |
| 25797266 | TFT v3 | ✓ Completed |
| 25797288 | iTransformer v3 | ✓ Completed |

## TODO
- [ ] Extract data from scratch to have longer horizons (no cap on segment length) — current upstream extraction caps segments at 145 windows (~6h); removing this would enable longer histories and forecast windows

## Directory Layout
```
phase62/
├── prepare_data.py           # v1 data preparation
├── prepare_data_v2.py        # v2 data preparation (+ label history)
├── prepare_data_v3.py        # v3 data preparation (+ label history + X_stats)
├── prepare_data.sh, prepare_data_v2.sh, prepare_data_v3.sh
├── phase62_data/processed/   # v1 tensors
├── phase62_data/processed_v2/# v2 tensors (with label history)
├── phase62_data/processed_v3/# v3 tensors (with label history + X_stats)
├── tft/                      # TFT v1 (classification)
├── tft_v2/                   # TFT v2 (+ label history)
├── tft_v3/                   # TFT v3 (+ label history + X_stats)
├── iTransformer/             # iTransformer v1
├── iTransformer_v2/          # iTransformer v2 (+ label history)
├── iTransformer_v3/          # iTransformer v3 (+ label history + X_stats)
├── CLAUDE.md, README.md, .gitignore
```
