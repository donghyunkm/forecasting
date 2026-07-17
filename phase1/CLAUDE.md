# Project Memory

## Project Overview
Time-series forecasting of physiological waveforms from MIMIC-III Waveform Database Matched (mimic3wdb-matched/1.0) using deep learning. Given 1 second (125 samples at 125Hz) of 3 signals, predict the next 0.2 seconds (25 samples) of each signal.

## Directory Structure
```
/gpfs/home/dk5565/forecasting/
├── lstm/          # LSTM baseline (working, verified)
├── diffusion/     # DDPM model (working, verified)
├── README.md
├── CLAUDE.md      # This file
└── .gitignore
```

## Output Directory Structure (epoch-specific)
Outputs are organized by epoch count to prevent overwriting between runs:
```
lstm/
├── checkpoints/epochs_20/best_model_{signal}.pt
└── outputs/epochs_20/{test_metrics.json, *.npy, *.png}

diffusion/
├── checkpoints/epochs_20/best_model_{signal}.pt
├── checkpoints/epochs_200/best_model_{signal}.pt
├── outputs/epochs_20/{test_metrics.json, *.npy, *.png}
└── outputs/epochs_200/{test_metrics.json, *.npy, *.png}
```

## Signals
- **ABP** (index 0) — Arterial Blood Pressure (mmHg)
- **PLETH** (index 1) — Photoplethysmogram (a.u.)
- **II** (index 2) — ECG Lead II (mV)

## Data
- Source: PhysioNet mimic3wdb-matched/1.0 (public, no credentials)
- 2 patients: p000160 (segment 3531764_0003), p000188 (segment 3285727_0007)
- 10 minutes per patient, offset 90s from segment start (to avoid NaN region)
- 150,000 total samples (75,000 per patient × 3 signals)
- Sampling rate: 125 Hz
- download_data.py caches to data/ so repeated experiments skip download

## Pipeline (same for both lstm/ and diffusion/)
```
python download_data.py            # Download + cache waveforms
python model.py --epochs N         # Train (saves best ckpt at min val loss)
python test.py --epochs N          # Test with saved checkpoint, saves .npy
python plot_predictions.py --epochs N  # Visualize predictions vs ground truth
python run_pipeline.py --epochs N  # Orchestrates all steps
```
All scripts accept `--epochs N` to specify which epoch-specific directory to use.
Default is 20 epochs if not specified.

## Architecture
- **LSTM:** 2-layer LSTM, input_size=3, hidden=64, output=25. ~52K params/signal.
- **Diffusion:** Conditional DDPM. LSTM condition encoder (125×3 → vector) + MLP denoiser. T=200 steps, linear beta [1e-4, 0.02]. ~349K params/signal.
- Both train 3 separate models (one per target signal), all using 3 signals as input.

## Data Split (Contiguous Temporal — No Data Leakage)
Each patient's 75,000-sample time series is split **chronologically before windowing**:
- First 70% (52,500 samples) → Train
- Next 15% (11,250 samples) → Validation
- Last 15% (11,250 samples) → Test

Sliding windows (stride=1) created independently within each contiguous block:
- Train: 52,351 windows/patient → 104,702 total
- Val: 11,101 windows/patient → 22,202 total
- Test: 11,101 windows/patient → 22,202 total

**Data leakage prevention:**
1. Temporal isolation — no window spans across split boundaries (zero sample overlap)
2. Normalization isolation — z-score mean/std computed from training blocks only, then applied to val/test

## Results (20 epochs)
| Signal | LSTM MAE | LSTM RMSE | Diffusion MAE | Diffusion RMSE |
|--------|----------|-----------|---------------|----------------|
| ABP    | 2.44 mmHg | 4.15 mmHg | 3.30 mmHg   | 10.09 mmHg     |
| PLETH  | 0.078    | 0.151     | 0.128         | 0.291          |
| II     | 0.022 mV | 0.046 mV | 0.036 mV     | 0.076 mV       |

## Results (100 epochs, Diffusion)
| Signal | MAE (20 ep) | MAE (100 ep) | RMSE (100 ep) | Improvement |
|--------|-------------|--------------|---------------|-------------|
| ABP    | 3.30 mmHg   | 3.21 mmHg    | 10.79 mmHg    | 3%          |
| PLETH  | 0.128       | 0.130        | 0.285         | -2%         |
| II     | 0.036 mV    | 0.033 mV     | 0.067 mV      | 8%          |

Best checkpoints: ABP epoch 46, PLETH epoch 71, II epoch 29.

## Diffusion Sampling Stability Fix (CRITICAL)
The DDPM reverse process (`p_sample`) explodes to infinity without output clipping.
A well-trained model (200 epochs) produces sharper noise predictions that, when divided
by `sqrt(alpha_bar_t)` at high timesteps, amplify errors exponentially across 200 steps.

**Symptoms:** MAE of 10^17+ despite good training loss.

**Fix in `diffusion/model.py`:**
1. Predict x₀ explicitly and clip to ±6.0 (z-normalized data range)
2. Recompute posterior mean from clipped x₀ using proper DDPM posterior formula
3. Gradient clipping (`clip_grad_norm_ max_norm=1.0`) during training

Without this fix, 20-epoch model stays stable by accident; 200-epoch model explodes.

## SLURM
- LSTM training: `sbatch lstm/main.sh` (20 epochs, L40S GPU, gl40s_short partition)
- Diffusion training: `sbatch diffusion/main.sh` (20 epochs, L40S GPU, gl40s_short partition)
- Conda env: CSDI (Python 3.11, PyTorch 2.13+cu126)
- Conda activation: `source /gpfs/share/apps/anaconda3/gpu/2023.09/etc/profile.d/conda.sh && conda activate CSDI`
- Use `python -u` for unbuffered logging

## Git
- Remote: https://github.com/donghyunkm/forecasting.git
- Branch: main
- .gitignore excludes: data/, checkpoints/, outputs/, logs/, __pycache__/

## Key Design Decisions
- Checkpoint saved at minimum validation loss (not final epoch)
- test.py is standalone — loads checkpoint independently, saves predictions/targets as .npy
- All metrics saved to outputs/epochs_N/test_metrics.json (both normalized and original scale)
- download_data.py tries multiple offsets (90s, 300s, 600s...) to find NaN-free regions
- 10 candidate patients available as fallbacks if primary patients have issues
- Epoch-specific output directories prevent overwriting between runs
- `get_checkpoint_dir(num_epochs)` and `get_output_dir(num_epochs)` in model.py control paths

## Potential Improvements (Diffusion)
- Cosine noise schedule (instead of linear)
- DDIM sampling (faster inference, no retraining needed)
- EMA model weights
- 1D U-Net denoiser with skip connections
- Cross-attention conditioning
- Classifier-free guidance
- More epochs (200+) and more data (more patients)
- Structured State Space (S4/Mamba) backbone

## Notes
- Diffusion models need more training than LSTM to converge (100-200+ epochs)
- All scripts (model.py, test.py, plot_predictions.py, run_pipeline.py) accept --epochs
- ABP has highest absolute error but all signals show good relative performance
- Training instability spikes observed at epochs ~122, ~153, ~162 — gradient clipping helps
