# Phase 4.1 — iTransformer: Vital Sign Forecasting from MIMIC-III Waveform Numerics

iTransformer (ICLR 2024) applied to the same vital sign forecasting task as the TFT model.

## Architecture

The iTransformer inverts the standard transformer:
- Each **variate** (channel) is treated as a token
- The full time series of each variate is linearly embedded into a d_model-dimensional token
- Self-attention is applied **across variates** to capture multivariate correlations
- FFN is applied per variate token to learn temporal representations
- A linear projection head maps each output variate token to its forecast

## Key Differences from TFT

| Aspect | TFT | iTransformer |
|--------|-----------|-------------|
| Token definition | Temporal (one token per timestep) | Variate (one token per channel) |
| Attention | Over time steps | Over variates |
| Temporal modeling | LSTM + attention | Linear embedding of full series |
| Architecture | Complex (VSN, GRN, static encoders) | Simple (standard transformer blocks) |
| Parameters | ~7.3M | ~1.5M |
| Input handling | Separate past/future paths | Single lookback window |

## Configuration

- d_model: 256
- n_heads: 4
- d_ff: 512
- n_layers: 3
- dropout: 0.1
- Optimizer: Adam, LR=1e-4
- Gradient clip: 1.0
- Same quantile loss with masking as TFT

## Data

Uses the same pre-processed data as TFT:
- Location: /gpfs/scratch/dk5565/phase41_data/processed/
- Input: 75 steps × 9 features (4 vitals + 4 masks + 1 time)
- Output: 25 steps × 4 vitals × 3 quantiles

## Usage

```bash
# Train + evaluate
sbatch train.sh

# Or manually:
python train.py --epochs 100
python test.py --epochs 100
python plot_predictions.py --epochs 100
```

## Reference

Liu et al., "iTransformer: Inverted Transformers Are Effective for Time Series Forecasting", ICLR 2024.
https://arxiv.org/abs/2310.06625
