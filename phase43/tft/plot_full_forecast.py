#!/usr/bin/env python3
"""
plot_full_forecast.py - Plot forecast samples showing FULL input window + predicted future.

Phase 4.3: Multi-horizon (1h/3h/6h). Shows the complete picture: historical input
followed by forecast with median prediction and 10th-90th confidence band.

Usage:
    python plot_full_forecast.py --epochs 100 --horizon 6h --num-samples 5
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from preprocess import PAST_MONTHS, HORIZON_MAP, SIGNAL_NAMES, NUM_SIGNALS

# Constants
SIGNAL_UNITS = ['mmHg', 'bpm', '%', 'breaths/min']
OUTPUT_QUANTILES = [0.1, 0.5, 0.9]
STEPS_PER_HOUR = 4  # 15-min resolution

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def steps_to_hours(steps):
    return steps / STEPS_PER_HOUR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--horizon', type=str, choices=['1h', '3h', '6h'], default='6h')
    parser.add_argument('--num-samples', type=int, default=5)
    args = parser.parse_args()

    horizon = args.horizon
    num_epochs = args.epochs
    num_samples = args.num_samples
    FUTURE_MONTHS = HORIZON_MAP[horizon]
    pred_len = FUTURE_MONTHS

    output_dir = os.path.join(BASE_DIR, 'outputs', f'tft_{horizon}_epochs_{num_epochs}')
    PROCESSED_DIR = f'/gpfs/scratch/dk5565/phase43_data/processed/horizon_{pred_len}'

    # Load predictions, targets, masks
    predictions = np.load(os.path.join(output_dir, 'test_predictions.npy'))
    targets = np.load(os.path.join(output_dir, 'test_targets.npy'))
    masks = np.load(os.path.join(output_dir, 'test_masks.npy'))

    # Load normalization params
    with open(os.path.join(PROCESSED_DIR, 'norm_params.json')) as f:
        norm_data = json.load(f)
    norm_mean = np.array(norm_data['mean'])
    norm_std = np.array(norm_data['std'])

    # Load test data to get historical input
    test_data = torch.load(os.path.join(PROCESSED_DIR, 'test_data.pt'), weights_only=False)
    historical = test_data['historical_ts_numeric'].numpy()  # (N, PAST_MONTHS, 5)
    # First 4 channels are the normalized vitals (5th is time)
    hist_vitals = historical[:, :, :4]  # (N, PAST_MONTHS, 4)

    # Denormalize historical (still normalized in .pt file)
    hist_denorm = hist_vitals * norm_std + norm_mean  # (N, PAST_MONTHS, 4)

    # Predictions and targets are ALREADY denormalized
    targets_denorm = targets
    preds_denorm = predictions

    print(f"[INFO] Loaded {len(predictions)} test samples")
    print(f"[INFO] Generating {num_samples} full forecast plots...")

    # Select evenly spaced samples
    indices = np.linspace(0, len(predictions) - 1, num_samples, dtype=int)

    for plot_i, idx in enumerate(indices):
        fig, axes = plt.subplots(NUM_SIGNALS, 1, figsize=(14, 12), sharex=True)

        for s in range(NUM_SIGNALS):
            ax = axes[s]

            # Historical
            hist_hours = steps_to_hours(np.arange(PAST_MONTHS))
            hist_vals = hist_denorm[idx, :, s]

            # Future
            future_hours = steps_to_hours(np.arange(PAST_MONTHS, PAST_MONTHS + FUTURE_MONTHS))
            true_vals = targets_denorm[idx, :, s]
            pred_vals = preds_denorm[idx, :, s, :]  # (FUTURE_MONTHS, 3)
            fut_mask = masks[idx, :, s]

            # Plot historical
            ax.plot(hist_hours, hist_vals, '-', color='tab:blue', lw=1.2, alpha=0.8, label='Input (history)')

            # Plot actual future
            ax.plot(future_hours, true_vals, 'o-', color='tab:blue', lw=1.5,
                    markersize=3, alpha=0.8, label='Actual (future)')

            # Plot predicted median
            ax.plot(future_hours, pred_vals[:, 1], '-', color='tab:orange',
                    lw=2.5, label='Predicted (median)')

            # Confidence band
            ax.fill_between(future_hours, pred_vals[:, 0], pred_vals[:, 2],
                           color='tab:orange', alpha=0.25, label='80% interval (q10-q90)')

            # Forecast boundary
            boundary_hour = PAST_MONTHS / STEPS_PER_HOUR
            ax.axvline(x=boundary_hour, color='green', linestyle='--', lw=1.5, alpha=0.7)

            # Formatting
            ax.set_ylabel(f'{SIGNAL_NAMES[s]}\n({SIGNAL_UNITS[s]})', fontsize=11)
            ax.grid(True, alpha=0.3)
            if s == 0:
                ax.legend(loc='upper right', fontsize=9, ncol=3)

        axes[-1].set_xlabel('Time (hours)', fontsize=12)
        axes[0].set_title(f'TFT Phase 4.3 \u2014 {horizon} Forecast (Sample {plot_i + 1})\n'
                         f'Input: 0\u2013{PAST_MONTHS/STEPS_PER_HOUR:.1f}h | '
                         f'Forecast: {PAST_MONTHS/STEPS_PER_HOUR:.1f}\u2013'
                         f'{(PAST_MONTHS+FUTURE_MONTHS)/STEPS_PER_HOUR:.1f}h',
                         fontsize=13, fontweight='bold')
        plt.tight_layout()

        save_path = os.path.join(output_dir, f'plot_full_forecast_sample_{plot_i}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    print(f"[SAVED] {num_samples} plots to {output_dir}/plot_full_forecast_sample_*.png")


if __name__ == '__main__':
    main()
