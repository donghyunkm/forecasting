#!/usr/bin/env python3
"""
plot_full_forecast.py - Plot forecast samples showing FULL input window + predicted future.

Shows the complete picture: 18.75h of historical input followed by 6.25h forecast
with median prediction and 10th-90th confidence band.

Phase 4.2: Complete windows (no imputation), historical has 5 features (4 vitals + 1 time).

Usage:
    python plot_full_forecast.py --epochs 100 --num-samples 5
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Constants
NUM_SIGNALS = 4
PAST_MONTHS = 75
FUTURE_MONTHS = 25
SIGNAL_NAMES = ['mean_bp', 'pulse', 'spo2', 'respiratory_rate']
SIGNAL_UNITS = ['mmHg', 'bpm', '%', 'breaths/min']
OUTPUT_QUANTILES = [0.1, 0.5, 0.9]
STEPS_PER_HOUR = 4  # 15-min resolution

PROCESSED_DIR = '/gpfs/scratch/dk5565/phase42_data/processed'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def steps_to_hours(steps):
    return steps / STEPS_PER_HOUR


def main(num_epochs=100, num_samples=5):
    output_dir = os.path.join(BASE_DIR, 'outputs', f'tft_epochs_{num_epochs}')

    # Load predictions, targets, masks
    predictions = np.load(os.path.join(output_dir, 'test_predictions.npy'))  # (N, 25, 4, 3)
    targets = np.load(os.path.join(output_dir, 'test_targets.npy'))          # (N, 25, 4)
    masks = np.load(os.path.join(output_dir, 'test_masks.npy'))              # (N, 25, 4)

    # Load normalization params
    with open(os.path.join(PROCESSED_DIR, 'norm_params.json')) as f:
        norm_data = json.load(f)
    norm_mean = np.array(norm_data['mean'])
    norm_std = np.array(norm_data['std'])

    # Load test data to get historical input
    import torch
    test_data = torch.load(os.path.join(PROCESSED_DIR, 'test_data.pt'), weights_only=False)
    historical = test_data['historical_ts_numeric'].numpy()  # (N, 75, 5)
    # First 4 channels are the normalized vitals (5th is time)
    hist_vitals = historical[:, :, :4]  # (N, 75, 4)

    # Denormalize historical (still normalized in .pt file)
    hist_denorm = hist_vitals * norm_std + norm_mean  # (N, 75, 4)

    # Predictions and targets from test.py are ALREADY denormalized
    targets_denorm = targets   # (N, 25, 4)
    preds_denorm = predictions # (N, 25, 4, 3)

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
            pred_vals = preds_denorm[idx, :, s, :]  # (25, 3)
            fut_mask = masks[idx, :, s]

            # Plot historical (all data is real in Phase 4.2, no imputed values)
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
        axes[0].set_title(f'TFT Phase 4.2 — Full Forecast (Sample {plot_i + 1})\n'
                         f'Input: 0–{PAST_MONTHS/STEPS_PER_HOUR:.1f}h | '
                         f'Forecast: {PAST_MONTHS/STEPS_PER_HOUR:.1f}–{(PAST_MONTHS+FUTURE_MONTHS)/STEPS_PER_HOUR:.1f}h',
                         fontsize=13, fontweight='bold')
        plt.tight_layout()

        save_path = os.path.join(output_dir, f'plot_full_forecast_sample_{plot_i}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    print(f"[SAVED] {num_samples} plots to {output_dir}/plot_full_forecast_sample_*.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--num-samples', type=int, default=5)
    args = parser.parse_args()
    main(num_epochs=args.epochs, num_samples=args.num_samples)
