#!/usr/bin/env python3
"""
plot_full_forecast.py - Plot forecast samples showing FULL input window + predicted future.

Shows the complete picture: 6h of historical input followed by 2h forecast
with median prediction and 10th-90th confidence band.

Phase 5 TFT: 12-dim input (7 correlations + 4 vitals + 1 time), 5-min stride.

Usage:
    python plot_full_forecast.py --epochs 100 --num-samples 20
"""

import os
import json
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Constants
NUM_VITALS = 4
HISTORY_STEPS = 72
PRED_STEPS = 24
VITAL_NAMES = ['ABPMean', 'PULSE', 'SpO2', 'RESP']
VITAL_UNITS = ['mmHg', 'bpm', '%', 'br/min']
STEP_MINUTES = 5

DATA_DIR = "/gpfs/home/dk5565/forecasting/phase5/phase5_data/processed"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def main(num_epochs=100, num_samples=20):
    output_dir = os.path.join(BASE_DIR, 'outputs', f'tft_epochs_{num_epochs}')

    # Load predictions and targets (already denormalized by test.py)
    predictions = np.load(os.path.join(output_dir, 'test_predictions.npy'))  # (N, 24, 4, 3)
    targets = np.load(os.path.join(output_dir, 'test_targets.npy'))          # (N, 24, 4)

    # Load normalization params
    with open(os.path.join(DATA_DIR, 'norm_params.json')) as f:
        norm_data = json.load(f)
    means = np.array(norm_data['means'])  # (11,)
    stds = np.array(norm_data['stds'])    # (11,)
    vital_means = means[7:11]  # (4,)
    vital_stds = stds[7:11]    # (4,)

    # Load test data for historical input
    test_data = torch.load(os.path.join(DATA_DIR, 'test_data.pt'),
                           map_location='cpu', weights_only=False)
    historical = test_data['historical_ts_numeric'].numpy()  # (N, 72, 12)

    # Denormalize historical vitals (channels 7-10)
    hist_vitals_norm = historical[:, :, 7:11]  # (N, 72, 4)
    hist_denorm = hist_vitals_norm * vital_stds[None, None, :] + vital_means[None, None, :]

    print(f"[INFO] Loaded {len(predictions)} test samples")
    print(f"[INFO] Generating {num_samples} full forecast plots...")

    # Select evenly spaced samples
    indices = np.linspace(0, len(predictions) - 1, num_samples, dtype=int)

    # Time axes
    hist_hours = np.arange(HISTORY_STEPS) * STEP_MINUTES / 60.0
    pred_hours = (HISTORY_STEPS + np.arange(PRED_STEPS)) * STEP_MINUTES / 60.0
    boundary_hour = HISTORY_STEPS * STEP_MINUTES / 60.0

    for plot_i, idx in enumerate(indices):
        fig, axes = plt.subplots(NUM_VITALS, 1, figsize=(14, 12), sharex=True)

        for s in range(NUM_VITALS):
            ax = axes[s]

            # Historical
            hist_vals = hist_denorm[idx, :, s]

            # Future
            true_vals = targets[idx, :, s]
            pred_vals = predictions[idx, :, s, :]  # (24, 3)

            # Plot historical
            ax.plot(hist_hours, hist_vals, '-', color='tab:blue', lw=1.2,
                    alpha=0.8, label='Input (history)')

            # Plot actual future
            ax.plot(pred_hours, true_vals, 'o-', color='tab:blue', lw=1.5,
                    markersize=3, alpha=0.8, label='Actual (future)')

            # Plot predicted median
            ax.plot(pred_hours, pred_vals[:, 1], '-', color='tab:orange',
                    lw=2.5, label='Predicted (median)')

            # Confidence band
            ax.fill_between(pred_hours, pred_vals[:, 0], pred_vals[:, 2],
                           color='tab:orange', alpha=0.25, label='80% interval (q10-q90)')

            # Forecast boundary
            ax.axvline(x=boundary_hour, color='green', linestyle='--', lw=1.5, alpha=0.7)

            # Formatting
            ax.set_ylabel(f'{VITAL_NAMES[s]}\n({VITAL_UNITS[s]})', fontsize=11)
            ax.grid(True, alpha=0.3)
            if s == 0:
                ax.legend(loc='upper right', fontsize=9, ncol=2)

        axes[-1].set_xlabel('Time (hours)', fontsize=12)
        axes[0].set_title(
            f'TFT Phase 5 — Full Forecast (Sample {plot_i + 1})\n'
            f'Input: 0–{boundary_hour:.1f}h | Forecast: {boundary_hour:.1f}–{pred_hours[-1]:.1f}h | '
            f'12-dim input (7 corr + 4 vitals + time)',
            fontsize=13, fontweight='bold')
        plt.tight_layout()

        save_path = os.path.join(output_dir, f'plot_full_forecast_sample_{plot_i}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    print(f"[SAVED] {num_samples} plots to {output_dir}/plot_full_forecast_sample_*.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--num-samples', type=int, default=20)
    args = parser.parse_args()
    main(num_epochs=args.epochs, num_samples=args.num_samples)
