#!/usr/bin/env python3
"""
plot_predictions.py - Visualize heart rate predictions vs ground truth.

Loads saved predictions and targets from outputs/ directory and creates
plots for the heart rate prediction task.

Usage:
    python plot_predictions.py
    python plot_predictions.py --epochs 50 --input-length 7500 --target-length 7500
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model import get_output_dir, NUM_EPOCHS
from preprocess import INPUT_LENGTH, TARGET_LENGTH

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_data(output_dir):
    """Load saved predictions and targets."""
    pred_path = os.path.join(output_dir, 'test_predictions_hr.npy')
    tgt_path = os.path.join(output_dir, 'test_targets_hr.npy')

    if not os.path.exists(pred_path) or not os.path.exists(tgt_path):
        raise FileNotFoundError(
            "Predictions/targets not found.\n"
            "Run test.py first to generate them."
        )

    predictions = np.load(pred_path)
    targets = np.load(tgt_path)
    print(f"[LOADED] predictions: {predictions.shape}, targets: {targets.shape}")

    # Load metrics if available
    metrics_path = os.path.join(output_dir, 'test_metrics.json')
    metrics = None
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)

    return predictions, targets, metrics


def plot_time_series(predictions, targets, metrics, output_dir, num_points=500):
    """Plot predicted vs actual HR as a time series."""
    n = min(num_points, len(predictions))
    idx = np.arange(n)

    fig, ax = plt.subplots(1, 1, figsize=(14, 5))
    ax.plot(idx, targets[:n], '-', color='tab:blue', linewidth=1.2,
            label='Ground Truth', alpha=0.85)
    ax.plot(idx, predictions[:n], '-', color='tab:red', linewidth=1.2,
            label='Prediction', alpha=0.75)

    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Heart Rate (BPM)')
    title = 'Heart Rate — Predicted vs Ground Truth'
    if metrics:
        title += f" (MAE={metrics['mae_bpm']:.2f} BPM)"
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    filepath = os.path.join(output_dir, 'plot_hr_timeseries.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {filepath}")


def plot_scatter(predictions, targets, metrics, output_dir):
    """Scatter plot of predicted vs actual heart rate."""
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))

    rng = np.random.default_rng(42)
    n_points = min(5000, len(predictions))
    idx = rng.choice(len(predictions), n_points, replace=False)

    ax.scatter(targets[idx], predictions[idx], alpha=0.2, s=8, color='tab:blue')

    vmin = min(targets.min(), predictions.min())
    vmax = max(targets.max(), predictions.max())
    ax.plot([vmin, vmax], [vmin, vmax], 'k--', linewidth=1.5, label='Perfect')

    ax.set_xlabel('Actual HR (BPM)', fontsize=11)
    ax.set_ylabel('Predicted HR (BPM)', fontsize=11)
    ax.set_title('Heart Rate — Predicted vs Actual', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_aspect('equal')
    ax.set_xlim([vmin, vmax])
    ax.set_ylim([vmin, vmax])

    if metrics:
        ax.text(0.05, 0.95,
                f"MAE: {metrics['mae_bpm']:.2f} BPM\n"
                f"RMSE: {metrics['rmse_bpm']:.2f} BPM\n"
                f"±5 BPM: {metrics['within_5bpm_pct']:.1f}%",
                transform=ax.transAxes, fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    filepath = os.path.join(output_dir, 'plot_hr_scatter.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {filepath}")


def plot_error_distribution(predictions, targets, metrics, output_dir):
    """Histogram of prediction errors."""
    errors = predictions - targets

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    abs_errors = np.abs(errors)
    ax.hist(abs_errors, bins=50, color='tab:orange', alpha=0.7, edgecolor='black', linewidth=0.3)
    ax.axvline(x=5.0, color='red', linestyle='--', linewidth=1.5, label='5 BPM threshold')
    ax.axvline(x=10.0, color='darkred', linestyle='--', linewidth=1.5, label='10 BPM threshold')
    ax.set_xlabel('Absolute Error (BPM)')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Absolute Errors')
    ax.legend()
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')

    ax = axes[1]
    ax.hist(errors, bins=50, color='tab:green', alpha=0.7, edgecolor='black', linewidth=0.3)
    ax.axvline(x=0, color='black', linestyle='-', linewidth=1.0)
    ax.set_xlabel('Error (BPM) — Predicted - Actual')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Signed Errors')
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')

    if metrics:
        ax.text(0.95, 0.95,
                f"Mean error: {np.mean(errors):.2f}\nStd: {np.std(errors):.2f}",
                transform=ax.transAxes, fontsize=9,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('Heart Rate Prediction Error Analysis', fontsize=13, fontweight='bold')
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'plot_hr_errors.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {filepath}")


def plot_bland_altman(predictions, targets, metrics, output_dir):
    """Bland-Altman plot (agreement analysis)."""
    mean_vals = (predictions + targets) / 2.0
    diff_vals = predictions - targets

    mean_diff = np.mean(diff_vals)
    std_diff = np.std(diff_vals)
    upper_loa = mean_diff + 1.96 * std_diff
    lower_loa = mean_diff - 1.96 * std_diff

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    rng = np.random.default_rng(42)
    n_points = min(5000, len(predictions))
    idx = rng.choice(len(predictions), n_points, replace=False)

    ax.scatter(mean_vals[idx], diff_vals[idx], alpha=0.2, s=8, color='tab:blue')
    ax.axhline(y=mean_diff, color='red', linestyle='-', linewidth=1.5,
               label=f'Mean: {mean_diff:.2f}')
    ax.axhline(y=upper_loa, color='orange', linestyle='--', linewidth=1.2,
               label=f'+1.96 SD: {upper_loa:.2f}')
    ax.axhline(y=lower_loa, color='orange', linestyle='--', linewidth=1.2,
               label=f'-1.96 SD: {lower_loa:.2f}')

    ax.set_xlabel('Mean of Predicted and Actual HR (BPM)', fontsize=11)
    ax.set_ylabel('Difference (Predicted - Actual) (BPM)', fontsize=11)
    ax.set_title('Bland-Altman Plot — Heart Rate Prediction', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    filepath = os.path.join(output_dir, 'plot_hr_bland_altman.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {filepath}")


def main(num_epochs=None, input_length=None, target_length=None):
    """Generate all plots for a specific run.

    Args:
        num_epochs: Number of epochs used during training (for directory lookup).
        input_length: Input window length in samples.
        target_length: Target window length in samples.
    """
    epochs = num_epochs if num_epochs is not None else NUM_EPOCHS
    in_len = input_length if input_length is not None else INPUT_LENGTH
    tgt_len = target_length if target_length is not None else TARGET_LENGTH
    output_dir = get_output_dir(epochs, in_len, tgt_len)

    print("=" * 60)
    print("Plotting Heart Rate Predictions vs Ground Truth")
    print("=" * 60)

    predictions, targets, metrics = load_data(output_dir)

    if metrics:
        print(f"\n[INFO] Test metrics: MAE={metrics['mae_bpm']:.2f} BPM, "
              f"RMSE={metrics['rmse_bpm']:.2f} BPM, "
              f"±5 BPM: {metrics['within_5bpm_pct']:.1f}%")

    print("\n[INFO] Generating plots...")

    plot_time_series(predictions, targets, metrics, output_dir)
    plot_scatter(predictions, targets, metrics, output_dir)
    plot_error_distribution(predictions, targets, metrics, output_dir)
    plot_bland_altman(predictions, targets, metrics, output_dir)

    print("\n" + "=" * 60)
    print("All plots saved to: " + output_dir)
    print("=" * 60)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Plot heart rate predictions')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS,
                        help=f'Number of epochs used during training (default: {NUM_EPOCHS})')
    parser.add_argument('--input-length', type=int, default=INPUT_LENGTH,
                        help=f'Input window length in samples (default: {INPUT_LENGTH})')
    parser.add_argument('--target-length', type=int, default=TARGET_LENGTH,
                        help=f'Target window length in samples (default: {TARGET_LENGTH})')
    args = parser.parse_args()
    main(num_epochs=args.epochs, input_length=args.input_length, target_length=args.target_length)
