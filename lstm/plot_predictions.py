#!/usr/bin/env python3
"""
plot_predictions.py - Visualize test predictions vs ground truth for all 3 signals.

Loads saved predictions and targets from outputs/ directory and creates
plots showing prediction vs ground truth for ABP, PLETH, and II.

Usage:
    python plot_predictions.py
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model import get_output_dir, NUM_EPOCHS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLING_RATE = 125  # Hz
FORECAST_HORIZON = 25  # steps
SIGNAL_NAMES = ['ABP', 'PLETH', 'II']
SIGNAL_UNITS = {'ABP': 'mmHg', 'PLETH': 'a.u.', 'II': 'mV'}
SIGNAL_COLORS = {'ABP': ('tab:blue', 'tab:red'),
                 'PLETH': ('tab:green', 'tab:orange'),
                 'II': ('tab:purple', 'tab:brown')}


def load_data(output_dir):
    """Load saved predictions and targets for all signals."""
    data = {}

    for name in SIGNAL_NAMES:
        pred_path = os.path.join(output_dir, f'test_predictions_{name.lower()}.npy')
        tgt_path = os.path.join(output_dir, f'test_targets_{name.lower()}.npy')

        if not os.path.exists(pred_path) or not os.path.exists(tgt_path):
            raise FileNotFoundError(
                f"Predictions/targets for {name} not found.\n"
                "Run test.py first to generate them."
            )

        predictions = np.load(pred_path)
        targets = np.load(tgt_path)
        data[name] = {'predictions': predictions, 'targets': targets}
        print(f"[LOADED] {name}: predictions {predictions.shape}, targets {targets.shape}")

    # Load metrics if available
    metrics_path = os.path.join(output_dir, 'test_metrics.json')
    metrics = None
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)

    return data, metrics


def plot_overlay_samples(data, metrics, output_dir, num_samples=4):
    """
    Plot individual forecast windows for all 3 signals side by side.
    Each row is a different sample, each column is a signal.
    """
    time_ms = np.arange(FORECAST_HORIZON) / SAMPLING_RATE * 1000

    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 3 * num_samples))

    # Pick evenly spaced samples
    n_total = len(data[SIGNAL_NAMES[0]]['predictions'])
    indices = np.linspace(0, n_total - 1, num_samples, dtype=int)

    for col, signal_name in enumerate(SIGNAL_NAMES):
        predictions = data[signal_name]['predictions']
        targets = data[signal_name]['targets']
        gt_color, pred_color = SIGNAL_COLORS[signal_name]
        unit = SIGNAL_UNITS[signal_name]

        for row, idx in enumerate(indices):
            ax = axes[row, col]
            ax.plot(time_ms, targets[idx], '-', color=gt_color,
                    linewidth=2.0, label='Ground Truth', alpha=0.9)
            ax.plot(time_ms, predictions[idx], '--', color=pred_color,
                    linewidth=2.0, label='Prediction', alpha=0.9)
            ax.set_xlim([0, time_ms[-1]])
            ax.grid(True, alpha=0.3, linestyle='--')

            if row == 0:
                mae = metrics[signal_name]['mae_raw'] if metrics else 0
                ax.set_title(f'{signal_name} (MAE={mae:.2f} {unit})',
                             fontsize=11, fontweight='bold')
            if row == num_samples - 1:
                ax.set_xlabel('Time (ms)')
            if col == 0:
                ax.set_ylabel(f'Sample #{idx}\n({unit})', fontsize=9)
            if row == 0 and col == 2:
                ax.legend(loc='upper right', fontsize=8)

    plt.suptitle('Prediction vs Ground Truth — All Signals',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()

    filepath = os.path.join(output_dir, 'plot_overlay_samples.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {filepath}")


def plot_continuous_waveform(data, metrics, output_dir, num_windows=30):
    """
    Concatenate consecutive predictions to show longer waveforms.
    One subplot per signal.
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)

    for i, signal_name in enumerate(SIGNAL_NAMES):
        predictions = data[signal_name]['predictions']
        targets = data[signal_name]['targets']
        gt_color, pred_color = SIGNAL_COLORS[signal_name]
        unit = SIGNAL_UNITS[signal_name]

        n = min(num_windows, len(predictions))
        pred_concat = predictions[:n].flatten()
        tgt_concat = targets[:n].flatten()
        time_s = np.arange(len(pred_concat)) / SAMPLING_RATE

        ax = axes[i]
        ax.plot(time_s, tgt_concat, '-', color=gt_color,
                linewidth=1.2, label='Ground Truth', alpha=0.85)
        ax.plot(time_s, pred_concat, '-', color=pred_color,
                linewidth=1.2, label='Prediction', alpha=0.75)

        mae = metrics[signal_name]['mae_raw'] if metrics else 0
        ax.set_ylabel(f'{signal_name}\n({unit})', fontsize=10)
        ax.set_title(f'{signal_name} — MAE: {mae:.2f} {unit}',
                     fontsize=11, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3, linestyle='--')

    axes[-1].set_xlabel('Time (s)', fontsize=11)
    plt.suptitle(f'Continuous Waveform — {num_windows} Forecast Windows',
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()

    filepath = os.path.join(output_dir, 'plot_continuous_waveform.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {filepath}")


def plot_error_analysis(data, metrics, output_dir):
    """
    Error analysis for all 3 signals: MAE per forecast step.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    step_time_ms = (np.arange(1, FORECAST_HORIZON + 1)) / SAMPLING_RATE * 1000

    for i, signal_name in enumerate(SIGNAL_NAMES):
        predictions = data[signal_name]['predictions']
        targets = data[signal_name]['targets']
        errors = predictions - targets
        unit = SIGNAL_UNITS[signal_name]

        mae_per_step = np.mean(np.abs(errors), axis=0)
        gt_color, _ = SIGNAL_COLORS[signal_name]

        ax = axes[i]
        ax.bar(step_time_ms, mae_per_step, width=5, color=gt_color,
               alpha=0.7, edgecolor='black', linewidth=0.3)
        ax.set_xlabel('Forecast Horizon (ms)', fontsize=10)
        ax.set_ylabel(f'MAE ({unit})', fontsize=10)
        ax.set_title(f'{signal_name}', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')

    plt.suptitle('MAE by Forecast Step — All Signals',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()

    filepath = os.path.join(output_dir, 'plot_error_analysis.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {filepath}")


def plot_scatter(data, metrics, output_dir):
    """Scatter plot of predicted vs actual for all 3 signals."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    rng = np.random.default_rng(42)

    for i, signal_name in enumerate(SIGNAL_NAMES):
        predictions = data[signal_name]['predictions']
        targets = data[signal_name]['targets']
        unit = SIGNAL_UNITS[signal_name]
        gt_color, _ = SIGNAL_COLORS[signal_name]

        # Subsample for clarity
        n_points = min(5000, predictions.size)
        idx = rng.choice(predictions.size, n_points, replace=False)
        pred_flat = predictions.flatten()[idx]
        tgt_flat = targets.flatten()[idx]

        ax = axes[i]
        ax.scatter(tgt_flat, pred_flat, alpha=0.15, s=6, color=gt_color)

        vmin = min(tgt_flat.min(), pred_flat.min())
        vmax = max(tgt_flat.max(), pred_flat.max())
        ax.plot([vmin, vmax], [vmin, vmax], 'k--', linewidth=1.5, label='Perfect')

        ax.set_xlabel(f'Actual ({unit})', fontsize=10)
        ax.set_ylabel(f'Predicted ({unit})', fontsize=10)
        ax.set_title(f'{signal_name}', fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_aspect('equal')
        ax.set_xlim([vmin, vmax])
        ax.set_ylim([vmin, vmax])

        if metrics:
            mae = metrics[signal_name]['mae_raw']
            rmse = metrics[signal_name]['rmse_raw']
            ax.text(0.05, 0.95, f"MAE: {mae:.2f}\nRMSE: {rmse:.2f}",
                    transform=ax.transAxes, fontsize=9,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle('Predicted vs Actual — All Signals',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()

    filepath = os.path.join(output_dir, 'plot_scatter.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {filepath}")


def main(num_epochs=None):
    """Generate all plots for a specific epoch run.
    
    Args:
        num_epochs: Number of epochs used during training (for directory lookup).
    """
    epochs = num_epochs if num_epochs is not None else NUM_EPOCHS
    output_dir = get_output_dir(epochs)

    print("=" * 60)
    print("Plotting Predictions vs Ground Truth — All 3 Signals")
    print("=" * 60)

    data, metrics = load_data(output_dir)

    if metrics:
        print("\n[INFO] Test metrics summary:")
        for name in SIGNAL_NAMES:
            m = metrics[name]
            print(f"  {name}: MAE={m['mae_raw']:.3f}, RMSE={m['rmse_raw']:.3f}")

    print("\n[INFO] Generating plots...")

    plot_overlay_samples(data, metrics, output_dir, num_samples=4)
    plot_continuous_waveform(data, metrics, output_dir, num_windows=30)
    plot_error_analysis(data, metrics, output_dir)
    plot_scatter(data, metrics, output_dir)

    print("\n" + "=" * 60)
    print("All plots saved to: " + output_dir)
    print("=" * 60)
    print("  - plot_overlay_samples.png     : Individual windows (3 signals x 4 samples)")
    print("  - plot_continuous_waveform.png  : Concatenated continuous signal")
    print("  - plot_error_analysis.png       : MAE per forecast step")
    print("  - plot_scatter.png              : Predicted vs actual scatter")
    print("=" * 60)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Plot LSTM predictions')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS,
                        help=f'Number of epochs used during training (default: {NUM_EPOCHS})')
    args = parser.parse_args()
    main(num_epochs=args.epochs)
