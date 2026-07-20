#!/usr/bin/env python3
"""
plot_predictions.py - Visualization for diffusion waveform forecasting (Phase 3).

Generates plots comparing predicted vs ground truth for all 4 aggregated
features of the target signal.

Usage:
    python plot_predictions.py --target II
    python plot_predictions.py --target PLETH --epochs 100
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from preprocess import (INPUT_LENGTH, OUTPUT_LENGTH, SIGNAL_NAMES, VALID_TARGETS,
                        INTERVAL_MINUTES, NUM_FEATURES, FEATURE_NAMES)
from model import get_output_dir, NUM_EPOCHS


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def plot_time_series(predictions, targets, target_signal, output_dir, num_samples=3):
    """Plot individual forecast examples showing all 4 features."""
    time_axis = np.arange(OUTPUT_LENGTH) * INTERVAL_MINUTES / 60

    for sample_i in range(num_samples):
        idx = sample_i * (len(predictions) // num_samples)
        if idx >= len(predictions):
            break

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes_flat = axes.flatten()

        for f_idx, (ax, fname) in enumerate(zip(axes_flat, FEATURE_NAMES)):
            ax.plot(time_axis, targets[idx, :, f_idx], 'b-', linewidth=1.5,
                    label='Ground Truth', alpha=0.8)
            ax.plot(time_axis, predictions[idx, :, f_idx], 'r--', linewidth=1.5,
                    label='Predicted', alpha=0.8)
            ax.set_ylabel(fname)
            ax.legend(loc='upper right', fontsize=7)
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('Forecast Horizon (hours)')

        plt.suptitle(f'Phase 3 Diffusion — {target_signal} Forecast (Sample {idx})', fontsize=13)
        plt.tight_layout()

        filepath = os.path.join(output_dir, f'plot_forecast_sample_{sample_i}.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[SAVED] {filepath}")


def plot_scatter_per_feature(predictions, targets, target_signal, output_dir):
    """Scatter plots of predicted vs actual for each feature."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes_flat = axes.flatten()

    for f_idx, (ax, fname) in enumerate(zip(axes_flat, FEATURE_NAMES)):
        pred_f = predictions[:, :, f_idx].flatten()
        tgt_f = targets[:, :, f_idx].flatten()

        ax.scatter(tgt_f, pred_f, alpha=0.05, s=3, c='tab:blue')
        lims = [min(tgt_f.min(), pred_f.min()), max(tgt_f.max(), pred_f.max())]
        ax.plot(lims, lims, 'r-', linewidth=1)
        ax.set_xlabel(f'Actual')
        ax.set_ylabel(f'Predicted')
        ax.set_title(f'{fname}')
        ax.grid(True, alpha=0.3)

        corr = np.corrcoef(tgt_f, pred_f)[0, 1]
        ax.text(0.05, 0.95, f'r={corr:.3f}', transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle(f'Phase 3 Diffusion — {target_signal} Predicted vs Actual (per feature)', fontsize=13)
    plt.tight_layout()

    filepath = os.path.join(output_dir, 'plot_scatter_per_feature.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {filepath}")


def plot_error_by_step(predictions, targets, target_signal, output_dir):
    """Plot MAE per feature as a function of forecast step."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes_flat = axes.flatten()
    steps = np.arange(1, OUTPUT_LENGTH + 1)
    hours = steps * INTERVAL_MINUTES / 60

    for f_idx, (ax, fname) in enumerate(zip(axes_flat, FEATURE_NAMES)):
        per_step_mae = np.mean(np.abs(predictions[:, :, f_idx] - targets[:, :, f_idx]), axis=0)
        per_step_rmse = np.sqrt(np.mean((predictions[:, :, f_idx] - targets[:, :, f_idx]) ** 2, axis=0))

        ax.plot(hours, per_step_mae, 'b-o', markersize=3, label='MAE', alpha=0.8)
        ax.plot(hours, per_step_rmse, 'r-s', markersize=3, label='RMSE', alpha=0.8)
        ax.set_title(fname)
        ax.set_xlabel('Forecast Horizon (hours)')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'Phase 3 Diffusion — {target_signal} Error vs Forecast Horizon', fontsize=13)
    plt.tight_layout()

    filepath = os.path.join(output_dir, 'plot_error_by_step.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {filepath}")


def plot_error_summary(predictions, targets, target_signal, output_dir):
    """Bar chart comparing MAE/RMSE/correlation across features."""
    maes = []
    rmses = []
    corrs = []
    for f_idx in range(NUM_FEATURES):
        pred_f = predictions[:, :, f_idx].flatten()
        tgt_f = targets[:, :, f_idx].flatten()
        maes.append(np.mean(np.abs(pred_f - tgt_f)))
        rmses.append(np.sqrt(np.mean((pred_f - tgt_f) ** 2)))
        corrs.append(np.corrcoef(pred_f, tgt_f)[0, 1])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    x = np.arange(NUM_FEATURES)
    axes[0].bar(x, maes, color='tab:blue', alpha=0.7)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(FEATURE_NAMES, rotation=30)
    axes[0].set_ylabel('MAE')
    axes[0].set_title('MAE per Feature')
    axes[0].grid(True, alpha=0.3, axis='y')

    axes[1].bar(x, rmses, color='tab:red', alpha=0.7)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(FEATURE_NAMES, rotation=30)
    axes[1].set_ylabel('RMSE')
    axes[1].set_title('RMSE per Feature')
    axes[1].grid(True, alpha=0.3, axis='y')

    axes[2].bar(x, corrs, color='tab:green', alpha=0.7)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(FEATURE_NAMES, rotation=30)
    axes[2].set_ylabel('Correlation (r)')
    axes[2].set_title('Correlation per Feature')
    axes[2].set_ylim(0, 1)
    axes[2].grid(True, alpha=0.3, axis='y')

    plt.suptitle(f'Phase 3 Diffusion — {target_signal} Performance Summary', fontsize=13)
    plt.tight_layout()

    filepath = os.path.join(output_dir, 'plot_feature_summary.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {filepath}")


def main(target_signal=None, num_epochs=None):
    """Generate all visualization plots."""
    target = target_signal if target_signal is not None else 'II'
    epochs = num_epochs if num_epochs is not None else NUM_EPOCHS
    output_dir = get_output_dir(target, epochs)

    pred_path = os.path.join(output_dir, 'test_predictions.npy')
    tgt_path = os.path.join(output_dir, 'test_targets.npy')

    if not os.path.exists(pred_path):
        raise FileNotFoundError(
            f"Predictions not found: {pred_path}\n"
            "Run test.py first to generate predictions."
        )

    predictions = np.load(pred_path)  # (N, 25, 4)
    targets = np.load(tgt_path)       # (N, 25, 4)

    print(f"[INFO] Loaded predictions: {predictions.shape}")
    print(f"[INFO] Loaded targets: {targets.shape}")
    print(f"[INFO] Target signal: {target}")
    print(f"[INFO] Features: {FEATURE_NAMES}")

    plot_time_series(predictions, targets, target, output_dir)
    plot_scatter_per_feature(predictions, targets, target, output_dir)
    plot_error_by_step(predictions, targets, target, output_dir)
    plot_error_summary(predictions, targets, target, output_dir)

    print(f"\n[INFO] All plots saved to: {output_dir}/")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Plot diffusion forecasting results')
    parser.add_argument('--target', type=str, default='II', choices=VALID_TARGETS,
                        help='Target signal (default: II)')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS,
                        help=f'Epochs used during training (default: {NUM_EPOCHS})')
    args = parser.parse_args()
    main(target_signal=args.target, num_epochs=args.epochs)
