#!/usr/bin/env python3
"""
plot_predictions.py - Generate forecast visualizations for Phase 4.

Uses visualization.py (from TFT-multi) display_target_trajectory function
to create plots with confidence bands matching their paper figures.

Also generates:
  - Per-vital scatter plots (predicted vs actual)
  - Error by forecast step
  - Summary bar chart

Usage:
    python plot_predictions.py --epochs 100
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from preprocess import PAST_MONTHS, FUTURE_MONTHS, SIGNAL_NAMES, NUM_SIGNALS
from train import get_output_dir, NUM_EPOCHS, OUTPUT_QUANTILES
import visualization as vis


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def plot_forecast_samples(predictions, targets, masks, output_dir, num_samples=6):
    """
    Plot sample forecasts using TFT-multi's display_target_trajectory.

    For each vital sign, shows observed history + predicted quantiles + true future.
    """
    # We need historical data to show — create synthetic "history" from targets context
    # Since we only saved future predictions, create time axis accordingly
    time_future = np.arange(1, FUTURE_MONTHS + 1)

    for sample_i in range(min(num_samples, len(predictions))):
        idx = sample_i * (len(predictions) // max(num_samples, 1))
        if idx >= len(predictions):
            break

        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        axes_flat = axes.flatten()

        for meas in range(NUM_SIGNALS):
            ax = axes_flat[meas]

            pred_meas = predictions[idx, :, meas, :]  # (25, 3)
            true_meas = targets[idx, :, meas]          # (25,)
            mask_meas = masks[idx, :, meas]            # (25,)

            # Plot true values
            ax.plot(time_future, true_meas, 'o-', color='tab:blue', lw=2,
                    label='Target', markersize=4)

            # Plot quantile predictions
            colors = ['tab:green', 'tab:orange', 'tab:red']
            for q_idx, q in enumerate(OUTPUT_QUANTILES):
                ax.plot(time_future, pred_meas[:, q_idx], '--s', color=colors[q_idx],
                        lw=1.5, markersize=3, label=f'Pred Q={q}')

            # Confidence band (10th-90th)
            ax.fill_between(time_future, pred_meas[:, 0], pred_meas[:, 2],
                            color='gray', alpha=0.2)

            # Mark imputed values
            imputed = mask_meas == 0
            if imputed.any():
                ax.scatter(time_future[imputed], true_meas[imputed],
                           marker='x', color='red', s=30, zorder=5, label='Imputed')

            ax.set_title(SIGNAL_NAMES[meas])
            ax.set_xlabel('Forecast Hour')
            ax.grid(True, alpha=0.3)
            if meas == 0:
                ax.legend(fontsize=8)

        # Hide unused subplot
        if NUM_SIGNALS < len(axes_flat):
            axes_flat[-1].axis('off')

        plt.suptitle(f'TFT-multi Forecast — Sample {sample_i + 1}', fontsize=14)
        plt.tight_layout()
        filepath = os.path.join(output_dir, f'plot_forecast_sample_{sample_i + 1}.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()

    print(f"[SAVED] {num_samples} forecast sample plots")


def plot_scatter_per_vital(predictions, targets, masks, output_dir):
    """Predicted vs actual scatter for 50th percentile."""
    median_idx = OUTPUT_QUANTILES.index(0.5)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes_flat = axes.flatten()

    for s, name in enumerate(SIGNAL_NAMES):
        ax = axes_flat[s]
        v_mask = masks[:, :, s].astype(bool) & (targets[:, :, s] > 0)
        pred_s = predictions[:, :, s, median_idx][v_mask]
        tgt_s = targets[:, :, s][v_mask]

        ax.scatter(tgt_s, pred_s, alpha=0.1, s=5, color='tab:blue')

        # Identity line
        lims = [min(tgt_s.min(), pred_s.min()), max(tgt_s.max(), pred_s.max())]
        ax.plot(lims, lims, 'r--', lw=1, alpha=0.8)

        # Stats
        if len(pred_s) > 1 and np.std(pred_s) > 0:
            corr = np.corrcoef(pred_s, tgt_s)[0, 1]
            mae = np.mean(np.abs(pred_s - tgt_s))
            ax.set_title(f'{name}\nr={corr:.3f}, MAE={mae:.2f}')
        else:
            ax.set_title(name)

        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted (Q=0.5)')
        ax.grid(True, alpha=0.3)

    axes_flat[-1].axis('off')
    plt.suptitle('TFT-multi — Predicted vs Actual (Median)', fontsize=14)
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'plot_scatter_per_vital.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {filepath}")


def plot_error_by_step(predictions, targets, masks, output_dir):
    """MAE by forecast hour for each vital sign."""
    median_idx = OUTPUT_QUANTILES.index(0.5)

    fig, ax = plt.subplots(figsize=(12, 6))
    hours = np.arange(1, FUTURE_MONTHS + 1)

    for s, name in enumerate(SIGNAL_NAMES):
        v_mask = masks[:, :, s].astype(bool) & (targets[:, :, s] > 0)
        per_step_mae = np.zeros(FUTURE_MONTHS)
        for t in range(FUTURE_MONTHS):
            step_mask = v_mask[:, t]
            if step_mask.any():
                per_step_mae[t] = np.mean(
                    np.abs(predictions[:, t, s, median_idx][step_mask] -
                           targets[:, t, s][step_mask]))

        ax.plot(hours, per_step_mae, '-o', markersize=4, label=name)

    ax.set_xlabel('Forecast Hour')
    ax.set_ylabel('MAE')
    ax.set_title('TFT-multi — MAE by Forecast Horizon')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'plot_error_by_step.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {filepath}")


def plot_calibration_summary(predictions, targets, masks, output_dir):
    """Per-vital calibration bar chart."""
    fig, ax = plt.subplots(figsize=(10, 5))

    calibrations = []
    for s, name in enumerate(SIGNAL_NAMES):
        v_mask = masks[:, :, s].astype(bool) & (targets[:, :, s] > 0)
        pred_q10 = predictions[:, :, s, 0][v_mask]
        pred_q90 = predictions[:, :, s, 2][v_mask]
        tgt = targets[:, :, s][v_mask]
        within = (tgt >= pred_q10) & (tgt <= pred_q90)
        calib = within.mean() if len(within) > 0 else 0
        calibrations.append(calib)

    bars = ax.bar(SIGNAL_NAMES, calibrations, color='tab:blue', alpha=0.7)
    ax.axhline(y=0.8, color='red', linestyle='--', label='Target (80%)')
    ax.set_ylabel('Calibration (% within 10th-90th)')
    ax.set_title('TFT-multi — Prediction Interval Calibration')
    ax.set_ylim(0, 1.0)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, val in zip(bars, calibrations):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f'{val:.1%}', ha='center', fontsize=10)

    plt.tight_layout()
    filepath = os.path.join(output_dir, 'plot_calibration_summary.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {filepath}")


def main(num_epochs=None):
    """Generate all visualization plots."""
    epochs = num_epochs if num_epochs is not None else NUM_EPOCHS
    output_dir = get_output_dir(epochs)

    # Load saved predictions
    pred_path = os.path.join(output_dir, 'test_predictions.npy')
    tgt_path = os.path.join(output_dir, 'test_targets.npy')
    mask_path = os.path.join(output_dir, 'test_masks.npy')

    if not os.path.exists(pred_path):
        raise FileNotFoundError(
            f"Predictions not found: {pred_path}\nRun test.py first.")

    predictions = np.load(pred_path)    # (N, 25, 5, 3)
    targets = np.load(tgt_path)          # (N, 25, 5)
    masks = np.load(mask_path)           # (N, 25, 5)

    print(f"[INFO] Loaded predictions: {predictions.shape}")
    print(f"[INFO] Loaded targets: {targets.shape}")
    print(f"[INFO] Vital signs: {SIGNAL_NAMES}")

    # Generate all plots
    plot_forecast_samples(predictions, targets, masks, output_dir)
    plot_scatter_per_vital(predictions, targets, masks, output_dir)
    plot_error_by_step(predictions, targets, masks, output_dir)
    plot_calibration_summary(predictions, targets, masks, output_dir)

    print(f"\n[INFO] All plots saved to: {output_dir}/")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Plot TFT-multi results')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS)
    args = parser.parse_args()
    main(num_epochs=args.epochs)
