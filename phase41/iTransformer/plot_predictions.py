#!/usr/bin/env python3
"""
plot_predictions.py - Generate forecast visualizations for iTransformer Phase 41.

Adapted from TFT plot_predictions.py for iTransformer:
  - 4 signals: mean_bp, pulse, spo2, respiratory_rate
  - FUTURE_MONTHS = 25 steps (6.25 hours)
  - PAST_MONTHS = 75 steps (18.75 hours) for historical context
  - X-axis in hours (step index / 4)

Plots generated:
  1. plot_forecast_sample_{i}.png (i=0..4): 5 sample forecasts with 4 subplots
  2. plot_scatter_per_vital.png: Predicted vs actual scatter per vital
  3. plot_error_by_step.png: MAE by forecast step (x-axis in hours)
  4. plot_calibration_summary.png: Calibration bar chart per vital

Usage:
    python plot_predictions.py --epochs 100
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================================
# Constants
# ============================================================================
NUM_SIGNALS = 4
FUTURE_MONTHS = 25
PAST_MONTHS = 75
SIGNAL_NAMES = ['mean_bp', 'pulse', 'spo2', 'respiratory_rate']
OUTPUT_QUANTILES = [0.1, 0.5, 0.9]
NUM_EPOCHS = 100

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Steps per hour at 15-min resolution
STEPS_PER_HOUR = 4


def get_output_dir(num_epochs=NUM_EPOCHS):
    return os.path.join(BASE_DIR, 'outputs', f'iTransformer_epochs_{num_epochs}')


def steps_to_hours(steps):
    """Convert step indices to hours."""
    return np.array(steps) / STEPS_PER_HOUR


def plot_forecast_samples(predictions, targets, masks, output_dir,
                          num_samples=5):
    """
    Plot sample forecasts showing historical context + predicted future.

    For each sample:
      - 4 subplots (one per vital sign)
      - Historical context (75 steps = 18.75 hours)
      - Actual future (25 steps = 6.25 hours)
      - Predicted median + 10th-90th confidence interval
      - Vertical line separating history from forecast
      - X-axis in hours
    """
    # Load historical context if available
    hist_path = os.path.join(output_dir, 'test_historical.npy')
    has_history = os.path.exists(hist_path)
    if has_history:
        historical = np.load(hist_path)  # (N, 75, 4)
    else:
        historical = None

    for sample_i in range(min(num_samples, len(predictions))):
        idx = sample_i * (len(predictions) // max(num_samples, 1))
        if idx >= len(predictions):
            break

        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        axes_flat = axes.flatten()

        for meas in range(NUM_SIGNALS):
            ax = axes_flat[meas]

            pred_meas = predictions[idx, :, meas, :]  # (25, 3)
            true_meas = targets[idx, :, meas]          # (25,)
            mask_meas = masks[idx, :, meas]            # (25,)

            # Future time axis (starts at hour corresponding to PAST_MONTHS)
            future_steps = np.arange(PAST_MONTHS, PAST_MONTHS + FUTURE_MONTHS)
            future_hours = steps_to_hours(future_steps)

            # Plot historical context if available
            if historical is not None:
                hist_steps = np.arange(PAST_MONTHS)
                hist_hours = steps_to_hours(hist_steps)
                ax.plot(hist_hours, historical[idx, :, meas],
                        '-', color='tab:gray', lw=1, alpha=0.7,
                        label='History')

            # Plot true future values
            ax.plot(future_hours, true_meas, 'o-', color='tab:blue', lw=1.5,
                    markersize=2, label='Actual', alpha=0.8)

            # Plot predicted median
            ax.plot(future_hours, pred_meas[:, 1], '-', color='tab:orange',
                    lw=2, label='Pred (median)')

            # Confidence band (10th-90th)
            ax.fill_between(future_hours, pred_meas[:, 0], pred_meas[:, 2],
                            color='tab:orange', alpha=0.2,
                            label='10th-90th interval')

            # Mark imputed values
            imputed = mask_meas == 0
            if imputed.any():
                ax.scatter(future_hours[imputed], true_meas[imputed],
                           marker='x', color='red', s=20, zorder=5,
                           label='Imputed')

            # Vertical line at forecast boundary
            history_end_hour = PAST_MONTHS / STEPS_PER_HOUR  # 18.75
            ax.axvline(x=history_end_hour, color='green', linestyle='--',
                       lw=1.5, alpha=0.7, label='Forecast start')

            ax.set_title(SIGNAL_NAMES[meas], fontsize=12, fontweight='bold')
            ax.set_xlabel('Time (hours)')
            ax.set_ylabel('Value')
            ax.grid(True, alpha=0.3)
            if meas == 0:
                ax.legend(fontsize=8, loc='upper left')

        plt.suptitle(
            f'iTransformer Phase 41 — Forecast Sample {sample_i + 1}\n'
            f'(4 vitals, 15-min resolution, {FUTURE_MONTHS} steps = '
            f'{FUTURE_MONTHS / STEPS_PER_HOUR:.1f}h forecast)',
            fontsize=13)
        plt.tight_layout()
        filepath = os.path.join(output_dir,
                                f'plot_forecast_sample_{sample_i}.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()

    print(f"[SAVED] {num_samples} forecast sample plots "
          f"(plot_forecast_sample_0..{num_samples-1}.png)")


def plot_scatter_per_vital(predictions, targets, masks, output_dir):
    """Predicted vs actual scatter for 50th percentile (median)."""
    median_idx = OUTPUT_QUANTILES.index(0.5)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes_flat = axes.flatten()

    for s, name in enumerate(SIGNAL_NAMES):
        ax = axes_flat[s]
        v_mask = masks[:, :, s].astype(bool) & (targets[:, :, s] > 0)
        pred_s = predictions[:, :, s, median_idx][v_mask]
        tgt_s = targets[:, :, s][v_mask]

        ax.scatter(tgt_s, pred_s, alpha=0.1, s=5, color='tab:blue')

        # Identity line
        if len(pred_s) > 0 and len(tgt_s) > 0:
            lims = [min(tgt_s.min(), pred_s.min()),
                    max(tgt_s.max(), pred_s.max())]
            ax.plot(lims, lims, 'r--', lw=1.5, alpha=0.8)

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

    plt.suptitle('iTransformer Phase 41 — Predicted vs Actual (Median)',
                 fontsize=14)
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'plot_scatter_per_vital.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {filepath}")


def plot_error_by_step(predictions, targets, masks, output_dir):
    """MAE by forecast step for each vital sign (x-axis in hours)."""
    median_idx = OUTPUT_QUANTILES.index(0.5)

    fig, ax = plt.subplots(figsize=(12, 6))
    # X-axis: forecast steps converted to hours
    step_hours = steps_to_hours(np.arange(FUTURE_MONTHS))

    for s, name in enumerate(SIGNAL_NAMES):
        v_mask = masks[:, :, s].astype(bool) & (targets[:, :, s] > 0)
        per_step_mae = np.zeros(FUTURE_MONTHS)
        for t in range(FUTURE_MONTHS):
            step_mask = v_mask[:, t]
            if step_mask.any():
                per_step_mae[t] = np.mean(
                    np.abs(predictions[:, t, s, median_idx][step_mask] -
                           targets[:, t, s][step_mask]))

        ax.plot(step_hours, per_step_mae, '-o', markersize=3, label=name,
                alpha=0.8)

    ax.set_xlabel('Forecast Horizon (hours)')
    ax.set_ylabel('MAE')
    ax.set_title('iTransformer Phase 41 — MAE by Forecast Step (15-min resolution)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'plot_error_by_step.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {filepath}")


def plot_calibration_summary(predictions, targets, masks, output_dir):
    """Per-vital calibration bar chart (% within 10th-90th interval)."""
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
    ax.axhline(y=0.8, color='red', linestyle='--', lw=1.5,
               label='Target (80%)')
    ax.set_ylabel('Calibration (% within 10th-90th)')
    ax.set_title('iTransformer Phase 41 — Prediction Interval Calibration')
    ax.set_ylim(0, 1.0)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, val in zip(bars, calibrations):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f'{val:.1%}', ha='center', fontsize=11)

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

    predictions = np.load(pred_path)    # (N, 25, 4, 3)
    targets = np.load(tgt_path)          # (N, 25, 4)
    masks = np.load(mask_path)           # (N, 25, 4)

    print(f"[INFO] Loaded predictions: {predictions.shape}")
    print(f"[INFO] Loaded targets: {targets.shape}")
    print(f"[INFO] Vital signs: {SIGNAL_NAMES}")
    print(f"[INFO] Resolution: 15 min, forecast: {FUTURE_MONTHS} steps "
          f"= {FUTURE_MONTHS / STEPS_PER_HOUR:.1f} hours")

    # Generate all plots
    plot_forecast_samples(predictions, targets, masks, output_dir)
    plot_scatter_per_vital(predictions, targets, masks, output_dir)
    plot_error_by_step(predictions, targets, masks, output_dir)
    plot_calibration_summary(predictions, targets, masks, output_dir)

    print(f"\n[INFO] All plots saved to: {output_dir}/")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Plot iTransformer results (Phase 41 — 4 vitals, 15-min)')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS)
    args = parser.parse_args()
    main(num_epochs=args.epochs)
