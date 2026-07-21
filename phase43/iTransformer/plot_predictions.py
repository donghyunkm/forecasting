#!/usr/bin/env python3
"""
plot_predictions.py - Generate forecast visualizations for iTransformer Phase 4.3.

Plots generated:
  1. plot_forecast_sample_{i}.png (i=0..4): 5 sample forecasts with 4 subplots
  2. plot_scatter_per_vital.png: Predicted vs actual scatter per vital
  3. plot_error_by_step.png: MAE by forecast step (x-axis in hours)
  4. plot_calibration_summary.png: Calibration bar chart per vital

Usage:
    python plot_predictions.py --epochs 100 --horizon 6h
"""

import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from preprocess import HORIZON_MAP

# ===========================================================================
# Constants
# ===========================================================================
NUM_SIGNALS = 4
PAST_MONTHS = 75
SIGNAL_NAMES = ['mean_bp', 'pulse', 'spo2', 'respiratory_rate']
OUTPUT_QUANTILES = [0.1, 0.5, 0.9]
NUM_EPOCHS = 100
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STEPS_PER_HOUR = 4


def get_output_dir(horizon, num_epochs=NUM_EPOCHS):
    return os.path.join(BASE_DIR, 'outputs', f'iTransformer_{horizon}_epochs_{num_epochs}')


def steps_to_hours(steps):
    return np.array(steps) / STEPS_PER_HOUR


def plot_forecast_samples(predictions, targets, masks, output_dir, future_months, num_samples=5):
    """Plot sample forecasts showing historical context + predicted future."""
    hist_path = os.path.join(output_dir, 'test_historical.npy')
    has_history = os.path.exists(hist_path)
    if has_history:
        historical = np.load(hist_path)
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
            pred_meas = predictions[idx, :, meas, :]
            true_meas = targets[idx, :, meas]
            mask_meas = masks[idx, :, meas]

            future_steps = np.arange(PAST_MONTHS, PAST_MONTHS + future_months)
            future_hours = steps_to_hours(future_steps)

            if historical is not None:
                hist_steps = np.arange(PAST_MONTHS)
                hist_hours = steps_to_hours(hist_steps)
                ax.plot(hist_hours, historical[idx, :, meas],
                        '-', color='tab:gray', lw=1, alpha=0.7, label='History')

            ax.plot(future_hours, true_meas, 'o-', color='tab:blue', lw=1.5,
                    markersize=2, label='Actual', alpha=0.8)
            ax.plot(future_hours, pred_meas[:, 1], '-', color='tab:orange',
                    lw=2, label='Pred (median)')
            ax.fill_between(future_hours, pred_meas[:, 0], pred_meas[:, 2],
                            color='tab:orange', alpha=0.2, label='10th-90th interval')

            imputed = mask_meas == 0
            if imputed.any():
                ax.scatter(future_hours[imputed], true_meas[imputed],
                           marker='x', color='red', s=20, zorder=5, label='Imputed')

            history_end_hour = PAST_MONTHS / STEPS_PER_HOUR
            ax.axvline(x=history_end_hour, color='green', linestyle='--',
                       lw=1.5, alpha=0.7, label='Forecast start')

            ax.set_title(SIGNAL_NAMES[meas], fontsize=12, fontweight='bold')
            ax.set_xlabel('Time (hours)')
            ax.set_ylabel('Value')
            ax.grid(True, alpha=0.3)
            if meas == 0:
                ax.legend(fontsize=8, loc='upper left')

        plt.suptitle(
            f'iTransformer Phase 4.3 \u2014 Forecast Sample {sample_i + 1}\n'
            f'(4 vitals, 15-min resolution, {future_months} steps = '
            f'{future_months / STEPS_PER_HOUR:.1f}h forecast)',
            fontsize=13)
        plt.tight_layout()
        filepath = os.path.join(output_dir, f'plot_forecast_sample_{sample_i}.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()

    print(f"[SAVED] {num_samples} forecast sample plots")


def plot_scatter_per_vital(predictions, targets, masks, output_dir, horizon):
    """Predicted vs actual scatter for median."""
    median_idx = OUTPUT_QUANTILES.index(0.5)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes_flat = axes.flatten()

    for s, name in enumerate(SIGNAL_NAMES):
        ax = axes_flat[s]
        v_mask = masks[:, :, s].astype(bool) & (targets[:, :, s] > 0)
        pred_s = predictions[:, :, s, median_idx][v_mask]
        tgt_s = targets[:, :, s][v_mask]

        ax.scatter(tgt_s, pred_s, alpha=0.1, s=5, color='tab:blue')
        if len(pred_s) > 0 and len(tgt_s) > 0:
            lims = [min(tgt_s.min(), pred_s.min()), max(tgt_s.max(), pred_s.max())]
            ax.plot(lims, lims, 'r--', lw=1.5, alpha=0.8)
        if len(pred_s) > 1 and np.std(pred_s) > 0:
            corr = np.corrcoef(pred_s, tgt_s)[0, 1]
            mae = np.mean(np.abs(pred_s - tgt_s))
            ax.set_title(f'{name}\nr={corr:.3f}, MAE={mae:.2f}')
        else:
            ax.set_title(name)
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted (Q=0.5)')
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'iTransformer Phase 4.3 ({horizon}) \u2014 Predicted vs Actual (Median)', fontsize=14)
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'plot_scatter_per_vital.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {filepath}")


def plot_error_by_step(predictions, targets, masks, output_dir, future_months, horizon):
    """MAE by forecast step for each vital sign."""
    median_idx = OUTPUT_QUANTILES.index(0.5)
    fig, ax = plt.subplots(figsize=(12, 6))
    step_hours = steps_to_hours(np.arange(future_months))

    for s, name in enumerate(SIGNAL_NAMES):
        v_mask = masks[:, :, s].astype(bool) & (targets[:, :, s] > 0)
        per_step_mae = np.zeros(future_months)
        for t in range(future_months):
            step_mask = v_mask[:, t]
            if step_mask.any():
                per_step_mae[t] = np.mean(
                    np.abs(predictions[:, t, s, median_idx][step_mask] -
                           targets[:, t, s][step_mask]))
        ax.plot(step_hours, per_step_mae, '-o', markersize=3, label=name, alpha=0.8)

    ax.set_xlabel('Forecast Horizon (hours)')
    ax.set_ylabel('MAE')
    ax.set_title(f'iTransformer Phase 4.3 ({horizon}) \u2014 MAE by Forecast Step (15-min resolution)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    filepath = os.path.join(output_dir, 'plot_error_by_step.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {filepath}")


def plot_calibration_summary(predictions, targets, masks, output_dir, horizon):
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
    ax.axhline(y=0.8, color='red', linestyle='--', lw=1.5, label='Target (80%)')
    ax.set_ylabel('Calibration (% within 10th-90th)')
    ax.set_title(f'iTransformer Phase 4.3 ({horizon}) \u2014 Prediction Interval Calibration')
    ax.set_ylim(0, 1.0)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, calibrations):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f'{val:.1%}', ha='center', fontsize=11)

    plt.tight_layout()
    filepath = os.path.join(output_dir, 'plot_calibration_summary.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {filepath}")


def main(horizon, num_epochs=None):
    epochs = num_epochs if num_epochs is not None else NUM_EPOCHS
    future_months = HORIZON_MAP[horizon]
    output_dir = get_output_dir(horizon, epochs)

    pred_path = os.path.join(output_dir, 'test_predictions.npy')
    tgt_path = os.path.join(output_dir, 'test_targets.npy')
    mask_path = os.path.join(output_dir, 'test_masks.npy')

    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"Predictions not found: {pred_path}\nRun test.py first.")

    predictions = np.load(pred_path)
    targets = np.load(tgt_path)
    masks = np.load(mask_path)

    print(f"[INFO] Loaded predictions: {predictions.shape}")
    print(f"[INFO] Loaded targets: {targets.shape}")
    print(f"[INFO] Horizon: {horizon} ({future_months} steps = {future_months / STEPS_PER_HOUR:.1f} hours)")
    print(f"[INFO] Vital signs: {SIGNAL_NAMES}")

    plot_forecast_samples(predictions, targets, masks, output_dir, future_months)
    plot_scatter_per_vital(predictions, targets, masks, output_dir, horizon)
    plot_error_by_step(predictions, targets, masks, output_dir, future_months, horizon)
    plot_calibration_summary(predictions, targets, masks, output_dir, horizon)

    print(f"\n[INFO] All plots saved to: {output_dir}/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Plot iTransformer results (Phase 4.3)')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS)
    parser.add_argument('--horizon', type=str, choices=['1h', '3h', '6h'], default='6h')
    args = parser.parse_args()
    main(horizon=args.horizon, num_epochs=args.epochs)
