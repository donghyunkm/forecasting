"""
Plotting script for Phase 5 iTransformer vital sign forecasting.

Generates:
1. Sample forecast plots (5 patients) with history + prediction + uncertainty bands
2. Scatter plots per vital with correlation and MAE
3. MAE by forecast step (x-axis in hours)
4. Calibration bar chart per vital with 80% target line

All saved to outputs/itransformer_epochs_{N}/
"""

import os
import sys
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


VITAL_NAMES = ['ABPMean', 'PULSE', 'SpO2', 'RESP']
VITAL_UNITS = ['mmHg', 'bpm', '%', 'br/min']
QUANTILES = [0.1, 0.5, 0.9]

# Time resolution: 5 minutes per step
STEP_MINUTES = 5
HISTORY_STEPS = 72   # 6 hours
PRED_STEPS = 24      # 2 hours


def load_results(output_dir):
    """Load predictions, targets, masks, and metrics."""
    predictions = np.load(os.path.join(output_dir, "test_predictions.npy"))  # (N, 24, 4, 3)
    targets = np.load(os.path.join(output_dir, "test_targets.npy"))          # (N, 24, 4)
    masks = np.load(os.path.join(output_dir, "test_masks.npy"))              # (N, 24, 4)

    with open(os.path.join(output_dir, "test_metrics.json"), 'r') as f:
        metrics = json.load(f)

    return predictions, targets, masks, metrics


def plot_sample_forecasts(predictions, targets, masks, output_dir, n_samples=5):
    """
    Plot 5 sample forecast plots showing 6h history + 2h prediction with uncertainty bands.
    Loads historical data from the test .pt file and denormalizes everything.
    """
    import torch

    # Load test data to get historical values
    data_dir = "/gpfs/home/dk5565/forecasting/phase5/phase5_data/processed"
    test_data = torch.load(os.path.join(data_dir, "test_data.pt"), map_location='cpu',
                           weights_only=False)
    historical = test_data['historical_ts_numeric'].numpy()  # (N, 72, 12)

    # Load norm params to denormalize history
    norm_path = os.path.join(data_dir, "norm_params.json")
    with open(norm_path, 'r') as f:
        norm_data = json.load(f)
    means = np.array(norm_data['means'])  # (11,)
    stds = np.array(norm_data['stds'])    # (11,)
    vital_means = means[7:11]  # (4,)
    vital_stds = stds[7:11]    # (4,)

    # Denormalize historical vitals (channels 7-10 in the 12-channel input)
    # Channel indices in historical_ts_numeric: 0-6=corr, 7-10=vitals, 11=time
    hist_vitals_norm = historical[:, :, 7:11]  # (N, 72, 4)
    hist_vitals = hist_vitals_norm * vital_stds[None, None, :] + vital_means[None, None, :]

    fig, axes = plt.subplots(n_samples, 4, figsize=(20, 3.5 * n_samples))

    # Select samples with good mask coverage
    valid_counts = masks.sum(axis=(1, 2))  # (N,)
    good_indices = np.where(valid_counts > 80)[0]
    if len(good_indices) < n_samples:
        good_indices = np.argsort(valid_counts)[-n_samples:]

    np.random.seed(42)
    sample_indices = np.random.choice(good_indices, size=n_samples, replace=False)
    sample_indices.sort()

    # Time axes in hours
    hist_hours = np.arange(HISTORY_STEPS) * STEP_MINUTES / 60.0        # 0 to 6h
    pred_hours = (HISTORY_STEPS + np.arange(PRED_STEPS)) * STEP_MINUTES / 60.0  # 6h to 8h

    for row, s_idx in enumerate(sample_indices):
        for col, (v_name, v_unit) in enumerate(zip(VITAL_NAMES, VITAL_UNITS)):
            ax = axes[row, col] if n_samples > 1 else axes[col]

            # History (denormalized)
            ax.plot(hist_hours, hist_vitals[s_idx, :, col],
                    'b-', linewidth=1, alpha=0.7, label='History')

            # Target (already denormalized from test.py)
            target_v = targets[s_idx, :, col]
            ax.plot(pred_hours, target_v, 'k-', linewidth=2, label='Actual')

            # Predictions (already denormalized from test.py)
            q_med = predictions[s_idx, :, col, 1]
            q_low = predictions[s_idx, :, col, 0]
            q_high = predictions[s_idx, :, col, 2]

            ax.plot(pred_hours, q_med, 'r-', linewidth=2, label='Pred (median)')
            ax.fill_between(pred_hours, q_low, q_high,
                           alpha=0.2, color='red', label='80% PI')

            # Vertical line at forecast start
            ax.axvline(x=HISTORY_STEPS * STEP_MINUTES / 60.0,
                       color='gray', linestyle='--', alpha=0.5)

            ax.set_xlabel('Time (hours)')
            ax.set_ylabel(f'{v_name} ({v_unit})')
            if row == 0:
                ax.set_title(f'{v_name}')
            if row == 0 and col == 0:
                ax.legend(fontsize=8, loc='upper left')
            ax.grid(True, alpha=0.3)

    plt.suptitle('iTransformer: Sample Forecast Plots (Phase 5)', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "sample_forecasts.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: sample_forecasts.png")


def plot_scatter_per_vital(predictions, targets, masks, metrics, output_dir):
    """Scatter plot of predicted vs actual for each vital with correlation and MAE."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    median_pred = predictions[:, :, :, 1]  # (N, 24, 4) - median quantile

    for idx, (v_name, v_unit) in enumerate(zip(VITAL_NAMES, VITAL_UNITS)):
        ax = axes[idx // 2, idx % 2]

        pred_v = median_pred[:, :, idx].flatten()
        target_v = targets[:, :, idx].flatten()
        mask_v = masks[:, :, idx].flatten()

        valid = mask_v > 0.5
        pred_valid = pred_v[valid]
        target_valid = target_v[valid]

        # Subsample for plotting if too many points
        n_valid = len(pred_valid)
        if n_valid > 10000:
            subsample = np.random.choice(n_valid, 10000, replace=False)
            pred_plot = pred_valid[subsample]
            target_plot = target_valid[subsample]
        else:
            pred_plot = pred_valid
            target_plot = target_valid

        ax.scatter(target_plot, pred_plot, alpha=0.1, s=5, c='tab:blue')

        # Perfect prediction line
        vmin = min(target_plot.min(), pred_plot.min())
        vmax = max(target_plot.max(), pred_plot.max())
        ax.plot([vmin, vmax], [vmin, vmax], 'r--', linewidth=1.5, label='Perfect')

        # Metrics text
        m = metrics[v_name]
        ax.text(0.05, 0.95, f"MAE: {m['MAE']:.2f} {v_unit}\nr: {m['correlation']:.3f}\nn: {m['n_valid']:,}",
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        ax.set_xlabel(f'Actual {v_name} ({v_unit})')
        ax.set_ylabel(f'Predicted {v_name} ({v_unit})')
        ax.set_title(f'{v_name}')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='datalim')

    plt.suptitle('iTransformer: Predicted vs Actual (Phase 5)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "scatter_per_vital.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: scatter_per_vital.png")


def plot_mae_by_step(predictions, targets, masks, output_dir):
    """Plot MAE by forecast step (x-axis in hours)."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    median_pred = predictions[:, :, :, 1]  # (N, 24, 4)
    step_hours = np.arange(PRED_STEPS) * STEP_MINUTES / 60  # 0 to ~2 hours

    colors = ['tab:red', 'tab:blue', 'tab:green', 'tab:orange']

    for idx, (v_name, color) in enumerate(zip(VITAL_NAMES, colors)):
        mae_per_step = []
        for step in range(PRED_STEPS):
            pred_step = median_pred[:, step, idx]
            target_step = targets[:, step, idx]
            mask_step = masks[:, step, idx]

            valid = mask_step > 0.5
            if valid.sum() > 0:
                mae = np.abs(pred_step[valid] - target_step[valid]).mean()
            else:
                mae = float('nan')
            mae_per_step.append(mae)

        ax.plot(step_hours, mae_per_step, '-o', color=color, markersize=4,
                linewidth=1.5, label=f'{v_name}')

    ax.set_xlabel('Forecast Horizon (hours)')
    ax.set_ylabel('MAE')
    ax.set_title('iTransformer: MAE by Forecast Step (Phase 5)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, step_hours[-1])

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mae_by_step.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: mae_by_step.png")


def plot_calibration(predictions, targets, masks, output_dir):
    """Calibration bar chart per vital with 80% target line."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    calibrations = []
    for idx, v_name in enumerate(VITAL_NAMES):
        q_low = predictions[:, :, idx, 0]    # quantile 0.1
        q_high = predictions[:, :, idx, 2]   # quantile 0.9
        target_v = targets[:, :, idx]
        mask_v = masks[:, :, idx]

        valid = mask_v > 0.5
        if valid.sum() > 0:
            in_interval = (target_v[valid] >= q_low[valid]) & (target_v[valid] <= q_high[valid])
            calibration = in_interval.mean() * 100
        else:
            calibration = 0.0
        calibrations.append(calibration)

    x = np.arange(len(VITAL_NAMES))
    bars = ax.bar(x, calibrations, color=['tab:red', 'tab:blue', 'tab:green', 'tab:orange'],
                  alpha=0.8, edgecolor='black', linewidth=0.5)

    # 80% target line
    ax.axhline(y=80, color='black', linestyle='--', linewidth=2, label='Target (80%)')

    # Add value labels on bars
    for bar, cal in zip(bars, calibrations):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f'{cal:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(VITAL_NAMES)
    ax.set_ylabel('Coverage (%)')
    ax.set_title('iTransformer: 80% Prediction Interval Calibration (Phase 5)')
    ax.set_ylim(0, 105)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "calibration.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: calibration.png")


def main(args):
    """Generate all plots."""
    output_dir = f"outputs/itransformer_epochs_{args.epochs}"

    if not os.path.exists(os.path.join(output_dir, "test_predictions.npy")):
        print(f"ERROR: Test results not found in {output_dir}")
        print(f"  Run test.py --epochs {args.epochs} first.")
        sys.exit(1)

    print(f"\n--- Loading Results from {output_dir} ---")
    predictions, targets, masks, metrics = load_results(output_dir)
    print(f"  Predictions: {predictions.shape}")
    print(f"  Targets:     {targets.shape}")
    print(f"  Masks:       {masks.shape}")

    print(f"\n--- Generating Plots ---")

    # 1. Sample forecasts
    plot_sample_forecasts(predictions, targets, masks, output_dir)

    # 2. Scatter per vital
    plot_scatter_per_vital(predictions, targets, masks, metrics, output_dir)

    # 3. MAE by step
    plot_mae_by_step(predictions, targets, masks, output_dir)

    # 4. Calibration
    plot_calibration(predictions, targets, masks, output_dir)

    print(f"\n--- All plots saved to {output_dir} ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot iTransformer predictions")
    parser.add_argument('--epochs', type=int, default=100, help='Epochs (for output directory naming)')
    args = parser.parse_args()

    main(args)
