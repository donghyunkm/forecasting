#!/usr/bin/env python3
"""
Phase 5 TFT Plotting: Generate forecast visualization plots.

Produces:
1. 5 sample forecast plots (history + prediction with uncertainty)
2. Scatter plots per vital (predicted vs actual)
3. MAE by forecast step (x-axis in hours at 5-min resolution)
4. Calibration bar chart
"""

import os
import sys
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

VITAL_NAMES = ['ABPMean', 'PULSE', 'SpO2', 'RESP']
VITAL_UNITS = ['mmHg', 'bpm', '%', 'br/min']
VITAL_INDICES = [7, 8, 9, 10]
QUANTILES = [0.1, 0.5, 0.9]

# Time resolution: 5 minutes per step
STEP_MINUTES = 5
HISTORY_STEPS = 72
FORECAST_STEPS = 24


def load_results(output_dir, norm_params_path):
    """Load test predictions, targets, and metrics."""
    predictions = np.load(os.path.join(output_dir, "test_predictions.npy"))  # (N, 24, 4, 3)
    targets = np.load(os.path.join(output_dir, "test_targets.npy"))          # (N, 24, 4)

    with open(os.path.join(output_dir, "test_metrics.json"), 'r') as f:
        metrics = json.load(f)

    with open(norm_params_path, 'r') as f:
        norm_params = json.load(f)

    return predictions, targets, metrics, norm_params


def load_historical_for_plots(data_dir):
    """Load test data to get historical values for sample plots."""
    import torch
    test_data = torch.load(os.path.join(data_dir, "test_data.pt"), map_location='cpu')
    # historical_ts_numeric: (N, 72, 12) — first 11 channels are features
    return test_data['historical_ts_numeric'].numpy()


def plot_sample_forecasts(predictions, targets, historical, norm_params, output_dir, n_samples=5):
    """Plot sample forecasts with history, prediction, and uncertainty bands."""
    means = np.array(norm_params['means'])[VITAL_INDICES]
    stds = np.array(norm_params['stds'])[VITAL_INDICES]

    # Denormalize historical vitals (indices 7-10 in the 12-channel historical)
    hist_vitals = historical[:, :, 7:11]  # (N, 72, 4) — normalized
    hist_vitals_denorm = hist_vitals * stds[np.newaxis, np.newaxis, :] + means[np.newaxis, np.newaxis, :]

    # Select samples (spread across dataset)
    N = predictions.shape[0]
    sample_indices = np.linspace(0, N - 1, n_samples, dtype=int)

    # Time axes (in hours)
    hist_hours = np.arange(HISTORY_STEPS) * STEP_MINUTES / 60.0
    forecast_hours = (HISTORY_STEPS + np.arange(FORECAST_STEPS)) * STEP_MINUTES / 60.0

    fig, axes = plt.subplots(n_samples, 4, figsize=(20, 4 * n_samples))
    if n_samples == 1:
        axes = axes[np.newaxis, :]

    for row, idx in enumerate(sample_indices):
        for col, (v_name, v_unit) in enumerate(zip(VITAL_NAMES, VITAL_UNITS)):
            ax = axes[row, col]

            # History
            ax.plot(hist_hours, hist_vitals_denorm[idx, :, col],
                    'b-', linewidth=1, alpha=0.7, label='History')

            # Target
            ax.plot(forecast_hours, targets[idx, :, col],
                    'k-', linewidth=2, label='Actual')

            # Median prediction
            ax.plot(forecast_hours, predictions[idx, :, col, 1],
                    'r-', linewidth=2, label='Pred (q0.5)')

            # Uncertainty band (q0.1 to q0.9)
            ax.fill_between(forecast_hours,
                            predictions[idx, :, col, 0],
                            predictions[idx, :, col, 2],
                            color='red', alpha=0.2, label='80% PI')

            # Vertical line at forecast start
            ax.axvline(x=HISTORY_STEPS * STEP_MINUTES / 60.0,
                       color='gray', linestyle='--', alpha=0.5)

            ax.set_xlabel('Hours')
            ax.set_ylabel(f'{v_name} ({v_unit})')
            if row == 0:
                ax.set_title(f'{v_name}')
            if row == 0 and col == 0:
                ax.legend(fontsize=8, loc='upper left')

    plt.suptitle('Phase 5 TFT: Sample Forecasts (5-min resolution)', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "sample_forecasts.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved sample_forecasts.png")


def plot_scatter(predictions, targets, output_dir):
    """Scatter plots: predicted vs actual for each vital (median quantile)."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    for col, (v_name, v_unit) in enumerate(zip(VITAL_NAMES, VITAL_UNITS)):
        ax = axes[col]

        pred = predictions[:, :, col, 1].flatten()  # Median
        tgt = targets[:, :, col].flatten()

        # Subsample for plotting if too many points
        n_pts = len(pred)
        if n_pts > 10000:
            idx = np.random.default_rng(42).choice(n_pts, 10000, replace=False)
            pred_plot = pred[idx]
            tgt_plot = tgt[idx]
        else:
            pred_plot = pred
            tgt_plot = tgt

        ax.scatter(tgt_plot, pred_plot, alpha=0.1, s=5, color='steelblue')

        # Identity line
        vmin = min(tgt_plot.min(), pred_plot.min())
        vmax = max(tgt_plot.max(), pred_plot.max())
        ax.plot([vmin, vmax], [vmin, vmax], 'r--', linewidth=1.5, label='y=x')

        # Correlation
        valid = ~(np.isnan(pred) | np.isnan(tgt))
        if valid.sum() > 2:
            from scipy.stats import pearsonr
            r, _ = pearsonr(pred[valid], tgt[valid])
            ax.set_title(f'{v_name} (r={r:.3f})')
        else:
            ax.set_title(v_name)

        ax.set_xlabel(f'Actual ({v_unit})')
        ax.set_ylabel(f'Predicted ({v_unit})')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Phase 5 TFT: Predicted vs Actual (Median Quantile)', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "scatter_per_vital.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved scatter_per_vital.png")


def plot_mae_by_step(predictions, targets, output_dir):
    """Plot MAE as a function of forecast step (x-axis in hours)."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    forecast_hours = (np.arange(FORECAST_STEPS) + 1) * STEP_MINUTES / 60.0

    for col, (v_name, v_unit) in enumerate(zip(VITAL_NAMES, VITAL_UNITS)):
        ax = axes[col]

        # MAE per step for each quantile
        for q_idx, q_val in enumerate(QUANTILES):
            mae_per_step = np.mean(
                np.abs(predictions[:, :, col, q_idx] - targets[:, :, col]), axis=0)
            ax.plot(forecast_hours, mae_per_step, '-o', markersize=3,
                    label=f'q{q_val}', linewidth=1.5)

        ax.set_xlabel('Forecast Horizon (hours)')
        ax.set_ylabel(f'MAE ({v_unit})')
        ax.set_title(v_name)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, forecast_hours[-1] + 0.05])

    plt.suptitle('Phase 5 TFT: MAE by Forecast Step (5-min resolution)', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mae_by_step.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved mae_by_step.png")


def plot_calibration(metrics, output_dir):
    """Bar chart of calibration (% within 10th-90th PI) per vital."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    calibrations = [metrics[v_name]['calibration_pct'] for v_name in VITAL_NAMES]

    bars = ax.bar(VITAL_NAMES, calibrations, color='steelblue', alpha=0.8, edgecolor='black')

    # Target line at 80%
    ax.axhline(y=80, color='red', linestyle='--', linewidth=2, label='Target (80%)')

    ax.set_ylabel('Calibration (%)')
    ax.set_title('Phase 5 TFT: Prediction Interval Calibration\n(% of targets within 10th-90th quantile)')
    ax.set_ylim([0, 105])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, val in zip(bars, calibrations):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f'{val:.1f}%', ha='center', fontsize=11, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "calibration.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved calibration.png")


def main():
    parser = argparse.ArgumentParser(description='Phase 5 TFT Plot Predictions')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Epochs config (to find output directory)')
    args = parser.parse_args()

    # ─── Paths ────────────────────────────────────────────────────────────────
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, f"outputs/tft_epochs_{args.epochs}")
    data_dir = "/gpfs/home/dk5565/forecasting/phase5/phase5_data/processed/"
    norm_params_path = os.path.join(data_dir, "norm_params.json")

    if not os.path.exists(os.path.join(output_dir, "test_predictions.npy")):
        print(f"ERROR: test_predictions.npy not found in {output_dir}")
        print("Run test.py first.")
        sys.exit(1)

    # ─── Load Results ─────────────────────────────────────────────────────────
    print("Loading results...")
    predictions, targets, metrics, norm_params = load_results(output_dir, norm_params_path)
    print(f"  Predictions: {predictions.shape}")
    print(f"  Targets: {targets.shape}")

    # Load historical data for sample plots
    print("Loading historical data for sample plots...")
    historical = load_historical_for_plots(data_dir)
    print(f"  Historical: {historical.shape}")

    # ─── Generate Plots ───────────────────────────────────────────────────────
    print("\nGenerating plots...")

    plot_sample_forecasts(predictions, targets, historical, norm_params, output_dir)
    plot_scatter(predictions, targets, output_dir)
    plot_mae_by_step(predictions, targets, output_dir)
    plot_calibration(metrics, output_dir)

    print(f"\nAll plots saved to {output_dir}/")
    print("Done!")


if __name__ == "__main__":
    main()
