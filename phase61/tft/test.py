#!/usr/bin/env python3
"""
Phase 6.1 TFT Testing: Evaluate trained model on test set.

Loads best checkpoint, runs inference, denormalizes predictions,
computes metrics, saves results, and generates plots.
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
from scipy.stats import pearsonr
from omegaconf import OmegaConf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model import TemporalFusionTransformer
from preprocess import create_dataloaders


CORRELATION_NAMES = [
    'PLETH_ACDC×PLETH_amp', 'ABP_area×ABP_tau', 'ABP_area×ShockIdx',
    'PLETH_amp×ShockIdx', 'PLETH_ACDC×ShockIdx', 'ShockIdx×ABP_tau',
    'PLETH_ACDC×ABP_tau'
]
NUM_CORR = 7


def denormalize(data, norm_params):
    """
    Invert normalization: z-score denorm → inverse Fisher z (tanh) → correlation space.

    Args:
        data: (..., 7) array with normalized values (in Fisher z-score space)
        norm_params: dict with 'corr_means' and 'corr_stds' lists (length 7)

    Returns:
        Denormalized array in original correlation space [-1, 1]
    """
    means = np.array(norm_params['corr_means'])
    stds = np.array(norm_params['corr_stds'])
    # Step 1: undo z-score → back to Fisher z-space
    z = data * stds + means
    # Step 2: undo Fisher z-transform → back to correlation space
    return np.tanh(z)


def compute_metrics(predictions, targets):
    """
    Compute comprehensive metrics.

    Args:
        predictions: (N, 12, 7) — point predictions
        targets: (N, 12, 7) — ground truth

    Returns:
        dict with all metrics
    """
    metrics = {}

    all_errors = []
    all_sq_errors = []

    for c_idx, c_name in enumerate(CORRELATION_NAMES):
        pred_c = predictions[:, :, c_idx].flatten()
        tgt_c = targets[:, :, c_idx].flatten()

        # MAE
        errors = np.abs(pred_c - tgt_c)
        mae = errors.mean()

        # RMSE
        rmse = np.sqrt(((pred_c - tgt_c) ** 2).mean())

        # Pearson correlation
        if len(pred_c) > 1 and pred_c.std() > 1e-8 and tgt_c.std() > 1e-8:
            corr, pval = pearsonr(pred_c, tgt_c)
        else:
            corr, pval = float('nan'), float('nan')

        # MAPE (correlations can be near zero, use threshold)
        nonzero = np.abs(tgt_c) > 0.05
        if nonzero.sum() > 0:
            mape = (errors[nonzero] / np.abs(tgt_c[nonzero])).mean() * 100
        else:
            mape = float('nan')

        metrics[c_name] = {
            'MAE': float(mae),
            'RMSE': float(rmse),
            'MAPE': float(mape),
            'pearson_r': float(corr),
            'pearson_pval': float(pval),
        }

        all_errors.append(errors)
        all_sq_errors.append((pred_c - tgt_c) ** 2)

    # Overall metrics
    all_errors_cat = np.concatenate(all_errors)
    all_sq_errors_cat = np.concatenate(all_sq_errors)
    metrics['overall'] = {
        'MAE': float(all_errors_cat.mean()),
        'RMSE': float(np.sqrt(all_sq_errors_cat.mean())),
    }

    return metrics


def plot_sample_forecasts(predictions, targets, historical_corr, output_dir, n_samples=4):
    """Plot sample forecast trajectories — one subplot per correlation feature."""
    n_hist = historical_corr.shape[1]  # 48
    n_pred = predictions.shape[1]      # 12

    rng = np.random.default_rng(42)
    sample_indices = rng.choice(len(predictions), size=n_samples, replace=False)

    t_hist = np.arange(-n_hist, 0) * 2.5
    t_pred = np.arange(0, n_pred) * 2.5

    for c_idx, c_name in enumerate(CORRELATION_NAMES):
        fig, axes = plt.subplots(n_samples, 1, figsize=(12, 3 * n_samples))
        if n_samples == 1:
            axes = [axes]

        for ax_idx, sample_idx in enumerate(sample_indices):
            ax = axes[ax_idx]

            ax.plot(t_hist, historical_corr[sample_idx, :, c_idx],
                    '-', color='steelblue', linewidth=1.5, label='Input history')
            ax.plot(t_pred, targets[sample_idx, :, c_idx],
                    'o-', color='steelblue', markersize=4, linewidth=2, label='True future')
            ax.plot(t_pred, predictions[sample_idx, :, c_idx],
                    's--', color='darkorange', markersize=4, linewidth=2, label='Predicted')

            ax.axvline(0, color='black', linestyle=':', linewidth=1.5, alpha=0.7)
            ax.set_ylabel('Correlation')
            ax.set_ylim(-1.1, 1.1)
            ax.axhline(0, color='gray', linestyle=':', alpha=0.3)
            ax.grid(True, alpha=0.2)
            ax.set_title(f'Sample {sample_idx}', fontsize=9)
            if ax_idx == 0:
                ax.legend(loc='upper right', fontsize=8)

        axes[-1].set_xlabel('Time relative to forecast start (minutes)')
        fig.suptitle(f'TFT (Phase 6.1) — {c_name}', fontsize=13, y=1.01)
        plt.tight_layout()

        fname = f"sample_forecasts_{c_idx:02d}_{c_name.replace('×', '_')}.png"
        plt.savefig(os.path.join(output_dir, fname), dpi=150, bbox_inches='tight')
        plt.close()

    print(f"  Saved: sample_forecasts_*.png (7 files, one per correlation)")


def plot_error_by_horizon(predictions, targets, output_dir):
    """Plot MAE as a function of forecast horizon for each correlation."""
    n_steps = predictions.shape[1]
    time_steps = np.arange(n_steps) * 2.5

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    for c_idx, c_name in enumerate(CORRELATION_NAMES):
        mae_per_step = np.abs(predictions[:, :, c_idx] - targets[:, :, c_idx]).mean(axis=0)
        ax.plot(time_steps, mae_per_step, 'o-', markersize=5, label=c_name)

    ax.set_xlabel('Forecast horizon (minutes)')
    ax.set_ylabel('MAE (correlation units)')
    ax.set_title('TFT (Phase 6.1) — MAE by Forecast Horizon')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "error_by_horizon.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: error_by_horizon.png")


def plot_scatter(predictions, targets, output_dir):
    """Scatter plot of predicted vs actual for each correlation."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for c_idx, c_name in enumerate(CORRELATION_NAMES):
        ax = axes[c_idx]
        pred_flat = predictions[:, :, c_idx].flatten()
        tgt_flat = targets[:, :, c_idx].flatten()

        if len(pred_flat) > 5000:
            idx = np.random.default_rng(42).choice(len(pred_flat), 5000, replace=False)
            pred_flat = pred_flat[idx]
            tgt_flat = tgt_flat[idx]

        ax.scatter(tgt_flat, pred_flat, alpha=0.15, s=5, c='steelblue')
        ax.plot([-1, 1], [-1, 1], 'r--', linewidth=1.5, label='Perfect')
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        ax.set_title(c_name, fontsize=9)
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2)

    axes[7].axis('off')

    plt.suptitle('TFT (Phase 6.1) — Predicted vs Actual', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "scatter_pred_vs_actual.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: scatter_pred_vs_actual.png")


def plot_metrics_bar(metrics, output_dir):
    """Bar chart of per-correlation MAE and Pearson r."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    names_short = [n.replace('×', '\n×\n') for n in CORRELATION_NAMES]
    maes = [metrics[c]['MAE'] for c in CORRELATION_NAMES]
    pearson_rs = [metrics[c]['pearson_r'] for c in CORRELATION_NAMES]

    x = np.arange(len(CORRELATION_NAMES))

    ax1.bar(x, maes, color='steelblue', alpha=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names_short, fontsize=7)
    ax1.set_ylabel('MAE')
    ax1.set_title('Per-Correlation MAE')
    ax1.grid(True, alpha=0.2, axis='y')

    ax2.bar(x, pearson_rs, color='darkorange', alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names_short, fontsize=7)
    ax2.set_ylabel('Pearson r')
    ax2.set_title('Per-Correlation Pearson r')
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.2, axis='y')

    plt.suptitle('TFT (Phase 6.1) — Test Metrics', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "metrics_bar.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: metrics_bar.png")


def plot_bland_altman(predictions, targets, output_dir):
    """Bland-Altman plots for each correlation."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for c_idx, c_name in enumerate(CORRELATION_NAMES):
        ax = axes[c_idx]
        pred_flat = predictions[:, :, c_idx].flatten()
        tgt_flat = targets[:, :, c_idx].flatten()

        mean_vals = (pred_flat + tgt_flat) / 2
        diff_vals = pred_flat - tgt_flat

        if len(mean_vals) > 5000:
            idx = np.random.default_rng(42).choice(len(mean_vals), 5000, replace=False)
            mean_plot = mean_vals[idx]
            diff_plot = diff_vals[idx]
        else:
            mean_plot = mean_vals
            diff_plot = diff_vals

        bias = diff_vals.mean()
        sd = diff_vals.std()
        upper_loa = bias + 1.96 * sd
        lower_loa = bias - 1.96 * sd

        ax.scatter(mean_plot, diff_plot, alpha=0.1, s=5, c='steelblue')
        ax.axhline(bias, color='red', linestyle='-', linewidth=1.5, label=f'Bias: {bias:.3f}')
        ax.axhline(upper_loa, color='orange', linestyle='--', linewidth=1,
                   label=f'+1.96SD: {upper_loa:.3f}')
        ax.axhline(lower_loa, color='orange', linestyle='--', linewidth=1,
                   label=f'−1.96SD: {lower_loa:.3f}')
        ax.axhline(0, color='gray', linestyle=':', alpha=0.5)

        ax.set_xlabel('Mean (pred, actual)')
        ax.set_ylabel('Difference (pred − actual)')
        ax.set_title(c_name, fontsize=9)
        ax.legend(fontsize=6, loc='upper right')
        ax.grid(True, alpha=0.2)

    axes[7].axis('off')

    plt.suptitle('TFT (Phase 6.1) — Bland-Altman Plots', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "bland_altman.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: bland_altman.png")


def main():
    parser = argparse.ArgumentParser(description='Phase 6.1 TFT Testing')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Epochs config (to find checkpoint directory)')
    args = parser.parse_args()

    # ─── Paths ────────────────────────────────────────────────────────────────
    base_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_dir = os.path.join(base_dir, f"checkpoints/tft_epochs_{args.epochs}")
    output_dir = os.path.join(base_dir, f"outputs/tft_epochs_{args.epochs}")
    os.makedirs(output_dir, exist_ok=True)

    checkpoint_path = os.path.join(checkpoint_dir, "best_model.pt")
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    # ─── Device ───────────────────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ─── Load Data ────────────────────────────────────────────────────────────
    print("Loading data...")
    _, _, test_loader, norm_params = create_dataloaders(batch_size=64)
    print(f"  Test batches: {len(test_loader)}")

    # ─── Load Model ───────────────────────────────────────────────────────────
    print("Loading model...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = OmegaConf.create(checkpoint['config'])
    model = TemporalFusionTransformer(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"  Loaded from epoch {checkpoint['epoch']} (val_loss={checkpoint['val_loss']:.6f})")

    # ─── Inference ────────────────────────────────────────────────────────────
    print("Running inference on test set...")
    all_predictions = []
    all_targets = []
    all_historical = []

    with torch.no_grad():
        for batch in test_loader:
            batch_device = {k: v.to(device) for k, v in batch.items()}

            output = model(batch_device)
            pred = output['predicted_quantiles']  # (batch, 12, 7)
            tgt = batch_device['target']          # (batch, 12, 7)

            all_predictions.append(pred.cpu().numpy())
            all_targets.append(tgt.cpu().numpy())
            all_historical.append(batch['historical_ts_numeric'].numpy())  # (batch, 48, 46)

    predictions = np.concatenate(all_predictions, axis=0)  # (N, 12, 7)
    targets = np.concatenate(all_targets, axis=0)          # (N, 12, 7)
    historical = np.concatenate(all_historical, axis=0)    # (N, 48, 46)
    print(f"  Predictions shape: {predictions.shape}")
    print(f"  Targets shape: {targets.shape}")
    print(f"  Historical shape: {historical.shape}")

    # ─── Denormalize ──────────────────────────────────────────────────────────
    print("Denormalizing...")
    predictions_denorm = denormalize(predictions, norm_params)
    targets_denorm = denormalize(targets, norm_params)
    # Denormalize historical (first 7 channels are correlations)
    historical_corr = denormalize(historical[:, :, :7], norm_params)

    # ─── Compute Metrics ──────────────────────────────────────────────────────
    print("Computing metrics...")
    metrics = compute_metrics(predictions_denorm, targets_denorm)

    # Print summary
    print("\n" + "=" * 70)
    print("TEST RESULTS — Phase 6.1 TFT (Correlation + Physio Forecasting)")
    print("=" * 70)
    print(f"\n{'Correlation':<25} {'MAE':<8} {'RMSE':<8} {'Pearson r':<10}")
    print("-" * 53)
    for c_name in CORRELATION_NAMES:
        m = metrics[c_name]
        print(f"{c_name:<25} {m['MAE']:<8.4f} {m['RMSE']:<8.4f} {m['pearson_r']:<10.4f}")
    print("-" * 53)
    print(f"{'Overall':<25} {metrics['overall']['MAE']:<8.4f} {metrics['overall']['RMSE']:<8.4f}")

    # ─── Save Results ─────────────────────────────────────────────────────────
    print(f"\nSaving results to {output_dir}/...")

    np.save(os.path.join(output_dir, "test_predictions.npy"), predictions_denorm)
    np.save(os.path.join(output_dir, "test_targets.npy"), targets_denorm)

    with open(os.path.join(output_dir, "test_metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=2)

    # ─── Generate Plots ───────────────────────────────────────────────────────
    print("\nGenerating plots...")
    plot_sample_forecasts(predictions_denorm, targets_denorm, historical_corr, output_dir)
    plot_error_by_horizon(predictions_denorm, targets_denorm, output_dir)
    plot_scatter(predictions_denorm, targets_denorm, output_dir)
    plot_bland_altman(predictions_denorm, targets_denorm, output_dir)
    plot_metrics_bar(metrics, output_dir)

    print("Done!")


if __name__ == "__main__":
    main()
