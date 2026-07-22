#!/usr/bin/env python3
"""
Phase 5 TFT Testing: Evaluate trained model on test set.

Loads best checkpoint, runs inference, denormalizes predictions,
computes metrics, and saves results.
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
from scipy.stats import pearsonr
from omegaconf import OmegaConf

from model import TemporalFusionTransformer
from preprocess import create_dataloaders


VITAL_NAMES = ['ABPMean', 'PULSE', 'SpO2', 'RESP']
VITAL_INDICES = [7, 8, 9, 10]  # Indices in the 11-dim feature vector
QUANTILES = [0.1, 0.5, 0.9]


def denormalize_vitals(data, norm_params):
    """
    Denormalize vital sign predictions/targets.

    Args:
        data: (..., 4) array with normalized vitals
        norm_params: dict with 'means' and 'stds' lists (length 11)

    Returns:
        Denormalized array of same shape
    """
    means = np.array(norm_params['means'])[VITAL_INDICES]
    stds = np.array(norm_params['stds'])[VITAL_INDICES]
    return data * stds + means


def compute_metrics(predictions, targets):
    """
    Compute comprehensive metrics.

    Args:
        predictions: (N, 24, 4, 3) — [samples, time, vitals, quantiles]
        targets: (N, 24, 4) — [samples, time, vitals]

    Returns:
        dict with all metrics
    """
    metrics = {}
    N, T, V, Q = predictions.shape

    # Per-vital, per-quantile metrics
    for v_idx, v_name in enumerate(VITAL_NAMES):
        metrics[v_name] = {}

        for q_idx, q_val in enumerate(QUANTILES):
            pred_q = predictions[:, :, v_idx, q_idx]  # (N, 24)
            tgt = targets[:, :, v_idx]                 # (N, 24)

            # MAE
            mae = np.mean(np.abs(pred_q - tgt))

            # MAPE (avoid division by zero)
            mask = np.abs(tgt) > 1e-6
            if mask.sum() > 0:
                mape = np.mean(np.abs((pred_q[mask] - tgt[mask]) / tgt[mask])) * 100
            else:
                mape = float('nan')

            metrics[v_name][f'MAE_q{q_val}'] = float(mae)
            metrics[v_name][f'MAPE_q{q_val}'] = float(mape)

        # Median prediction metrics (q=0.5)
        pred_median = predictions[:, :, v_idx, 1]  # quantile index 1 = 0.5
        tgt = targets[:, :, v_idx]

        # Pearson correlation (flatten across time and samples)
        pred_flat = pred_median.flatten()
        tgt_flat = tgt.flatten()
        valid = ~(np.isnan(pred_flat) | np.isnan(tgt_flat))
        if valid.sum() > 2:
            corr, pval = pearsonr(pred_flat[valid], tgt_flat[valid])
        else:
            corr, pval = float('nan'), float('nan')
        metrics[v_name]['pearson_r'] = float(corr)
        metrics[v_name]['pearson_pval'] = float(pval)

        # Calibration: % of targets within [q0.1, q0.9] prediction interval
        lower = predictions[:, :, v_idx, 0]  # q0.1
        upper = predictions[:, :, v_idx, 2]  # q0.9
        in_interval = (tgt >= lower) & (tgt <= upper)
        calibration = float(np.mean(in_interval) * 100)
        metrics[v_name]['calibration_pct'] = calibration  # Target: 80%

    # Overall metrics (median quantile)
    pred_all_median = predictions[:, :, :, 1]  # (N, 24, 4)
    overall_mae = float(np.mean(np.abs(pred_all_median - targets)))
    overall_rmse = float(np.sqrt(np.mean((pred_all_median - targets) ** 2)))

    metrics['overall'] = {
        'MAE': overall_mae,
        'RMSE': overall_rmse,
    }

    # Per-vital MAE/RMSE (median)
    for v_idx, v_name in enumerate(VITAL_NAMES):
        pred_v = predictions[:, :, v_idx, 1]
        tgt_v = targets[:, :, v_idx]
        metrics[v_name]['MAE'] = float(np.mean(np.abs(pred_v - tgt_v)))
        metrics[v_name]['RMSE'] = float(np.sqrt(np.mean((pred_v - tgt_v) ** 2)))

    return metrics


def main():
    parser = argparse.ArgumentParser(description='Phase 5 TFT Testing')
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

    with torch.no_grad():
        for batch in test_loader:
            batch_device = {k: v.to(device) for k, v in batch.items()}

            output = model(batch_device)
            pred = output['predicted_quantiles']  # (batch, 24, 4*3)
            tgt = batch_device['target']          # (batch, 24, 4)

            # Reshape predictions: (batch, 24, 4, 3)
            pred = pred.view(pred.shape[0], pred.shape[1], 4, 3)

            all_predictions.append(pred.cpu().numpy())
            all_targets.append(tgt.cpu().numpy())

    predictions = np.concatenate(all_predictions, axis=0)  # (N, 24, 4, 3)
    targets = np.concatenate(all_targets, axis=0)          # (N, 24, 4)
    print(f"  Predictions shape: {predictions.shape}")
    print(f"  Targets shape: {targets.shape}")

    # ─── Denormalize ──────────────────────────────────────────────────────────
    print("Denormalizing...")
    means = np.array(norm_params['means'])[VITAL_INDICES]
    stds = np.array(norm_params['stds'])[VITAL_INDICES]

    # Denormalize predictions: (N, 24, 4, 3)
    predictions_denorm = predictions * stds[np.newaxis, np.newaxis, :, np.newaxis] + \
                         means[np.newaxis, np.newaxis, :, np.newaxis]

    # Denormalize targets: (N, 24, 4)
    targets_denorm = targets * stds[np.newaxis, np.newaxis, :] + means[np.newaxis, np.newaxis, :]

    # ─── Compute Metrics ──────────────────────────────────────────────────────
    print("Computing metrics...")
    metrics = compute_metrics(predictions_denorm, targets_denorm)

    # Print summary
    print("\n" + "=" * 60)
    print("TEST RESULTS (denormalized)")
    print("=" * 60)
    print(f"\n{'Vital':<10} {'MAE':<8} {'RMSE':<8} {'Corr':<8} {'Calib%':<8}")
    print("-" * 44)
    for v_name in VITAL_NAMES:
        m = metrics[v_name]
        print(f"{v_name:<10} {m['MAE']:<8.3f} {m['RMSE']:<8.3f} "
              f"{m['pearson_r']:<8.4f} {m['calibration_pct']:<8.1f}")
    print("-" * 44)
    print(f"{'Overall':<10} {metrics['overall']['MAE']:<8.3f} {metrics['overall']['RMSE']:<8.3f}")

    print(f"\nPer-quantile MAE:")
    print(f"{'Vital':<10} {'q0.1':<10} {'q0.5':<10} {'q0.9':<10}")
    print("-" * 40)
    for v_name in VITAL_NAMES:
        m = metrics[v_name]
        print(f"{v_name:<10} {m['MAE_q0.1']:<10.3f} {m['MAE_q0.5']:<10.3f} {m['MAE_q0.9']:<10.3f}")

    # ─── Save Results ─────────────────────────────────────────────────────────
    print(f"\nSaving results to {output_dir}/...")

    np.save(os.path.join(output_dir, "test_predictions.npy"), predictions_denorm)
    np.save(os.path.join(output_dir, "test_targets.npy"), targets_denorm)

    with open(os.path.join(output_dir, "test_metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=2)

    print("Done!")


if __name__ == "__main__":
    main()
