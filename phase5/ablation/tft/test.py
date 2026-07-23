#!/usr/bin/env python3
"""
Phase 5 Ablation TFT Testing: Evaluate trained model on test set.

Vitals-only ablation study. Loads best checkpoint, runs inference,
denormalizes predictions, computes metrics, and saves results.

norm_params.json has means/stds of length 4 (directly for 4 vitals).
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


QUANTILES = [0.1, 0.5, 0.9]


def compute_metrics(predictions, targets, masks, vital_names):
    """
    Compute comprehensive metrics with mask support.

    Args:
        predictions: (N, 24, 4, 3) — [samples, time, vitals, quantiles]
        targets: (N, 24, 4) — [samples, time, vitals]
        masks: (N, 24, 4) — [samples, time, vitals] valid=1
        vital_names: list of 4 vital names

    Returns:
        dict with all metrics
    """
    metrics = {}
    N, T, V, Q = predictions.shape

    for v_idx, v_name in enumerate(vital_names):
        metrics[v_name] = {}
        mask_v = masks[:, :, v_idx].astype(bool)  # (N, 24)

        for q_idx, q_val in enumerate(QUANTILES):
            pred_q = predictions[:, :, v_idx, q_idx]  # (N, 24)
            tgt = targets[:, :, v_idx]                 # (N, 24)

            # Apply mask
            pred_masked = pred_q[mask_v]
            tgt_masked = tgt[mask_v]

            if len(tgt_masked) == 0:
                metrics[v_name][f'MAE_q{q_val}'] = float('nan')
                metrics[v_name][f'MAPE_q{q_val}'] = float('nan')
                continue

            # MAE
            mae = np.mean(np.abs(pred_masked - tgt_masked))

            # MAPE (avoid division by zero)
            valid_denom = np.abs(tgt_masked) > 1e-6
            if valid_denom.sum() > 0:
                mape = np.mean(np.abs((pred_masked[valid_denom] - tgt_masked[valid_denom])
                                      / tgt_masked[valid_denom])) * 100
            else:
                mape = float('nan')

            metrics[v_name][f'MAE_q{q_val}'] = float(mae)
            metrics[v_name][f'MAPE_q{q_val}'] = float(mape)

        # Median prediction metrics (q=0.5)
        pred_median = predictions[:, :, v_idx, 1]
        tgt = targets[:, :, v_idx]
        pred_masked = pred_median[mask_v]
        tgt_masked = tgt[mask_v]

        # MAE and RMSE
        if len(tgt_masked) > 0:
            metrics[v_name]['MAE'] = float(np.mean(np.abs(pred_masked - tgt_masked)))
            metrics[v_name]['RMSE'] = float(np.sqrt(np.mean((pred_masked - tgt_masked) ** 2)))
        else:
            metrics[v_name]['MAE'] = float('nan')
            metrics[v_name]['RMSE'] = float('nan')

        # Pearson correlation
        valid = ~(np.isnan(pred_masked) | np.isnan(tgt_masked))
        if valid.sum() > 2:
            corr, pval = pearsonr(pred_masked[valid], tgt_masked[valid])
        else:
            corr, pval = float('nan'), float('nan')
        metrics[v_name]['pearson_r'] = float(corr)
        metrics[v_name]['pearson_pval'] = float(pval)

        # Calibration: % of targets within [q0.1, q0.9] prediction interval (target: 80%)
        lower = predictions[:, :, v_idx, 0]  # q0.1
        upper = predictions[:, :, v_idx, 2]  # q0.9
        in_interval = (targets[:, :, v_idx] >= lower) & (targets[:, :, v_idx] <= upper)
        # Apply mask
        in_interval_masked = in_interval[mask_v]
        if len(in_interval_masked) > 0:
            calibration = float(np.mean(in_interval_masked) * 100)
        else:
            calibration = float('nan')
        metrics[v_name]['calibration_80'] = calibration

    # Overall metrics (median quantile, masked)
    all_pred_median = predictions[:, :, :, 1]  # (N, 24, 4)
    all_mask = masks.astype(bool)
    pred_all_masked = all_pred_median[all_mask]
    tgt_all_masked = targets[all_mask]

    if len(tgt_all_masked) > 0:
        overall_mae = float(np.mean(np.abs(pred_all_masked - tgt_all_masked)))
        overall_rmse = float(np.sqrt(np.mean((pred_all_masked - tgt_all_masked) ** 2)))
    else:
        overall_mae = float('nan')
        overall_rmse = float('nan')

    metrics['overall'] = {
        'MAE': overall_mae,
        'RMSE': overall_rmse,
    }

    return metrics


def main():
    parser = argparse.ArgumentParser(description='Phase 5 TFT Ablation Testing (Vitals-Only)')
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

    # Ablation norm_params: means/stds are length 4 (directly for vitals)
    vital_names = norm_params['vital_names']
    means = np.array(norm_params['means'])  # (4,)
    stds = np.array(norm_params['stds'])    # (4,)
    print(f"  Vital names: {vital_names}")
    print(f"  Means: {means}")
    print(f"  Stds: {stds}")

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
    all_masks = []

    with torch.no_grad():
        for batch in test_loader:
            batch_device = {k: v.to(device) for k, v in batch.items()}

            output = model(batch_device)
            pred = output['predicted_quantiles']  # (batch, 24, 4*3)
            tgt = batch_device['target']          # (batch, 24, 4)
            mask = batch_device['target_mask']    # (batch, 24, 4)

            # Reshape predictions: (batch, 24, 4, 3)
            pred = pred.view(pred.shape[0], pred.shape[1], 4, 3)

            all_predictions.append(pred.cpu().numpy())
            all_targets.append(tgt.cpu().numpy())
            all_masks.append(mask.cpu().numpy())

    predictions = np.concatenate(all_predictions, axis=0)  # (N, 24, 4, 3)
    targets = np.concatenate(all_targets, axis=0)          # (N, 24, 4)
    masks = np.concatenate(all_masks, axis=0)              # (N, 24, 4)
    print(f"  Predictions shape: {predictions.shape}")
    print(f"  Targets shape: {targets.shape}")
    print(f"  Masks shape: {masks.shape}")

    # ─── Denormalize ──────────────────────────────────────────────────────────
    print("Denormalizing...")
    # means/stds are directly length 4 for the 4 vitals
    # Denormalize predictions: (N, 24, 4, 3)
    predictions_denorm = predictions * stds[np.newaxis, np.newaxis, :, np.newaxis] + \
                         means[np.newaxis, np.newaxis, :, np.newaxis]

    # Denormalize targets: (N, 24, 4)
    targets_denorm = targets * stds[np.newaxis, np.newaxis, :] + means[np.newaxis, np.newaxis, :]

    # ─── Compute Metrics ──────────────────────────────────────────────────────
    print("Computing metrics...")
    metrics = compute_metrics(predictions_denorm, targets_denorm, masks, vital_names)

    # Print summary
    print("\n" + "=" * 60)
    print("TEST RESULTS - TFT ABLATION (Vitals-Only, denormalized)")
    print("=" * 60)
    print(f"\n{'Vital':<10} {'MAE':<8} {'RMSE':<8} {'Corr':<8} {'Calib%':<8}")
    print("-" * 44)
    for v_name in vital_names:
        m = metrics[v_name]
        print(f"{v_name:<10} {m['MAE']:<8.3f} {m['RMSE']:<8.3f} "
              f"{m['pearson_r']:<8.4f} {m['calibration_80']:<8.1f}")
    print("-" * 44)
    print(f"{'Overall':<10} {metrics['overall']['MAE']:<8.3f} {metrics['overall']['RMSE']:<8.3f}")

    print(f"\nPer-quantile MAE:")
    print(f"{'Vital':<10} {'q0.1':<10} {'q0.5':<10} {'q0.9':<10}")
    print("-" * 40)
    for v_name in vital_names:
        m = metrics[v_name]
        print(f"{v_name:<10} {m['MAE_q0.1']:<10.3f} {m['MAE_q0.5']:<10.3f} {m['MAE_q0.9']:<10.3f}")

    # ─── Save Results ─────────────────────────────────────────────────────────
    print(f"\nSaving results to {output_dir}/...")

    np.save(os.path.join(output_dir, "test_predictions.npy"), predictions_denorm)
    np.save(os.path.join(output_dir, "test_targets.npy"), targets_denorm)
    np.save(os.path.join(output_dir, "test_masks.npy"), masks)

    with open(os.path.join(output_dir, "test_metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=2)

    print("Done!")


if __name__ == "__main__":
    main()
