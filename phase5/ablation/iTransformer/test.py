"""
Test script for Phase 5 iTransformer ablation (vitals-only).

Loads best model, runs inference on test set, denormalizes predictions,
and computes per-vital MAE/RMSE/MAPE/calibration/correlation metrics.
"""

import os
import sys
import argparse
import json
import numpy as np
import torch

from model import iTransformer, build_model
from preprocess import create_dataloaders, VITAL_NAMES, VITAL_INDICES


QUANTILES = [0.1, 0.5, 0.9]


def denormalize_vitals(predictions_norm, targets_norm, norm_params):
    """
    Denormalize vital sign predictions and targets.

    Both predictions and targets are in normalized space (z-scored).
    In vitals-only ablation, vital_mean and vital_std are directly (4,).

    Args:
        predictions_norm: (N, 24, 4, 3) - normalized quantile predictions
        targets_norm: (N, 24, 4) - normalized targets
        norm_params: dict with 'vital_mean' (4,) and 'vital_std' (4,)

    Returns:
        predictions_denorm: (N, 24, 4, 3) in original units
        targets_denorm: (N, 24, 4) in original units
    """
    vital_mean = norm_params['vital_mean'].numpy()  # (4,)
    vital_std = norm_params['vital_std'].numpy()    # (4,)

    # Denormalize predictions: pred_orig = pred_norm * std + mean
    predictions_denorm = predictions_norm * vital_std[None, None, :, None] + vital_mean[None, None, :, None]

    # Denormalize targets: target_orig = target_norm * std + mean
    targets_denorm = targets_norm * vital_std[None, None, :] + vital_mean[None, None, :]

    return predictions_denorm, targets_denorm


def compute_metrics(predictions, targets, masks):
    """
    Compute per-vital and overall metrics.

    Args:
        predictions: (N, 24, 4, 3) denormalized quantile predictions
        targets: (N, 24, 4) denormalized targets
        masks: (N, 24, 4) validity masks

    Returns:
        dict with per-vital and overall metrics
    """
    metrics = {}

    # Median prediction (quantile index 1 = 0.5)
    median_pred = predictions[:, :, :, 1]  # (N, 24, 4)

    # Overall metrics (across all vitals)
    all_errors = []
    all_sq_errors = []

    for v_idx, v_name in enumerate(VITAL_NAMES):
        pred_v = median_pred[:, :, v_idx]   # (N, 24)
        target_v = targets[:, :, v_idx]     # (N, 24)
        mask_v = masks[:, :, v_idx]         # (N, 24)

        # Valid entries
        valid = mask_v > 0.5
        if valid.sum() == 0:
            metrics[v_name] = {'MAE': float('nan'), 'RMSE': float('nan'),
                               'MAPE': float('nan'), 'calibration_80': float('nan'),
                               'correlation': float('nan')}
            continue

        pred_valid = pred_v[valid]
        target_valid = target_v[valid]

        # MAE
        errors = np.abs(pred_valid - target_valid)
        mae = errors.mean()

        # RMSE
        rmse = np.sqrt(((pred_valid - target_valid) ** 2).mean())

        # MAPE (avoid division by zero)
        nonzero = np.abs(target_valid) > 1e-6
        if nonzero.sum() > 0:
            mape = (errors[nonzero] / np.abs(target_valid[nonzero])).mean() * 100
        else:
            mape = float('nan')

        # Correlation
        if len(pred_valid) > 1 and pred_valid.std() > 1e-8 and target_valid.std() > 1e-8:
            correlation = np.corrcoef(pred_valid, target_valid)[0, 1]
        else:
            correlation = float('nan')

        # Calibration: fraction of targets within 80% prediction interval (q0.1 to q0.9)
        q_low = predictions[:, :, v_idx, 0]   # (N, 24) - quantile 0.1
        q_high = predictions[:, :, v_idx, 2]  # (N, 24) - quantile 0.9
        in_interval = (targets[:, :, v_idx] >= q_low) & (targets[:, :, v_idx] <= q_high)
        calibration_80 = in_interval[valid].mean()

        metrics[v_name] = {
            'MAE': float(mae),
            'RMSE': float(rmse),
            'MAPE': float(mape),
            'calibration_80': float(calibration_80),
            'correlation': float(correlation),
            'n_valid': int(valid.sum()),
        }

        # Collect for overall
        all_errors.append(errors)
        all_sq_errors.append((pred_valid - target_valid) ** 2)

    # Overall metrics
    if all_errors:
        all_errors_cat = np.concatenate(all_errors)
        all_sq_errors_cat = np.concatenate(all_sq_errors)
        metrics['overall'] = {
            'MAE': float(all_errors_cat.mean()),
            'RMSE': float(np.sqrt(all_sq_errors_cat.mean())),
        }
    else:
        metrics['overall'] = {'MAE': float('nan'), 'RMSE': float('nan')}

    return metrics


@torch.no_grad()
def run_inference(model, test_loader, device):
    """Run model inference on test set."""
    model.eval()

    all_predictions = []
    all_targets = []
    all_masks = []

    for batch in test_loader:
        historical = batch['historical'].to(device)
        target = batch['target']        # keep on CPU
        target_mask = batch['target_mask']

        predictions = model(historical)  # (B, 24, 4, 3)
        all_predictions.append(predictions.cpu().numpy())
        all_targets.append(target.numpy())
        all_masks.append(target_mask.numpy())

    predictions = np.concatenate(all_predictions, axis=0)  # (N, 24, 4, 3)
    targets = np.concatenate(all_targets, axis=0)          # (N, 24, 4)
    masks = np.concatenate(all_masks, axis=0)              # (N, 24, 4)

    return predictions, targets, masks


def test(args):
    """Main test function."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Output directory
    output_dir = f"outputs/itransformer_epochs_{args.epochs}"
    model_path = os.path.join(output_dir, "best_model.pt")

    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        print(f"  Run train.py --epochs {args.epochs} first.")
        sys.exit(1)

    # Data
    print("\n--- Loading Data ---")
    train_loader, val_loader, test_loader, norm_params = create_dataloaders(
        batch_size=64, num_workers=4
    )

    # Also load saved norm_params (in case they differ)
    norm_path = os.path.join(output_dir, "norm_params.pt")
    if os.path.exists(norm_path):
        norm_params = torch.load(norm_path, map_location='cpu')
        print(f"  Loaded norm_params from: {norm_path}")

    # Model
    print("\n--- Loading Model ---")
    model = build_model(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"  Loaded weights from: {model_path}")

    # Inference
    print("\n--- Running Inference ---")
    predictions_norm, targets_norm, masks = run_inference(model, test_loader, device)
    print(f"  Predictions shape: {predictions_norm.shape}")
    print(f"  Targets shape:     {targets_norm.shape}")
    print(f"  Masks shape:       {masks.shape}")

    # Denormalize
    print("\n--- Denormalizing ---")
    predictions, targets = denormalize_vitals(predictions_norm, targets_norm, norm_params)
    print(f"  Vital means: {norm_params['vital_mean'].numpy()}")
    print(f"  Vital stds:  {norm_params['vital_std'].numpy()}")

    # Compute metrics
    print("\n--- Computing Metrics ---")
    metrics = compute_metrics(predictions, targets, masks)

    # Print results
    print(f"\n{'='*70}")
    print(f"  iTransformer Test Results (Phase 5 — Vitals-Only Ablation)")
    print(f"{'='*70}")
    print(f"  {'Vital':<12} {'MAE':>8} {'RMSE':>8} {'MAPE%':>8} {'Calib80':>8} {'Corr':>8}")
    print(f"  {'-'*56}")

    for v_name in VITAL_NAMES:
        m = metrics[v_name]
        print(f"  {v_name:<12} {m['MAE']:>8.2f} {m['RMSE']:>8.2f} "
              f"{m['MAPE']:>8.1f} {m['calibration_80']:>8.3f} {m['correlation']:>8.3f}")

    print(f"  {'-'*56}")
    print(f"  {'OVERALL':<12} {metrics['overall']['MAE']:>8.2f} {metrics['overall']['RMSE']:>8.2f}")
    print(f"{'='*70}")

    # Save outputs
    print(f"\n--- Saving Results ---")
    np.save(os.path.join(output_dir, "test_predictions.npy"), predictions)
    np.save(os.path.join(output_dir, "test_targets.npy"), targets)
    np.save(os.path.join(output_dir, "test_masks.npy"), masks)

    with open(os.path.join(output_dir, "test_metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"  Saved: test_predictions.npy  {predictions.shape}")
    print(f"  Saved: test_targets.npy      {targets.shape}")
    print(f"  Saved: test_masks.npy        {masks.shape}")
    print(f"  Saved: test_metrics.json")
    print(f"  Output dir: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test iTransformer ablation (vitals-only)")
    parser.add_argument('--epochs', type=int, default=100, help='Epochs (for output directory naming)')
    args = parser.parse_args()

    test(args)
