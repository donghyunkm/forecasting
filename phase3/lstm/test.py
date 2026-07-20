#!/usr/bin/env python3
"""
test.py - Evaluation script for LSTM multivariate waveform forecasting (Phase 3).

Loads the best checkpoint for a given target signal, evaluates on the held-out
test set, and saves predictions, targets, and metrics.

Usage:
    python test.py --target II
    python test.py --target PLETH --epochs 50
    python test.py --target RESP
    python test.py --target ABP
"""

import os
import json
import numpy as np
import torch

from preprocess import (create_dataloaders, INPUT_LENGTH, OUTPUT_LENGTH,
                        NUM_SIGNALS, NUM_FEATURES, SIGNAL_NAMES, VALID_TARGETS,
                        INTERVAL_MINUTES, FEATURE_NAMES)
from model import LSTMForecaster, get_checkpoint_dir, get_output_dir, NUM_EPOCHS


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def test_model(device, target_signal='II', num_epochs=NUM_EPOCHS):
    """
    Load best checkpoint, evaluate on test set, return results.

    Args:
        device: torch device.
        target_signal: Target signal name.
        num_epochs: Number of epochs used during training.

    Returns:
        Tuple of (metrics_dict, predictions_denorm, targets_denorm).
        Predictions and targets have shape (N, 25, 6).
    """
    checkpoint_dir = get_checkpoint_dir(target_signal, num_epochs)
    checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pt')

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Run model.py first to train the model."
        )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    print(f"[INFO] Loaded checkpoint: epoch {checkpoint['epoch']}, "
          f"val_loss={checkpoint['val_loss']:.6f}")

    # Get normalization params
    norm_params = checkpoint['norm_params']
    target_idx = norm_params['target_signal_idx']
    norm_mean = np.array(norm_params['norm_mean'])  # (NUM_SIGNALS, NUM_FEATURES)
    norm_std = np.array(norm_params['norm_std'])    # (NUM_SIGNALS, NUM_FEATURES)

    # Denormalization parameters for the target signal — all 6 features
    target_means = norm_mean[target_idx, :]  # (6,)
    target_stds = norm_std[target_idx, :]    # (6,)
    print(f"[INFO] Target: {target_signal} — {NUM_FEATURES} features")
    for i, fname in enumerate(FEATURE_NAMES):
        print(f"       {fname}: mean={target_means[i]:.4f}, std={target_stds[i]:.4f}")

    # Create model and load weights
    model = LSTMForecaster().to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Load test data
    _, _, test_loader, _ = create_dataloaders(target_signal=target_signal)
    print(f"[INFO] Test set: {len(test_loader.dataset)} samples")

    # Run inference
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            predictions = model(x_batch)  # (batch, 25, 6)
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(y_batch.numpy())

    predictions_norm = np.concatenate(all_predictions, axis=0)  # (N, 25, 6)
    targets_norm = np.concatenate(all_targets, axis=0)          # (N, 25, 6)

    # Denormalize to original units — broadcast (6,) over (N, 25, 6)
    predictions_denorm = predictions_norm * target_stds[None, None, :] + target_means[None, None, :]
    targets_denorm = targets_norm * target_stds[None, None, :] + target_means[None, None, :]

    # Overall metrics (across all features)
    mse_norm = float(np.mean((predictions_norm - targets_norm) ** 2))
    mae_norm = float(np.mean(np.abs(predictions_norm - targets_norm)))
    rmse_norm = float(np.sqrt(mse_norm))

    mse_denorm = float(np.mean((predictions_denorm - targets_denorm) ** 2))
    mae_denorm = float(np.mean(np.abs(predictions_denorm - targets_denorm)))
    rmse_denorm = float(np.sqrt(mse_denorm))

    # Per-feature metrics
    per_feature_mae = {}
    per_feature_rmse = {}
    per_feature_corr = {}
    for f_idx, fname in enumerate(FEATURE_NAMES):
        pred_f = predictions_denorm[:, :, f_idx].flatten()
        tgt_f = targets_denorm[:, :, f_idx].flatten()
        per_feature_mae[fname] = float(np.mean(np.abs(pred_f - tgt_f)))
        per_feature_rmse[fname] = float(np.sqrt(np.mean((pred_f - tgt_f) ** 2)))
        per_feature_corr[fname] = float(np.corrcoef(pred_f, tgt_f)[0, 1])

    # Per-step MAE (averaged across features)
    per_step_mae = np.mean(np.abs(predictions_denorm - targets_denorm), axis=(0, 2))  # (25,)
    per_step_rmse = np.sqrt(np.mean((predictions_denorm - targets_denorm) ** 2, axis=(0, 2)))

    # Overall correlation
    overall_corr = float(np.corrcoef(predictions_denorm.flatten(), targets_denorm.flatten())[0, 1])

    metrics = {
        'target_signal': target_signal,
        'mse_normalized': mse_norm,
        'mae_normalized': mae_norm,
        'rmse_normalized': rmse_norm,
        'mse_original_units': mse_denorm,
        'mae_original_units': mae_denorm,
        'rmse_original_units': rmse_denorm,
        'correlation': overall_corr,
        'per_feature_mae': per_feature_mae,
        'per_feature_rmse': per_feature_rmse,
        'per_feature_corr': per_feature_corr,
        'per_step_mae': per_step_mae.tolist(),
        'per_step_rmse': per_step_rmse.tolist(),
        'num_test_samples': int(len(predictions_norm)),
        'forecast_steps': OUTPUT_LENGTH,
        'num_features': NUM_FEATURES,
        'feature_names': FEATURE_NAMES,
        'input_steps': INPUT_LENGTH,
        'interval_minutes': INTERVAL_MINUTES,
        'target_norm_means': target_means.tolist(),
        'target_norm_stds': target_stds.tolist(),
        'checkpoint_epoch': int(checkpoint['epoch']),
        'checkpoint_val_loss': float(checkpoint['val_loss']),
    }

    return metrics, predictions_denorm, targets_denorm


def run_test(target_signal='II', num_epochs=NUM_EPOCHS):
    """Test model and save results.

    Args:
        target_signal: Target signal name.
        num_epochs: Number of epochs used during training.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {device}")

    output_dir = get_output_dir(target_signal, num_epochs)
    os.makedirs(output_dir, exist_ok=True)

    metrics, predictions, targets = test_model(device, target_signal, num_epochs)

    # Save predictions and targets (in original units)
    pred_path = os.path.join(output_dir, 'test_predictions.npy')
    tgt_path = os.path.join(output_dir, 'test_targets.npy')
    np.save(pred_path, predictions)
    np.save(tgt_path, targets)
    print(f"[SAVED] {pred_path}")
    print(f"[SAVED] {tgt_path}")

    # Save metrics
    metrics_path = os.path.join(output_dir, 'test_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"[SAVED] {metrics_path}")

    # Print summary
    print("\n" + "=" * 60)
    print(f"TEST RESULTS — {target_signal} Forecasting (LSTM)")
    print("=" * 60)
    print(f"  Input:  {INPUT_LENGTH} intervals ({INPUT_LENGTH * INTERVAL_MINUTES / 60:.1f} hrs)")
    print(f"  Output: {OUTPUT_LENGTH} intervals × {NUM_FEATURES} features "
          f"({OUTPUT_LENGTH * INTERVAL_MINUTES / 60:.1f} hrs)")
    print(f"  ---")
    print(f"  Overall MAE (normalized):  {metrics['mae_normalized']:.4f}")
    print(f"  Overall RMSE (normalized): {metrics['rmse_normalized']:.4f}")
    print(f"  Overall MAE (original):    {metrics['mae_original_units']:.4f}")
    print(f"  Overall RMSE (original):   {metrics['rmse_original_units']:.4f}")
    print(f"  Overall Correlation:       {metrics['correlation']:.4f}")
    print(f"  ---")
    print(f"  Per-feature MAE (original units):")
    for fname in FEATURE_NAMES:
        print(f"    {fname:>10}: MAE={metrics['per_feature_mae'][fname]:.4f}  "
              f"RMSE={metrics['per_feature_rmse'][fname]:.4f}  "
              f"r={metrics['per_feature_corr'][fname]:.4f}")
    print(f"  ---")
    print(f"  Test samples:      {metrics['num_test_samples']}")
    print(f"  Best epoch:        {metrics['checkpoint_epoch']}")
    print("=" * 60)
    print("\n[INFO] Run plot_predictions.py to visualize results.")

    return metrics


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Test LSTM forecaster')
    parser.add_argument('--target', type=str, default='II', choices=VALID_TARGETS,
                        help=f'Target signal (default: II)')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS,
                        help=f'Number of epochs used during training (default: {NUM_EPOCHS})')
    args = parser.parse_args()
    run_test(target_signal=args.target, num_epochs=args.epochs)
