#!/usr/bin/env python3
"""
test.py - Standalone testing script for all 3 LSTM forecasting models.

Loads the best checkpoint for each signal (ABP, PLETH, II), evaluates on
the held-out test set, and saves predictions and targets as .npy files.

Usage:
    python test.py
"""

import os
import json
import numpy as np
import torch

from preprocess import create_dataloaders, FORECAST_HORIZON, NUM_SIGNALS, SIGNAL_NAMES
from model import LSTMForecaster, CHECKPOINT_DIR, OUTPUT_DIR


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def test_single_model(target_idx, device):
    """
    Load best checkpoint for a signal, evaluate on test set, return results.

    Args:
        target_idx: Index of target signal (0=ABP, 1=PLETH, 2=II).
        device: torch device.

    Returns:
        Tuple of (metrics_dict, predictions_original_scale, targets_original_scale).
    """
    signal_name = SIGNAL_NAMES[target_idx]
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f'best_model_{signal_name.lower()}.pt')

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Run model.py first to train all models."
        )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    print(f"\n[{signal_name}] Loaded checkpoint: epoch {checkpoint['epoch']}, "
          f"val_loss={checkpoint['val_loss']:.6f}")

    # Get normalization params
    norm_params = checkpoint['norm_params']
    target_mean = norm_params['means'][target_idx]
    target_std = norm_params['stds'][target_idx]
    print(f"[{signal_name}] Normalization: mean={target_mean:.2f}, std={target_std:.2f}")

    # Create model and load weights
    model = LSTMForecaster().to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Load test data for this target
    _, _, test_loader, _ = create_dataloaders(target_idx)
    print(f"[{signal_name}] Test set: {len(test_loader.dataset)} samples")

    # Run inference
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            predictions = model(x_batch)
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(y_batch.numpy())

    predictions_norm = np.concatenate(all_predictions, axis=0)
    targets_norm = np.concatenate(all_targets, axis=0)

    # Denormalize to original scale
    predictions_raw = predictions_norm * target_std + target_mean
    targets_raw = targets_norm * target_std + target_mean

    # Compute metrics
    mse_norm = float(np.mean((predictions_norm - targets_norm) ** 2))
    mae_norm = float(np.mean(np.abs(predictions_norm - targets_norm)))
    rmse_norm = float(np.sqrt(mse_norm))

    mse_raw = float(np.mean((predictions_raw - targets_raw) ** 2))
    mae_raw = float(np.mean(np.abs(predictions_raw - targets_raw)))
    rmse_raw = float(np.sqrt(mse_raw))

    metrics = {
        'signal_name': signal_name,
        'target_idx': target_idx,
        'mse_normalized': mse_norm,
        'mae_normalized': mae_norm,
        'rmse_normalized': rmse_norm,
        'mse_raw': mse_raw,
        'mae_raw': mae_raw,
        'rmse_raw': rmse_raw,
        'num_test_samples': int(predictions_raw.shape[0]),
        'forecast_horizon': FORECAST_HORIZON,
        'checkpoint_epoch': int(checkpoint['epoch']),
        'checkpoint_val_loss': float(checkpoint['val_loss']),
        'unit': 'mmHg' if signal_name == 'ABP' else 'normalized_unit',
    }

    return metrics, predictions_raw, targets_raw


def run_test():
    """Test all 3 models and save results."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {device}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_metrics = {}

    for target_idx in range(NUM_SIGNALS):
        signal_name = SIGNAL_NAMES[target_idx]

        metrics, predictions, targets = test_single_model(target_idx, device)
        all_metrics[signal_name] = metrics

        # Save predictions and targets per signal
        pred_path = os.path.join(OUTPUT_DIR, f'test_predictions_{signal_name.lower()}.npy')
        tgt_path = os.path.join(OUTPUT_DIR, f'test_targets_{signal_name.lower()}.npy')
        np.save(pred_path, predictions)
        np.save(tgt_path, targets)
        print(f"[SAVED] {pred_path}")
        print(f"[SAVED] {tgt_path}")

    # Save combined metrics
    metrics_path = os.path.join(OUTPUT_DIR, 'test_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n[SAVED] {metrics_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("TEST RESULTS — All Signals")
    print("=" * 60)
    print(f"{'Signal':<8} | {'MSE (norm)':>12} | {'MAE (norm)':>12} | "
          f"{'RMSE (norm)':>12} | {'MAE (raw)':>12} | {'Epoch':>6}")
    print("-" * 75)
    for name in SIGNAL_NAMES:
        m = all_metrics[name]
        print(f"{name:<8} | {m['mse_normalized']:>12.6f} | {m['mae_normalized']:>12.6f} | "
              f"{m['rmse_normalized']:>12.6f} | {m['mae_raw']:>12.4f} | {m['checkpoint_epoch']:>6}")
    print("-" * 75)
    print(f"\nTest samples per signal: {all_metrics[SIGNAL_NAMES[0]]['num_test_samples']}")
    print(f"Forecast horizon: {FORECAST_HORIZON} steps ({FORECAST_HORIZON/125*1000:.0f} ms)")
    print("=" * 60)
    print("\n[INFO] Run plot_predictions.py to visualize results.")

    return all_metrics


if __name__ == '__main__':
    run_test()
