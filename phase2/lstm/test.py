#!/usr/bin/env python3
"""
test.py - Standalone testing script for heart rate prediction LSTM model.

Loads the best checkpoint, evaluates on the held-out test set,
and saves predictions and targets as .npy files.

Usage:
    python test.py
    python test.py --epochs 50 --input-length 7500 --target-length 7500
"""

import os
import json
import numpy as np
import torch

from preprocess import create_dataloaders, INPUT_LENGTH, TARGET_LENGTH, NUM_SIGNALS, SIGNAL_NAMES
from model import LSTMHeartRate, get_checkpoint_dir, get_output_dir, NUM_EPOCHS


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def test_model(device, num_epochs=NUM_EPOCHS, input_length=INPUT_LENGTH, target_length=TARGET_LENGTH):
    """
    Load best checkpoint, evaluate on test set, return results.

    Args:
        device: torch device.
        num_epochs: Number of epochs used during training (for directory lookup).
        input_length: Input window length in samples.
        target_length: Target window length in samples.

    Returns:
        Tuple of (metrics_dict, predictions_bpm, targets_bpm).
    """
    checkpoint_dir = get_checkpoint_dir(num_epochs, input_length, target_length)
    checkpoint_path = os.path.join(checkpoint_dir, 'best_model_hr.pt')

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Run model.py first to train the model."
        )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    print(f"[HR] Loaded checkpoint: epoch {checkpoint['epoch']}, "
          f"val_loss={checkpoint['val_loss']:.6f}")

    # Get normalization params
    norm_params = checkpoint['norm_params']
    hr_mean = norm_params['hr_mean']
    hr_std = norm_params['hr_std']
    print(f"[HR] HR normalization: mean={hr_mean:.2f} BPM, std={hr_std:.2f} BPM")

    # Create model and load weights
    model = LSTMHeartRate().to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Load test data
    _, _, test_loader, _ = create_dataloaders(input_length=input_length, target_length=target_length)
    print(f"[HR] Test set: {len(test_loader.dataset)} samples")

    # Run inference
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            predictions = model(x_batch)
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(y_batch.numpy())

    predictions_norm = np.concatenate(all_predictions, axis=0)  # (N, 1)
    targets_norm = np.concatenate(all_targets, axis=0)  # (N, 1)

    # Denormalize to BPM
    predictions_bpm = predictions_norm * hr_std + hr_mean
    targets_bpm = targets_norm * hr_std + hr_mean

    # Flatten to 1D
    predictions_bpm = predictions_bpm.flatten()
    targets_bpm = targets_bpm.flatten()

    # Compute metrics
    mse_norm = float(np.mean((predictions_norm - targets_norm) ** 2))
    mae_norm = float(np.mean(np.abs(predictions_norm - targets_norm)))
    rmse_norm = float(np.sqrt(mse_norm))

    mse_bpm = float(np.mean((predictions_bpm - targets_bpm) ** 2))
    mae_bpm = float(np.mean(np.abs(predictions_bpm - targets_bpm)))
    rmse_bpm = float(np.sqrt(mse_bpm))

    # Percentage within thresholds
    errors_bpm = np.abs(predictions_bpm - targets_bpm)
    within_5bpm = float(np.mean(errors_bpm < 5.0) * 100)
    within_10bpm = float(np.mean(errors_bpm < 10.0) * 100)

    metrics = {
        'mse_normalized': mse_norm,
        'mae_normalized': mae_norm,
        'rmse_normalized': rmse_norm,
        'mse_bpm': mse_bpm,
        'mae_bpm': mae_bpm,
        'rmse_bpm': rmse_bpm,
        'within_5bpm_pct': within_5bpm,
        'within_10bpm_pct': within_10bpm,
        'num_test_samples': int(len(predictions_bpm)),
        'hr_mean_bpm': hr_mean,
        'hr_std_bpm': hr_std,
        'input_length': input_length,
        'target_length': target_length,
        'checkpoint_epoch': int(checkpoint['epoch']),
        'checkpoint_val_loss': float(checkpoint['val_loss']),
        'unit': 'BPM',
    }

    return metrics, predictions_bpm, targets_bpm


def run_test(num_epochs=NUM_EPOCHS, input_length=INPUT_LENGTH, target_length=TARGET_LENGTH):
    """Test model and save results.

    Args:
        num_epochs: Number of epochs used during training (for directory lookup).
        input_length: Input window length in samples.
        target_length: Target window length in samples.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Using device: {device}")

    output_dir = get_output_dir(num_epochs, input_length, target_length)
    os.makedirs(output_dir, exist_ok=True)

    metrics, predictions, targets = test_model(device, num_epochs, input_length, target_length)

    # Save predictions and targets
    pred_path = os.path.join(output_dir, 'test_predictions_hr.npy')
    tgt_path = os.path.join(output_dir, 'test_targets_hr.npy')
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
    print("TEST RESULTS — Heart Rate Prediction")
    print("=" * 60)
    print(f"  Input:  {input_length} samples ({input_length/125:.1f}s)")
    print(f"  Target: {target_length} samples ({target_length/125:.1f}s)")
    print(f"  ---")
    print(f"  MAE (BPM):  {metrics['mae_bpm']:.2f} BPM")
    print(f"  RMSE (BPM): {metrics['rmse_bpm']:.2f} BPM")
    print(f"  Within ±5 BPM:  {metrics['within_5bpm_pct']:.1f}%")
    print(f"  Within ±10 BPM: {metrics['within_10bpm_pct']:.1f}%")
    print(f"  ---")
    print(f"  Test samples:     {metrics['num_test_samples']}")
    print(f"  Best epoch:       {metrics['checkpoint_epoch']}")
    print(f"  HR range: mean={metrics['hr_mean_bpm']:.1f}, std={metrics['hr_std_bpm']:.1f} BPM")
    print("=" * 60)
    print("\n[INFO] Run plot_predictions.py to visualize results.")

    return metrics


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Test LSTM model for heart rate prediction')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS,
                        help=f'Number of epochs used during training (default: {NUM_EPOCHS})')
    parser.add_argument('--input-length', type=int, default=INPUT_LENGTH,
                        help=f'Input window length in samples (default: {INPUT_LENGTH})')
    parser.add_argument('--target-length', type=int, default=TARGET_LENGTH,
                        help=f'Target window length in samples (default: {TARGET_LENGTH})')
    args = parser.parse_args()
    run_test(num_epochs=args.epochs, input_length=args.input_length, target_length=args.target_length)
