#!/usr/bin/env python3
"""
test.py - Evaluation for TFT vital sign forecasting (Phase 4.3).

Phase 4.3 trains separate models for 1h/3h/6h forecast horizons.
Accepts --horizon flag to select which model to evaluate.

Metrics:
  - Per-vital MAE/MAPE for each quantile (computed only on real/masked values)
  - Calibration: % of true values within 10th-90th prediction interval (target: 80%)
  - Correlation: Pearson r on median predictions
  - Overall MAE, RMSE, calibration

Saves:
  - test_predictions.npy: (N, FUTURE_MONTHS, 4, 3)
  - test_targets.npy: (N, FUTURE_MONTHS, 4)
  - test_masks.npy: (N, FUTURE_MONTHS, 4)
  - test_metrics.json

Usage:
    python test.py --epochs 100 --horizon 6h
"""

import os
import json
import argparse
import numpy as np
import torch
from omegaconf import OmegaConf

from model import TemporalFusionTransformer
from preprocess import (create_dataloaders, HORIZON_MAP, PAST_MONTHS,
                        NUM_SIGNALS, SIGNAL_NAMES)
from train import (get_checkpoint_dir, get_output_dir, process_batch,
                   NUM_EPOCHS, OUTPUT_QUANTILES, BATCH_SIZE)


def test_model(device, horizon, num_epochs=NUM_EPOCHS):
    """
    Load best checkpoint, run inference on test set, compute metrics.
    """
    FUTURE_MONTHS = HORIZON_MAP[horizon]
    pred_len = FUTURE_MONTHS

    checkpoint_dir = get_checkpoint_dir(horizon, num_epochs)
    checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pt')

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\nRun train.py first.")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    print(f"[INFO] Loaded checkpoint: epoch {checkpoint['epoch']}, "
          f"train_loss={checkpoint['train_loss']:.6f}")

    # Get normalization params from Phase 4.3 processed data
    norm_params_path = f'/gpfs/scratch/dk5565/phase43_data/processed/horizon_{pred_len}/norm_params.json'
    with open(norm_params_path) as f:
        norm_params = json.load(f)
    norm_mean = np.array(norm_params['mean'])   # (4,)
    norm_std = np.array(norm_params['std'])     # (4,)

    # Recreate model
    config = OmegaConf.create(checkpoint['config'])
    model = TemporalFusionTransformer(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Load test data
    _, _, test_loader, _ = create_dataloaders(batch_size=BATCH_SIZE, horizon=horizon)
    print(f"[INFO] Test set: {len(test_loader.dataset)} samples")

    # Run inference
    quantiles_tensor = torch.tensor(OUTPUT_QUANTILES).to(device)
    all_predictions = []
    all_targets = []
    all_masks = []

    with torch.no_grad():
        for data in test_loader:
            if device.type == "cuda":
                for k in list(data.keys()):
                    data[k] = data[k].to(device)

            batch_outputs = model(data)
            prediction = batch_outputs['predicted_quantiles'].cpu().numpy()
            true = data['target'].cpu().numpy()
            mask = data['target_mask'].cpu().numpy()

            all_predictions.append(prediction)
            all_targets.append(true)
            all_masks.append(mask)

    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    masks = np.concatenate(all_masks, axis=0)

    # Reshape predictions: (N, FUTURE_MONTHS*4*3) → (N, FUTURE_MONTHS, 4, 3)
    predictions = predictions.reshape(
        predictions.shape[0], FUTURE_MONTHS, NUM_SIGNALS, len(OUTPUT_QUANTILES))

    # Denormalize predictions and targets
    predictions_denorm = (predictions * norm_std[None, None, :, None]
                          + norm_mean[None, None, :, None])
    targets_denorm = targets * norm_std[None, None, :] + norm_mean[None, None, :]

    # =========================================================================
    # Metrics
    # =========================================================================
    num_quantiles = len(OUTPUT_QUANTILES)

    per_vital_mae = {}
    per_vital_mape = {}
    per_vital_calibration = {}

    for meas in range(NUM_SIGNALS):
        name = SIGNAL_NAMES[meas]
        pred_meas = predictions_denorm[:, :, meas, :]
        true_meas = targets_denorm[:, :, meas]
        mask_meas = masks[:, :, meas]

        mae_per_q = np.zeros(num_quantiles)
        mape_per_q = np.zeros(num_quantiles)
        within_bound_count = 0
        valid_count = 0

        for i in range(pred_meas.shape[0]):
            for j in range(FUTURE_MONTHS):
                if mask_meas[i, j] == 1 and true_meas[i, j] > 0:
                    valid_count += 1
                    y = true_meas[i, j]

                    for q_idx in range(num_quantiles):
                        mae_per_q[q_idx] += np.abs(pred_meas[i, j, q_idx] - y)
                        mape_per_q[q_idx] += np.abs(
                            (pred_meas[i, j, q_idx] - y) / y)

                    # Check if within 10th-90th bounds
                    if pred_meas[i, j, 0] <= y <= pred_meas[i, j, 2]:
                        within_bound_count += 1

        if valid_count > 0:
            mae_per_q /= valid_count
            mape_per_q /= valid_count
            calibration = within_bound_count / valid_count
        else:
            calibration = 0.0

        per_vital_mae[name] = {f'q{q:.1f}': float(mae_per_q[qi])
                               for qi, q in enumerate(OUTPUT_QUANTILES)}
        per_vital_mape[name] = {f'q{q:.1f}': float(mape_per_q[qi])
                                for qi, q in enumerate(OUTPUT_QUANTILES)}
        per_vital_calibration[name] = float(calibration)

    # Overall metrics (50th percentile = median)
    median_idx = OUTPUT_QUANTILES.index(0.5)
    pred_median = predictions_denorm[:, :, :, median_idx]

    # Masked overall MAE/RMSE
    valid_mask = masks.astype(bool) & (targets_denorm > 0)
    overall_mae = float(np.mean(
        np.abs(pred_median[valid_mask] - targets_denorm[valid_mask])))
    overall_rmse = float(np.sqrt(np.mean(
        (pred_median[valid_mask] - targets_denorm[valid_mask]) ** 2)))

    # Per-vital correlation
    per_vital_corr = {}
    for s, name in enumerate(SIGNAL_NAMES):
        v_mask = masks[:, :, s].astype(bool) & (targets_denorm[:, :, s] > 0)
        pred_s = pred_median[:, :, s][v_mask]
        tgt_s = targets_denorm[:, :, s][v_mask]
        if len(pred_s) > 1 and np.std(pred_s) > 0 and np.std(tgt_s) > 0:
            per_vital_corr[name] = float(np.corrcoef(pred_s, tgt_s)[0, 1])
        else:
            per_vital_corr[name] = 0.0

    # Overall calibration
    overall_calibration = float(
        np.mean([per_vital_calibration[n] for n in SIGNAL_NAMES]))

    metrics = {
        'overall_mae_median': overall_mae,
        'overall_rmse_median': overall_rmse,
        'overall_calibration': overall_calibration,
        'per_vital_mae': per_vital_mae,
        'per_vital_mape': per_vital_mape,
        'per_vital_calibration': per_vital_calibration,
        'per_vital_correlation': per_vital_corr,
        'num_test_samples': int(len(predictions_denorm)),
        'forecast_steps': FUTURE_MONTHS,
        'forecast_hours': FUTURE_MONTHS * 15 / 60,
        'num_signals': NUM_SIGNALS,
        'signal_names': SIGNAL_NAMES,
        'quantiles': OUTPUT_QUANTILES,
        'input_steps': PAST_MONTHS,
        'input_hours': PAST_MONTHS * 15 / 60,
        'resolution_minutes': 15,
        'horizon': horizon,
        'norm_mean': norm_mean.tolist(),
        'norm_std': norm_std.tolist(),
        'checkpoint_epoch': int(checkpoint['epoch']),
    }

    return metrics, predictions_denorm, targets_denorm, masks


def run_test(horizon, num_epochs=NUM_EPOCHS):
    """Run evaluation and save results."""
    FUTURE_MONTHS = HORIZON_MAP[horizon]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Device: {device}")

    output_dir = get_output_dir(horizon, num_epochs)
    os.makedirs(output_dir, exist_ok=True)

    metrics, predictions, targets, masks = test_model(device, horizon, num_epochs)

    # Save predictions and targets
    pred_path = os.path.join(output_dir, 'test_predictions.npy')
    tgt_path = os.path.join(output_dir, 'test_targets.npy')
    mask_path = os.path.join(output_dir, 'test_masks.npy')
    np.save(pred_path, predictions)
    np.save(tgt_path, targets)
    np.save(mask_path, masks)
    print(f"[SAVED] {pred_path}  shape={predictions.shape}")
    print(f"[SAVED] {tgt_path}  shape={targets.shape}")
    print(f"[SAVED] {mask_path}  shape={masks.shape}")

    # Save metrics
    metrics_path = os.path.join(output_dir, 'test_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"[SAVED] {metrics_path}")

    # Print summary
    print("\n" + "=" * 60)
    print(f"TEST RESULTS \u2014 TFT Vital Sign Forecasting (Phase 4.3, {horizon})")
    print("=" * 60)
    print(f"  Horizon: {horizon} ({FUTURE_MONTHS} steps)")
    print(f"  Resolution: 15 minutes")
    print(f"  Input:  {PAST_MONTHS} steps ({PAST_MONTHS * 15 / 60:.1f} hours)")
    print(f"  Output: {FUTURE_MONTHS} steps ({FUTURE_MONTHS * 15 / 60:.1f} hours) "
          f"\u00d7 {NUM_SIGNALS} vitals \u00d7 {len(OUTPUT_QUANTILES)} quantiles")
    print(f"  Test samples: {metrics['num_test_samples']}")
    print(f"  Best epoch:   {metrics['checkpoint_epoch']}")
    print(f"  ---")
    print(f"  Overall MAE (median):  {metrics['overall_mae_median']:.4f}")
    print(f"  Overall RMSE (median): {metrics['overall_rmse_median']:.4f}")
    print(f"  Overall Calibration:   {metrics['overall_calibration']:.4f}")
    print(f"  ---")
    print(f"  Per-vital sign (MAE q0.5 / MAPE q0.5 / Calibration / Correlation):")
    for name in SIGNAL_NAMES:
        mae_q5 = metrics['per_vital_mae'][name].get('q0.5', 0)
        mape_q5 = metrics['per_vital_mape'][name].get('q0.5', 0)
        calib = metrics['per_vital_calibration'][name]
        corr = metrics['per_vital_correlation'][name]
        print(f"    {name:>20}: MAE={mae_q5:.4f}  MAPE={mape_q5:.4f}  "
              f"Calib={calib:.4f}  r={corr:.4f}")
    print("=" * 60)

    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate TFT forecaster (Phase 4.3 \u2014 multi-horizon)')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS)
    parser.add_argument('--horizon', type=str, choices=['1h', '3h', '6h'], default='6h')
    args = parser.parse_args()
    run_test(horizon=args.horizon, num_epochs=args.epochs)
