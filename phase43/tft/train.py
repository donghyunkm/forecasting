#!/usr/bin/env python3
"""
train.py - Training script for TFT vital sign forecasting (Phase 4.3).

Phase 4.3 trains separate models for 1h/3h/6h forecast horizons.
Accepts --horizon flag to select prediction length.

Usage:
    python train.py --epochs 100 --horizon 6h
    python train.py --epochs 100 --horizon 1h
    python train.py --epochs 100 --horizon 3h
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Dict, Tuple
from omegaconf import OmegaConf

from model import TemporalFusionTransformer
from preprocess import create_dataloaders, HORIZON_MAP, PAST_MONTHS, NUM_SIGNALS, SIGNAL_NAMES


# =============================================================================
# Configuration
# =============================================================================
NUM_EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
MAX_GRAD_NORM = 100
DROPOUT = 0.3
STATE_SIZE = 240
LSTM_LAYERS = 2
ATTENTION_HEADS = 2
OUTPUT_QUANTILES = [0.1, 0.5, 0.9]
PATIENCE_LIMIT = 20

NUM_HISTORICAL_NUMERIC = 5   # 4 vitals + 1 time (no masks)
NUM_FUTURE_NUMERIC = 1
NUM_STATIC_NUMERIC = 1

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def get_checkpoint_dir(horizon, num_epochs=NUM_EPOCHS):
    return os.path.join(BASE_DIR, 'checkpoints', f'tft_{horizon}_epochs_{num_epochs}')


def get_output_dir(horizon, num_epochs=NUM_EPOCHS):
    return os.path.join(BASE_DIR, 'outputs', f'tft_{horizon}_epochs_{num_epochs}')


# =============================================================================
# Loss functions (exact copy from TFT-multi notebook)
# =============================================================================

def compute_quantile_loss_instance_wise(outputs, targets, masks, desired_quantiles, future_months):
    """
    Compute quantile loss with masking.

    outputs: [num_samples x num_horizons x num_features x num_quantiles]
    targets: [num_samples x num_horizons x num_features]
    masks: [num_samples x num_horizons x num_features]
    desired_quantiles: tensor of quantile values
    """
    errors = targets.unsqueeze(-1) - outputs
    # errors: [num_samples x num_horizons x num_features x num_quantiles]

    # mask to account for losses only on real values
    for i in range(masks.shape[-1]):
        for j in range(len(desired_quantiles)):
            errors[..., i, j] = errors[..., i, j] * masks[..., i]

    # compute the loss separately for each sample, time-step, quantile
    losses_array = torch.max((desired_quantiles - 1) * errors, desired_quantiles * errors)

    return losses_array


def get_quantiles_loss_and_q_risk(outputs, targets, masks, desired_quantiles, future_months):
    """
    Compute quantile loss and q-risk.
    """
    outputs = outputs.reshape((outputs.shape[0], future_months, NUM_SIGNALS, len(desired_quantiles)))
    losses_array = compute_quantile_loss_instance_wise(
        outputs=outputs, targets=targets, masks=masks,
        desired_quantiles=desired_quantiles, future_months=future_months)

    # sum losses over quantiles and average across time and observations
    q_loss = (losses_array.sum(dim=-1)).sum(dim=-1).mean(dim=-1).mean()

    # compute q_risk for each quantile
    q_risk = 2 * (losses_array.sum(dim=1).sum(dim=0)) / (targets.abs().sum().unsqueeze(-1))
    q_risk = q_risk.sum(dim=0)

    return q_loss, q_risk, losses_array


def process_batch(batch, model, quantiles_tensor, device, future_months):
    """
    Process a single batch.
    """
    if device.type == "cuda":
        for k in list(batch.keys()):
            batch[k] = batch[k].to(device)

    batch_outputs = model(batch)
    labels = batch['target']          # [batch, future_months, num_feat]
    target_masks = batch['target_mask']

    # [batch, future_months, num_feat*num_quantiles]
    predicted_quantiles = batch_outputs['predicted_quantiles']

    q_loss, q_risk, _ = get_quantiles_loss_and_q_risk(
        outputs=predicted_quantiles,
        targets=labels,
        masks=target_masks,
        desired_quantiles=quantiles_tensor,
        future_months=future_months)

    return q_loss, q_risk


# =============================================================================
# Training loop
# =============================================================================

def train_model(device, horizon, num_epochs=NUM_EPOCHS):
    """
    Train TFT model for a specific forecast horizon.
    """
    FUTURE_MONTHS = HORIZON_MAP[horizon]

    output_dir = get_output_dir(horizon, num_epochs)
    checkpoint_dir = get_checkpoint_dir(horizon, num_epochs)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Data
    print("[INFO] Loading data...")
    train_loader, val_loader, test_loader, norm_params = create_dataloaders(
        batch_size=BATCH_SIZE, horizon=horizon)

    # Model configuration
    data_props = {
        'num_historical_numeric': 5,  # 4 vitals + 1 time (no masks in Phase 4.3)
        'num_historical_categorical': 0,
        'historical_categorical_cardinalities': [],
        'num_static_numeric': NUM_STATIC_NUMERIC,  # 1
        'num_static_categorical': 0,
        'static_categorical_cardinalities': [],
        'num_future_numeric': NUM_FUTURE_NUMERIC,  # 1
        'num_future_categorical': 0,
        'future_categorical_cardinalities': [],
        'num_feature_predicted': 4,  # 4 vitals
    }

    configuration = {
        'optimization': {
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'max_grad_norm': MAX_GRAD_NORM,
        },
        'model': {
            'dropout': DROPOUT,
            'state_size': STATE_SIZE,
            'output_quantiles': OUTPUT_QUANTILES,
            'lstm_layers': LSTM_LAYERS,
            'attention_heads': ATTENTION_HEADS,
        },
        'task_type': 'regression',
        'target_window_start': None,
        'data_props': data_props,
    }

    config = OmegaConf.create(configuration)
    tft_model = TemporalFusionTransformer(config)
    tft_model.to(device)

    total_params = sum(p.numel() for p in tft_model.parameters())
    print(f"[INFO] Model params: {total_params:,}")
    print(f"[INFO] Config: state_size={STATE_SIZE}, lstm_layers={LSTM_LAYERS}, "
          f"heads={ATTENTION_HEADS}, dropout={DROPOUT}")

    # Optimizer
    opt = optim.Adam(
        filter(lambda p: p.requires_grad, tft_model.parameters()),
        lr=LEARNING_RATE)

    quantiles_tensor = torch.tensor(OUTPUT_QUANTILES).to(device)

    # Training loop with early stopping
    loss_arr = []
    loss_arr_test = []
    patience = 0
    min_loss = 9999
    best_model_state = None
    best_epoch = 0

    print(f"\n[INFO] Training for {num_epochs} epochs (patience={PATIENCE_LIMIT})...")
    print("-" * 70)
    print(f" {'Epoch':>5} | {'Train Loss':>12} | {'Val Loss':>12} | {'Status'}")
    print("-" * 70)

    start_time = time.time()

    for epoch in range(num_epochs):
        # Training
        tft_model.train()
        loss_e = 0

        for data in train_loader:
            opt.zero_grad()
            loss, _ = process_batch(batch=data, model=tft_model,
                                    quantiles_tensor=quantiles_tensor,
                                    device=device, future_months=FUTURE_MONTHS)
            loss_e += loss.item()
            loss.backward()

            if MAX_GRAD_NORM > 0:
                nn.utils.clip_grad_norm_(tft_model.parameters(), MAX_GRAD_NORM)

            opt.step()

        loss_arr.append(loss_e)

        # Validation
        tft_model.eval()
        loss_e_test = 0
        with torch.no_grad():
            for test_data in val_loader:
                batch_loss, _ = process_batch(batch=test_data, model=tft_model,
                                              quantiles_tensor=quantiles_tensor,
                                              device=device, future_months=FUTURE_MONTHS)
                loss_e_test += batch_loss.item()

        loss_arr_test.append(loss_e_test)

        # Early stopping
        status = ""
        if min_loss > loss_arr_test[-1]:
            min_loss = loss_arr_test[-1]
            best_model_state = tft_model.state_dict().copy()
            best_epoch = epoch + 1
            patience = 0
            status = "* best *"
        else:
            patience += 1
            if patience > PATIENCE_LIMIT:
                print(f"{epoch+1:>5} | {loss_arr[-1]:>12.4f} | {loss_arr_test[-1]:>12.4f} | "
                      f"Early stop (patience={PATIENCE_LIMIT})")
                break

        # Print progress
        if epoch < 5 or (epoch + 1) % 5 == 0 or status:
            print(f"{epoch+1:>5} | {loss_arr[-1]:>12.4f} | {loss_arr_test[-1]:>12.4f} | {status}")

    elapsed = time.time() - start_time
    print("-" * 70)
    print(f"[INFO] Training complete in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"[INFO] Best epoch: {best_epoch}, Best val loss: {min_loss:.6f}")

    # Save best model
    checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pt')
    torch.save({
        'model_state_dict': best_model_state,
        'config': OmegaConf.to_container(config, resolve=True),
        'epoch': best_epoch,
        'train_loss': loss_arr[best_epoch - 1] if best_epoch <= len(loss_arr) else loss_arr[-1],
        'val_loss': min_loss,
        'norm_params': norm_params,
        'num_epochs': num_epochs,
        'horizon': horizon,
        'future_months': FUTURE_MONTHS,
    }, checkpoint_path)
    print(f"[SAVED] {checkpoint_path}")

    # Plot training curves
    fig, ax = plt.subplots(figsize=(10, 5))
    epochs_x = np.arange(1, len(loss_arr) + 1)
    ax.plot(epochs_x, loss_arr, '-', color='tab:blue', label='Train Loss', alpha=0.8)
    ax.plot(epochs_x, loss_arr_test, '--', color='tab:red', label='Val Loss', alpha=0.8)
    ax.axvline(x=best_epoch, color='green', linestyle=':', label=f'Best (epoch {best_epoch})')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Quantile Loss')
    ax.set_title(f'TFT \u2014 {horizon} Forecast (Phase 4.3) \u2014 Training Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    curves_path = os.path.join(output_dir, 'training_curves.png')
    plt.savefig(curves_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {curves_path}")

    return loss_arr, loss_arr_test, best_epoch


def main():
    """Entry point for training."""
    parser = argparse.ArgumentParser(description='Train TFT for vital sign forecasting (Phase 4.3)')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS)
    parser.add_argument('--horizon', type=str, choices=['1h', '3h', '6h'], default='6h')
    args = parser.parse_args()

    horizon = args.horizon
    num_epochs = args.epochs
    FUTURE_MONTHS = HORIZON_MAP[horizon]

    print("=" * 60)
    print(f"TFT \u2014 {horizon} Forecast (Phase 4.3)")
    print("=" * 60)
    print(f"Vital signs: {SIGNAL_NAMES}")
    print(f"Input: {PAST_MONTHS} steps")
    print(f"Output: {FUTURE_MONTHS} steps \u00d7 {NUM_SIGNALS} vitals \u00d7 {len(OUTPUT_QUANTILES)} quantiles")
    print(f"Quantiles: {OUTPUT_QUANTILES}")
    print(f"Epochs: {num_epochs}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Device: {device}")

    train_model(device, horizon, num_epochs)


if __name__ == '__main__':
    main()
