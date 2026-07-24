#!/usr/bin/env python3
"""
Phase 6.1 TFT Training: Temporal Fusion Transformer for correlation forecasting
with physiological feature inputs.

Input: 48 steps (2h) × 46 features (7 correlations + 38 physio stats + 1 time)
Output: 12 steps (30min) × 7 correlations (point predictions, Huber loss in Fisher z-space)
"""

import os
import sys
import argparse
import time
import json
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

from model import TemporalFusionTransformer
from preprocess import create_dataloaders


def get_config():
    """Build TFT configuration for Phase 6.1 (correlation + physio features)."""
    config = OmegaConf.create({
        'task_type': 'regression',
        'target_window_start': None,
        'data_props': {
            'num_historical_numeric': 46,     # 7 correlations + 38 physio stats + 1 time
            'num_historical_categorical': 0,
            'historical_categorical_cardinalities': [],
            'num_static_numeric': 1,          # placeholder
            'num_static_categorical': 0,
            'static_categorical_cardinalities': [],
            'num_future_numeric': 1,          # time position (known into future)
            'num_future_categorical': 0,
            'future_categorical_cardinalities': [],
            'num_feature_predicted': 7,       # predict all 7 correlations
        },
        'model': {
            'state_size': 240,
            'dropout': 0.3,
            'lstm_layers': 2,
            'attention_heads': 2,
            'output_quantiles': [0.5],        # single output per feature = point prediction
        },
    })
    return config


def train_one_epoch(model, train_loader, optimizer, device, max_grad_norm):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    criterion = nn.HuberLoss(delta=1.0)

    for batch in train_loader:
        # Move to device
        batch_device = {k: v.to(device) for k, v in batch.items()}

        optimizer.zero_grad()

        output = model(batch_device)
        predictions = output['predicted_quantiles']  # (batch, 12, 7)
        targets = batch_device['target']             # (batch, 12, 7)

        loss = criterion(predictions, targets)
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def validate(model, val_loader, device):
    """Validate model."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    criterion = nn.HuberLoss(delta=1.0)

    with torch.no_grad():
        for batch in val_loader:
            batch_device = {k: v.to(device) for k, v in batch.items()}

            output = model(batch_device)
            predictions = output['predicted_quantiles']
            targets = batch_device['target']

            loss = criterion(predictions, targets)
            total_loss += loss.item()
            n_batches += 1

    return total_loss / n_batches


def save_training_curves(train_losses, val_losses, output_dir):
    """Save training and validation loss curves."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)

    ax.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=1.5)
    ax.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Huber Loss')
    ax.set_title('Phase 6.1 TFT Training Curves (Correlation + Physio Forecasting)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Mark best epoch
    best_epoch = np.argmin(val_losses) + 1
    best_val = min(val_losses)
    ax.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.5, label=f'Best (epoch {best_epoch})')
    ax.scatter([best_epoch], [best_val], color='g', s=100, zorder=5)
    ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_curves.png"), dpi=150)
    plt.close()
    print(f"  Saved training_curves.png")


def main():
    parser = argparse.ArgumentParser(description='Phase 6.1 TFT Training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    args = parser.parse_args()

    # ─── Hyperparameters ──────────────────────────────────────────────────────
    EPOCHS = args.epochs
    BATCH_SIZE = 64
    LR = 1e-3
    MAX_GRAD_NORM = 100.0
    PATIENCE = 20

    # ─── Directories ──────────────────────────────────────────────────────────
    base_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_dir = os.path.join(base_dir, f"checkpoints/tft_epochs_{EPOCHS}")
    output_dir = os.path.join(base_dir, f"outputs/tft_epochs_{EPOCHS}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # ─── Device ───────────────────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # ─── Data ─────────────────────────────────────────────────────────────────
    print("\nLoading data...")
    train_loader, val_loader, test_loader, norm_params = create_dataloaders(
        batch_size=BATCH_SIZE)

    # ─── Model ────────────────────────────────────────────────────────────────
    config = get_config()
    model = TemporalFusionTransformer(config).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {num_params:,}")
    print(f"Input: 48 steps × 46 features (7 corr + 38 physio + 1 time)")
    print(f"Output: 12 steps × 7 correlations")
    print(f"Loss: Huber (delta=1.0) in Fisher z-space")

    # ─── Optimizer ────────────────────────────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # ─── Training Loop ────────────────────────────────────────────────────────
    print(f"\nTraining for {EPOCHS} epochs (patience={PATIENCE})...")
    print("-" * 60)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, device, MAX_GRAD_NORM)
        val_loss = validate(model, val_loader, device)

        elapsed = time.time() - t0

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Check improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0

            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': OmegaConf.to_container(config),
            }, os.path.join(checkpoint_dir, "best_model.pt"))
        else:
            patience_counter += 1

        # Print progress
        marker = " *" if epoch == best_epoch else ""
        print(f"  Epoch {epoch:3d}/{EPOCHS} | "
              f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
              f"Time: {elapsed:.1f}s | Patience: {patience_counter}/{PATIENCE}{marker}")

        # Early stopping
        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}. Best epoch: {best_epoch}")
            break

    # ─── Save Results ─────────────────────────────────────────────────────────
    print("\n" + "-" * 60)
    print(f"Best epoch: {best_epoch}, Val loss: {best_val_loss:.6f}")

    # Save training curves
    save_training_curves(train_losses, val_losses, output_dir)

    # Save training history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'total_epochs': len(train_losses),
        'config': {
            'epochs_max': EPOCHS,
            'batch_size': BATCH_SIZE,
            'lr': LR,
            'max_grad_norm': MAX_GRAD_NORM,
            'patience': PATIENCE,
            'loss': 'Huber (delta=1.0)',
            'num_input_features': 46,
        },
    }
    with open(os.path.join(output_dir, "training_history.json"), 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\nCheckpoint: {checkpoint_dir}/best_model.pt")
    print(f"Outputs:    {output_dir}/")
    print("Done!")


if __name__ == "__main__":
    main()
