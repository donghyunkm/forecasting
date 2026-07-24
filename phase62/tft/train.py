#!/usr/bin/env python3
"""
Phase 6.2 TFT Training: Classification of cluster labels from correlation history.

Input: 48 steps (2h) × 8 features (7 correlations + 1 time)
Output: 12 steps (30min) × 7 classes (cluster label classification)
Loss: Weighted cross-entropy
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
    """Build TFT configuration for Phase 6.2 (cluster classification)."""
    config = OmegaConf.create({
        'task_type': 'classification',
        'target_window_start': None,
        'data_props': {
            'num_historical_numeric': 8,      # 7 correlations + 1 time
            'num_historical_categorical': 0,
            'historical_categorical_cardinalities': [],
            'num_static_numeric': 1,          # placeholder
            'num_static_categorical': 0,
            'static_categorical_cardinalities': [],
            'num_future_numeric': 1,          # time position (known into future)
            'num_future_categorical': 0,
            'future_categorical_cardinalities': [],
            'num_classes': 7,                 # 7 cluster classes
        },
        'model': {
            'state_size': 240,
            'dropout': 0.3,
            'lstm_layers': 2,
            'attention_heads': 2,
            'output_quantiles': [0.5],        # unused for classification
        },
    })
    return config


def train_one_epoch(model, train_loader, optimizer, criterion, device, max_grad_norm):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    n_batches = 0

    for batch in train_loader:
        batch_device = {k: v.to(device) for k, v in batch.items()}

        optimizer.zero_grad()

        output = model(batch_device)
        logits = output['logits']           # (B, 12, 7)
        targets = batch_device['target']    # (B, 12) int64

        # Reshape for cross-entropy: (B*12, 7) vs (B*12,)
        B, T, C = logits.shape
        loss = criterion(logits.reshape(B * T, C), targets.reshape(B * T))
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        # Accuracy
        preds = logits.argmax(dim=-1)  # (B, 12)
        total_correct += (preds == targets).sum().item()
        total_samples += B * T

        total_loss += loss.item()
        n_batches += 1

    acc = total_correct / max(total_samples, 1)
    return total_loss / n_batches, acc


def validate(model, val_loader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    n_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            batch_device = {k: v.to(device) for k, v in batch.items()}

            output = model(batch_device)
            logits = output['logits']
            targets = batch_device['target']

            B, T, C = logits.shape
            loss = criterion(logits.reshape(B * T, C), targets.reshape(B * T))

            preds = logits.argmax(dim=-1)
            total_correct += (preds == targets).sum().item()
            total_samples += B * T

            total_loss += loss.item()
            n_batches += 1

    acc = total_correct / max(total_samples, 1)
    return total_loss / n_batches, acc


def save_training_curves(train_losses, val_losses, train_accs, val_accs, output_dir):
    """Save training and validation curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(train_losses) + 1)

    ax1.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=1.5)
    ax1.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=1.5)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Cross-Entropy Loss')
    ax1.set_title('Phase 6.2 TFT — Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, train_accs, 'b-', label='Train Acc', linewidth=1.5)
    ax2.plot(epochs, val_accs, 'r-', label='Val Acc', linewidth=1.5)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Phase 6.2 TFT — Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    best_epoch = np.argmin(val_losses) + 1
    ax1.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.5)
    ax2.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_curves.png"), dpi=150)
    plt.close()
    print(f"  Saved training_curves.png")


def main():
    parser = argparse.ArgumentParser(description='Phase 6.2 TFT Training')
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()

    EPOCHS = args.epochs
    BATCH_SIZE = 64
    LR = 1e-3
    MAX_GRAD_NORM = 100.0
    PATIENCE = 20

    base_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_dir = os.path.join(base_dir, f"checkpoints/tft_epochs_{EPOCHS}")
    output_dir = os.path.join(base_dir, f"outputs/tft_epochs_{EPOCHS}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    print("\nLoading data...")
    train_loader, val_loader, test_loader, norm_params = create_dataloaders(batch_size=BATCH_SIZE)

    # Class weights for imbalanced data
    class_weights = torch.tensor(norm_params['class_weights'], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    print(f"  Class weights: {norm_params['class_weights']}")

    config = get_config()
    model = TemporalFusionTransformer(config).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {num_params:,}")
    print(f"Task: Classification (7 classes)")
    print(f"Loss: Weighted cross-entropy")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print(f"\nTraining for {EPOCHS} epochs (patience={PATIENCE})...")
    print("-" * 70)

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device, MAX_GRAD_NORM)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        elapsed = time.time() - t0

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'config': OmegaConf.to_container(config),
            }, os.path.join(checkpoint_dir, "best_model.pt"))
        else:
            patience_counter += 1

        marker = " *" if epoch == best_epoch else ""
        print(f"  Epoch {epoch:3d}/{EPOCHS} | "
              f"Train: {train_loss:.4f} (acc={train_acc:.4f}) | "
              f"Val: {val_loss:.4f} (acc={val_acc:.4f}) | "
              f"Time: {elapsed:.1f}s | Pat: {patience_counter}/{PATIENCE}{marker}")

        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}. Best epoch: {best_epoch}")
            break

    print("\n" + "-" * 70)
    print(f"Best epoch: {best_epoch}, Val loss: {best_val_loss:.6f}")

    save_training_curves(train_losses, val_losses, train_accs, val_accs, output_dir)

    history = {
        'train_losses': train_losses, 'val_losses': val_losses,
        'train_accs': train_accs, 'val_accs': val_accs,
        'best_epoch': best_epoch, 'best_val_loss': best_val_loss,
        'total_epochs': len(train_losses),
        'config': {'epochs_max': EPOCHS, 'batch_size': BATCH_SIZE, 'lr': LR,
                   'max_grad_norm': MAX_GRAD_NORM, 'patience': PATIENCE,
                   'loss': 'weighted cross-entropy'},
    }
    with open(os.path.join(output_dir, "training_history.json"), 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\nCheckpoint: {checkpoint_dir}/best_model.pt")
    print("Done!")


if __name__ == "__main__":
    main()
