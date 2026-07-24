#!/usr/bin/env python3
"""
Phase 6.2 v3 TFT Training: Cluster classification with label history + X_stats input.

Input: 48 steps × 47 features (7 corr + 38 physio + 1 time + 1 label history)
Output: 12 steps × 7 classes (weighted cross-entropy)
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
    config = OmegaConf.create({
        'task_type': 'classification',
        'target_window_start': None,
        'data_props': {
            'num_historical_numeric': 47,     # 7 corr + 38 physio + 1 time + 1 label
            'num_historical_categorical': 0,
            'historical_categorical_cardinalities': [],
            'num_static_numeric': 1,
            'num_static_categorical': 0,
            'static_categorical_cardinalities': [],
            'num_future_numeric': 1,
            'num_future_categorical': 0,
            'future_categorical_cardinalities': [],
            'num_classes': 7,
        },
        'model': {
            'state_size': 240,
            'dropout': 0.3,
            'lstm_layers': 2,
            'attention_heads': 2,
            'output_quantiles': [0.5],
        },
    })
    return config


def train_one_epoch(model, train_loader, optimizer, criterion, device, max_grad_norm):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    n_batches = 0

    for batch in train_loader:
        batch_device = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()

        output = model(batch_device)
        logits = output['logits']
        targets = batch_device['target']

        B, T, C = logits.shape
        loss = criterion(logits.reshape(B * T, C), targets.reshape(B * T))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        preds = logits.argmax(dim=-1)
        total_correct += (preds == targets).sum().item()
        total_samples += B * T
        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches, total_correct / max(total_samples, 1)


def validate(model, val_loader, criterion, device):
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

    return total_loss / n_batches, total_correct / max(total_samples, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()

    EPOCHS = args.epochs
    BATCH_SIZE = 64
    LR = 1e-3
    MAX_GRAD_NORM = 100.0
    PATIENCE = 20

    base_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_dir = os.path.join(base_dir, f"checkpoints/tft_v3_epochs_{EPOCHS}")
    output_dir = os.path.join(base_dir, f"outputs/tft_v3_epochs_{EPOCHS}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print("\nLoading data (v3 — with label history + X_stats)...")
    train_loader, val_loader, test_loader, norm_params = create_dataloaders(batch_size=BATCH_SIZE)

    class_weights = torch.tensor(norm_params['class_weights'], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    config = get_config()
    model = TemporalFusionTransformer(config).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    print(f"Input: 48 × 47 (7 corr + 38 physio + 1 time + 1 label history)")

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
                'epoch': epoch, 'model_state_dict': model.state_dict(),
                'val_loss': val_loss, 'val_acc': val_acc,
                'config': OmegaConf.to_container(config),
            }, os.path.join(checkpoint_dir, "best_model.pt"))
        else:
            patience_counter += 1

        marker = " *" if epoch == best_epoch else ""
        print(f"  Epoch {epoch:3d}/{EPOCHS} | Train: {train_loss:.4f} (acc={train_acc:.4f}) | "
              f"Val: {val_loss:.4f} (acc={val_acc:.4f}) | Time: {elapsed:.1f}s | Pat: {patience_counter}/{PATIENCE}{marker}")

        if patience_counter >= PATIENCE:
            print(f"\nEarly stopping at epoch {epoch}. Best: {best_epoch}")
            break

    # Save curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epochs_range = range(1, len(train_losses) + 1)
    ax1.plot(epochs_range, train_losses, 'b-', label='Train')
    ax1.plot(epochs_range, val_losses, 'r-', label='Val')
    ax1.axvline(best_epoch, color='g', linestyle='--', alpha=0.7)
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss'); ax1.legend(); ax1.grid(True, alpha=0.3)
    ax1.set_title('TFT v3 — Loss')
    ax2.plot(epochs_range, train_accs, 'b-', label='Train')
    ax2.plot(epochs_range, val_accs, 'r-', label='Val')
    ax2.axvline(best_epoch, color='g', linestyle='--', alpha=0.7)
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Accuracy'); ax2.legend(); ax2.grid(True, alpha=0.3)
    ax2.set_title('TFT v3 — Accuracy')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_curves.png"), dpi=150)
    plt.close()

    history = {
        'train_losses': train_losses, 'val_losses': val_losses,
        'train_accs': train_accs, 'val_accs': val_accs,
        'best_epoch': best_epoch, 'best_val_loss': best_val_loss,
    }
    with open(os.path.join(output_dir, "training_history.json"), 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\nBest epoch: {best_epoch}, Val loss: {best_val_loss:.6f}")
    print("Done!")


if __name__ == "__main__":
    main()
