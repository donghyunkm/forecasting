"""
Training script for Phase 6.2 iTransformer — Cluster Label Forecasting.

Loss: Weighted cross-entropy
Optimizer: Adam with cosine annealing
"""

import os
import sys
import time
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model import iTransformer, build_model
from preprocess import create_dataloaders, NUM_CLASSES


def train_epoch(model, loader, optimizer, criterion, device, grad_clip=1.0):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    n_batches = 0

    for batch in loader:
        historical = batch['historical'].to(device)  # (B, 48, 8)
        target = batch['target'].to(device)          # (B, 12)

        optimizer.zero_grad()
        logits = model(historical)  # (B, 12, 7)

        B, T, C = logits.shape
        loss = criterion(logits.reshape(B * T, C), target.reshape(B * T))
        loss.backward()

        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        preds = logits.argmax(dim=-1)
        total_correct += (preds == target).sum().item()
        total_samples += B * T
        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1), total_correct / max(total_samples, 1)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    n_batches = 0

    for batch in loader:
        historical = batch['historical'].to(device)
        target = batch['target'].to(device)

        logits = model(historical)
        B, T, C = logits.shape
        loss = criterion(logits.reshape(B * T, C), target.reshape(B * T))

        preds = logits.argmax(dim=-1)
        total_correct += (preds == target).sum().item()
        total_samples += B * T
        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1), total_correct / max(total_samples, 1)


def save_training_curves(train_losses, val_losses, train_accs, val_accs, best_epoch, output_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epochs = range(1, len(train_losses) + 1)

    ax1.plot(epochs, train_losses, 'b-', label='Train', linewidth=1.5)
    ax1.plot(epochs, val_losses, 'r-', label='Val', linewidth=1.5)
    ax1.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.7)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Cross-Entropy Loss')
    ax1.set_title('iTransformer (Phase 6.2) — Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, train_accs, 'b-', label='Train', linewidth=1.5)
    ax2.plot(epochs, val_accs, 'r-', label='Val', linewidth=1.5)
    ax2.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.7)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('iTransformer (Phase 6.2) — Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_curves.png"), dpi=150, bbox_inches='tight')
    plt.close()


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    output_dir = f"outputs/itransformer_epochs_{args.epochs}"
    os.makedirs(output_dir, exist_ok=True)

    print("\n--- Loading Data ---")
    train_loader, val_loader, test_loader, norm_params = create_dataloaders(
        batch_size=args.batch_size, num_workers=4)

    # Class weights
    class_weights = torch.tensor(norm_params['class_weights'], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    print(f"  Class weights: {norm_params['class_weights']}")

    torch.save(norm_params, os.path.join(output_dir, "norm_params.pt"))

    print("\n--- Building Model ---")
    model = build_model(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    print(f"\n--- Training for {args.epochs} epochs ---")
    print(f"    Task: Classification (7 clusters)")
    print(f"    Loss: Weighted cross-entropy")

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_loss = float('inf')
    patience_counter = 0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, args.grad_clip)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]['lr']

        print(f"  Epoch {epoch:3d}/{args.epochs} | "
              f"Train: {train_loss:.4f} (acc={train_acc:.4f}) | "
              f"Val: {val_loss:.4f} (acc={val_acc:.4f}) | "
              f"LR: {lr_now:.2e} | Time: {elapsed:.1f}s", end="")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pt"))
            print(" ★", end="")
        else:
            patience_counter += 1

        print()

        if patience_counter >= args.patience:
            print(f"\n  Early stopping at epoch {epoch} (best: epoch {best_epoch})")
            break

    print(f"\n--- Training Complete ---")
    print(f"  Best validation loss: {best_val_loss:.5f} (epoch {best_epoch})")

    save_training_curves(train_losses, val_losses, train_accs, val_accs, best_epoch, output_dir)

    metadata = {
        'best_epoch': best_epoch, 'best_val_loss': best_val_loss,
        'total_epochs': len(train_losses),
        'train_losses': train_losses, 'val_losses': val_losses,
        'train_accs': train_accs, 'val_accs': val_accs,
        'config': {'epochs': args.epochs, 'batch_size': args.batch_size,
                   'lr': args.lr, 'grad_clip': args.grad_clip,
                   'patience': args.patience, 'loss': 'weighted cross-entropy'},
    }
    with open(os.path.join(output_dir, "training_metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--patience', type=int, default=20)
    args = parser.parse_args()
    train(args)
