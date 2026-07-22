"""
Training script for Phase 5 iTransformer vital sign forecasting.

Config: epochs=100, batch_size=64, lr=1e-4, grad_clip=1.0, patience=20
Loss: Quantile (pinball) loss with quantiles [0.1, 0.5, 0.9]
Optimizer: Adam with early stopping
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
from preprocess import create_dataloaders, VITAL_NAMES


# ─── Loss Function ───────────────────────────────────────────────────────────

QUANTILES = [0.1, 0.5, 0.9]


def quantile_loss(predictions, targets, target_mask, quantiles=QUANTILES):
    """
    Pinball (quantile) loss.

    Args:
        predictions: (B, pred_len, n_output_vars, n_quantiles) = (B, 24, 4, 3)
        targets: (B, pred_len, n_output_vars) = (B, 24, 4)
        target_mask: (B, pred_len, n_output_vars) = (B, 24, 4)
        quantiles: list of quantile levels

    Returns:
        Scalar loss
    """
    quantiles_t = torch.tensor(quantiles, device=predictions.device, dtype=predictions.dtype)  # (3,)

    # Expand targets for quantile dimension
    targets_expanded = targets.unsqueeze(-1)  # (B, 24, 4, 1)
    mask_expanded = target_mask.unsqueeze(-1)  # (B, 24, 4, 1)

    # Compute errors
    errors = targets_expanded - predictions  # (B, 24, 4, 3)

    # Pinball loss: q * max(error, 0) + (1-q) * max(-error, 0)
    loss = torch.max(quantiles_t * errors, (quantiles_t - 1) * errors)  # (B, 24, 4, 3)

    # Apply mask
    loss = loss * mask_expanded

    # Mean over valid entries
    n_valid = mask_expanded.sum().clamp(min=1.0)
    return loss.sum() / n_valid


# ─── Training ────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, device, grad_clip=1.0):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        historical = batch['historical'].to(device)   # (B, 72, 12)
        target = batch['target'].to(device)           # (B, 24, 4)
        target_mask = batch['target_mask'].to(device) # (B, 24, 4)

        optimizer.zero_grad()

        # Forward
        predictions = model(historical)  # (B, 24, 4, 3)

        # Loss
        loss = quantile_loss(predictions, target, target_mask)

        # Backward
        loss.backward()

        # Gradient clipping
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate(model, loader, device):
    """Validate and return average loss."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        historical = batch['historical'].to(device)
        target = batch['target'].to(device)
        target_mask = batch['target_mask'].to(device)

        predictions = model(historical)
        loss = quantile_loss(predictions, target, target_mask)

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def train(args):
    """Main training loop."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Output directory
    output_dir = f"outputs/itransformer_epochs_{args.epochs}"
    os.makedirs(output_dir, exist_ok=True)

    # Data
    print("\n--- Loading Data ---")
    train_loader, val_loader, test_loader, norm_params = create_dataloaders(
        batch_size=args.batch_size, num_workers=4
    )

    # Save norm_params for test time
    torch.save(norm_params, os.path.join(output_dir, "norm_params.pt"))

    # Model
    print("\n--- Building Model ---")
    model = build_model(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    # Learning rate scheduler (cosine annealing)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    # Training loop
    print(f"\n--- Training for {args.epochs} epochs ---")
    print(f"    Batch size: {args.batch_size}, LR: {args.lr}, Patience: {args.patience}")

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, args.grad_clip)
        train_losses.append(train_loss)

        # Validate
        val_loss = validate(model, val_loader, device)
        val_losses.append(val_loss)

        # Step scheduler
        scheduler.step()

        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]['lr']

        print(f"  Epoch {epoch:3d}/{args.epochs} | "
              f"Train: {train_loss:.5f} | Val: {val_loss:.5f} | "
              f"LR: {lr_now:.2e} | Time: {elapsed:.1f}s", end="")

        # Early stopping
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
    print(f"  Model saved to: {output_dir}/best_model.pt")

    # Save training curves
    save_training_curves(train_losses, val_losses, best_epoch, output_dir)

    # Save training metadata
    metadata = {
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'total_epochs': len(train_losses),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'config': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'grad_clip': args.grad_clip,
            'patience': args.patience,
        }
    }
    with open(os.path.join(output_dir, "training_metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)


def save_training_curves(train_losses, val_losses, best_epoch, output_dir):
    """Save training curves plot."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=1.5)
    ax.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=1.5)
    ax.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.7, label=f'Best (epoch {best_epoch})')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Quantile Loss')
    ax.set_title('iTransformer Training Curves (Phase 5)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "training_curves.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Training curves saved to: {output_dir}/training_curves.png")


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train iTransformer for vital sign forecasting")
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping norm')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    args = parser.parse_args()

    train(args)
