"""
Training for Phase 6.2 v3 iTransformer — Cluster classification with label history + X_stats.
Input: (B, 48, 47) = 7 corr + 38 physio + 1 time + 1 label history
"""

import os, sys, time, argparse, json
import numpy as np
import torch
import torch.nn as nn
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model import iTransformer, build_model
from preprocess import create_dataloaders, NUM_CLASSES


def train_epoch(model, loader, optimizer, criterion, device, grad_clip=1.0):
    model.train()
    total_loss, total_correct, total_samples, n_batches = 0., 0, 0, 0

    for batch in loader:
        historical = batch['historical'].to(device)
        target = batch['target'].to(device)

        optimizer.zero_grad()
        logits = model(historical)
        B, T, C = logits.shape
        loss = criterion(logits.reshape(B*T, C), target.reshape(B*T))
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_correct += (logits.argmax(-1) == target).sum().item()
        total_samples += B * T
        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1), total_correct / max(total_samples, 1)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss, total_correct, total_samples, n_batches = 0., 0, 0, 0

    for batch in loader:
        historical = batch['historical'].to(device)
        target = batch['target'].to(device)
        logits = model(historical)
        B, T, C = logits.shape
        loss = criterion(logits.reshape(B*T, C), target.reshape(B*T))
        total_correct += (logits.argmax(-1) == target).sum().item()
        total_samples += B * T
        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1), total_correct / max(total_samples, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--grad_clip', type=float, default=1.0)
    parser.add_argument('--patience', type=int, default=20)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    output_dir = f"outputs/itransformer_v3_epochs_{args.epochs}"
    os.makedirs(output_dir, exist_ok=True)

    print("\n--- Loading Data (v3 — with label history + X_stats) ---")
    train_loader, val_loader, _, norm_params = create_dataloaders(batch_size=args.batch_size)

    class_weights = torch.tensor(norm_params['class_weights'], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    torch.save(norm_params, os.path.join(output_dir, "norm_params.pt"))

    print("\n--- Building Model ---")
    model = build_model(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr*0.01)

    print(f"\n--- Training (patience={args.patience}) ---")
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    best_val_loss = float('inf')
    patience_counter, best_epoch = 0, 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, args.grad_clip)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        train_losses.append(train_loss); val_losses.append(val_loss)
        train_accs.append(train_acc); val_accs.append(val_acc)

        elapsed = time.time() - t0
        print(f"  Epoch {epoch:3d}/{args.epochs} | Train: {train_loss:.4f} (acc={train_acc:.4f}) | "
              f"Val: {val_loss:.4f} (acc={val_acc:.4f}) | Time: {elapsed:.1f}s", end="")

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
            print(f"\n  Early stopping (best: epoch {best_epoch})")
            break

    print(f"\nBest val loss: {best_val_loss:.5f} (epoch {best_epoch})")

    # Save curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ep = range(1, len(train_losses)+1)
    ax1.plot(ep, train_losses, 'b-', label='Train'); ax1.plot(ep, val_losses, 'r-', label='Val')
    ax1.axvline(best_epoch, color='g', linestyle='--'); ax1.legend(); ax1.grid(True, alpha=0.3)
    ax1.set_title('iTransformer v3 — Loss')
    ax2.plot(ep, train_accs, 'b-', label='Train'); ax2.plot(ep, val_accs, 'r-', label='Val')
    ax2.axvline(best_epoch, color='g', linestyle='--'); ax2.legend(); ax2.grid(True, alpha=0.3)
    ax2.set_title('iTransformer v3 — Accuracy')
    plt.tight_layout(); plt.savefig(os.path.join(output_dir, "training_curves.png"), dpi=150); plt.close()

    json.dump({'best_epoch': best_epoch, 'best_val_loss': best_val_loss,
               'train_losses': train_losses, 'val_losses': val_losses,
               'train_accs': train_accs, 'val_accs': val_accs},
              open(os.path.join(output_dir, "training_metadata.json"), 'w'), indent=2)


if __name__ == "__main__":
    main()
