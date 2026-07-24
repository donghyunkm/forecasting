#!/usr/bin/env python3
"""Phase 6.2 v3 TFT Testing: Evaluate cluster classification with label history + X_stats."""

import os, sys, argparse, json
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix
from omegaconf import OmegaConf
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from model import TemporalFusionTransformer
from preprocess import create_dataloaders

NUM_CLASSES = 7


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_dir = os.path.join(base_dir, f"checkpoints/tft_v3_epochs_{args.epochs}")
    output_dir = os.path.join(base_dir, f"outputs/tft_v3_epochs_{args.epochs}")
    os.makedirs(output_dir, exist_ok=True)

    checkpoint_path = os.path.join(checkpoint_dir, "best_model.pt")
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print("Loading data...")
    _, _, test_loader, norm_params = create_dataloaders(batch_size=64)

    print("Loading model...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = OmegaConf.create(checkpoint['config'])
    model = TemporalFusionTransformer(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"  Loaded from epoch {checkpoint['epoch']} (val_loss={checkpoint['val_loss']:.4f}, val_acc={checkpoint['val_acc']:.4f})")

    print("Running inference...")
    all_logits, all_targets = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch_device = {k: v.to(device) for k, v in batch.items()}
            output = model(batch_device)
            all_logits.append(output['logits'].cpu().numpy())
            all_targets.append(batch_device['target'].cpu().numpy())

    logits = np.concatenate(all_logits, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    predictions = logits.argmax(axis=-1)

    pred_flat = predictions.flatten()
    tgt_flat = targets.flatten()

    accuracy = accuracy_score(tgt_flat, pred_flat)
    f1_macro = f1_score(tgt_flat, pred_flat, average='macro')
    f1_weighted = f1_score(tgt_flat, pred_flat, average='weighted')
    precision, recall, f1, support = precision_recall_fscore_support(
        tgt_flat, pred_flat, labels=list(range(NUM_CLASSES)), zero_division=0)
    cm = confusion_matrix(tgt_flat, pred_flat, labels=list(range(NUM_CLASSES)))

    horizon_accs = [accuracy_score(targets[:, t], predictions[:, t]) for t in range(targets.shape[1])]

    print(f"\n{'='*70}")
    print(f"TEST RESULTS — Phase 6.2 v3 TFT (with label history + X_stats)")
    print(f"{'='*70}")
    print(f"  Accuracy:     {accuracy:.4f}")
    print(f"  Macro F1:     {f1_macro:.4f}")
    print(f"  Weighted F1:  {f1_weighted:.4f}")
    print(f"\n  {'Class':<12} {'Prec':<8} {'Recall':<8} {'F1':<8} {'Support':<10}")
    print(f"  {'-'*46}")
    for c in range(NUM_CLASSES):
        print(f"  Cluster {c:<3} {precision[c]:<8.4f} {recall[c]:<8.4f} {f1[c]:<8.4f} {support[c]:<10,}")
    print(f"\n  Per-horizon accuracy:")
    for t, acc in enumerate(horizon_accs):
        print(f"    Step {t+1:2d} ({(t+1)*2.5:5.1f} min): {acc:.4f}")
    print(f"{'='*70}")

    # Save metrics
    metrics = {
        'overall': {'accuracy': float(accuracy), 'f1_macro': float(f1_macro), 'f1_weighted': float(f1_weighted)},
        'per_class': {f'cluster_{c}': {'precision': float(precision[c]), 'recall': float(recall[c]),
                      'f1': float(f1[c]), 'support': int(support[c])} for c in range(NUM_CLASSES)},
        'per_horizon_accuracy': horizon_accs,
        'confusion_matrix': cm.tolist(),
    }
    with open(os.path.join(output_dir, "test_metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=2)
    np.save(os.path.join(output_dir, "test_predictions.npy"), predictions)
    np.save(os.path.join(output_dir, "test_targets.npy"), targets)

    # Plots
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', ax=ax,
                xticklabels=[f'P{i}' for i in range(NUM_CLASSES)],
                yticklabels=[f'T{i}' for i in range(NUM_CLASSES)])
    ax.set_title('TFT v3 — Confusion Matrix'); plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=150); plt.close()

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(np.arange(len(horizon_accs)) * 2.5, horizon_accs, 'o-', color='steelblue', linewidth=2)
    ax.axhline(accuracy, color='red', linestyle='--', label=f'Mean: {accuracy:.4f}')
    ax.axhline(1/NUM_CLASSES, color='gray', linestyle=':', label='Random')
    ax.set_xlabel('Horizon (min)'); ax.set_ylabel('Accuracy')
    ax.set_title('TFT v3 — Accuracy by Horizon'); ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(output_dir, "accuracy_by_horizon.png"), dpi=150); plt.close()

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.bar(range(NUM_CLASSES), f1, color='steelblue', alpha=0.8)
    ax.axhline(f1_macro, color='red', linestyle='--', label=f'Macro F1: {f1_macro:.4f}')
    ax.set_xticks(range(NUM_CLASSES)); ax.set_xticklabels([f'C{i}' for i in range(NUM_CLASSES)])
    ax.set_ylabel('F1'); ax.set_title('TFT v3 — Per-Class F1'); ax.legend(); ax.grid(True, alpha=0.2, axis='y')
    plt.tight_layout(); plt.savefig(os.path.join(output_dir, "per_class_f1.png"), dpi=150); plt.close()

    print("Plots saved. Done!")


if __name__ == "__main__":
    main()
