#!/usr/bin/env python3
"""
Phase 6.2 TFT Testing: Evaluate cluster label classification.

Metrics: accuracy, macro/weighted F1, per-class precision/recall/F1, confusion matrix.
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
from sklearn.metrics import (accuracy_score, f1_score, precision_recall_fscore_support,
                             confusion_matrix, classification_report)
from omegaconf import OmegaConf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from model import TemporalFusionTransformer
from preprocess import create_dataloaders

NUM_CLASSES = 7


def run_inference(model, test_loader, device):
    """Run model inference on test set."""
    model.eval()
    all_logits = []
    all_targets = []

    with torch.no_grad():
        for batch in test_loader:
            batch_device = {k: v.to(device) for k, v in batch.items()}
            output = model(batch_device)
            logits = output['logits']  # (B, 12, 7)
            targets = batch_device['target']  # (B, 12)

            all_logits.append(logits.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    logits = np.concatenate(all_logits, axis=0)   # (N, 12, 7)
    targets = np.concatenate(all_targets, axis=0)  # (N, 12)

    return logits, targets


def compute_metrics(logits, targets):
    """Compute classification metrics."""
    predictions = logits.argmax(axis=-1)  # (N, 12)

    # Flatten for overall metrics
    pred_flat = predictions.flatten()
    tgt_flat = targets.flatten()

    # Overall accuracy
    accuracy = accuracy_score(tgt_flat, pred_flat)

    # F1 scores
    f1_macro = f1_score(tgt_flat, pred_flat, average='macro')
    f1_weighted = f1_score(tgt_flat, pred_flat, average='weighted')

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        tgt_flat, pred_flat, labels=list(range(NUM_CLASSES)), zero_division=0)

    # Confusion matrix
    cm = confusion_matrix(tgt_flat, pred_flat, labels=list(range(NUM_CLASSES)))

    # Per-horizon accuracy
    horizon_accs = []
    for t in range(targets.shape[1]):
        acc_t = accuracy_score(targets[:, t], predictions[:, t])
        horizon_accs.append(acc_t)

    metrics = {
        'overall': {
            'accuracy': float(accuracy),
            'f1_macro': float(f1_macro),
            'f1_weighted': float(f1_weighted),
        },
        'per_class': {},
        'per_horizon_accuracy': horizon_accs,
        'confusion_matrix': cm.tolist(),
    }

    for c in range(NUM_CLASSES):
        metrics['per_class'][f'cluster_{c}'] = {
            'precision': float(precision[c]),
            'recall': float(recall[c]),
            'f1': float(f1[c]),
            'support': int(support[c]),
        }

    return metrics, predictions


def plot_confusion_matrix(cm, output_dir):
    """Plot confusion matrix heatmap."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))

    # Normalize by row (true label)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', ax=ax,
                xticklabels=[f'Pred {i}' for i in range(NUM_CLASSES)],
                yticklabels=[f'True {i}' for i in range(NUM_CLASSES)])
    ax.set_xlabel('Predicted Cluster')
    ax.set_ylabel('True Cluster')
    ax.set_title('TFT (Phase 6.2) — Normalized Confusion Matrix')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: confusion_matrix.png")


def plot_accuracy_by_horizon(horizon_accs, output_dir):
    """Plot accuracy at each forecast step."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    time_steps = np.arange(len(horizon_accs)) * 2.5  # minutes

    ax.plot(time_steps, horizon_accs, 'o-', color='steelblue', markersize=6, linewidth=2)
    ax.axhline(np.mean(horizon_accs), color='red', linestyle='--', alpha=0.7,
               label=f'Mean: {np.mean(horizon_accs):.4f}')
    ax.axhline(1.0 / NUM_CLASSES, color='gray', linestyle=':', alpha=0.5,
               label=f'Random: {1.0/NUM_CLASSES:.4f}')

    ax.set_xlabel('Forecast Horizon (minutes)')
    ax.set_ylabel('Accuracy')
    ax.set_title('TFT (Phase 6.2) — Accuracy by Forecast Horizon')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "accuracy_by_horizon.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: accuracy_by_horizon.png")


def plot_per_class_metrics(metrics, output_dir):
    """Bar chart of per-class F1 scores."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    classes = [f'Cluster {i}' for i in range(NUM_CLASSES)]
    f1_scores = [metrics['per_class'][f'cluster_{i}']['f1'] for i in range(NUM_CLASSES)]
    supports = [metrics['per_class'][f'cluster_{i}']['support'] for i in range(NUM_CLASSES)]

    x = np.arange(NUM_CLASSES)
    bars = ax.bar(x, f1_scores, color='steelblue', alpha=0.8)

    # Annotate with support counts
    for i, (bar, sup) in enumerate(zip(bars, supports)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'n={sup:,}', ha='center', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylabel('F1 Score')
    ax.set_title('TFT (Phase 6.2) — Per-Class F1 Score')
    ax.set_ylim(0, 1)
    ax.axhline(metrics['overall']['f1_macro'], color='red', linestyle='--',
               label=f'Macro F1: {metrics["overall"]["f1_macro"]:.4f}')
    ax.legend()
    ax.grid(True, alpha=0.2, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "per_class_f1.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: per_class_f1.png")


def main():
    parser = argparse.ArgumentParser(description='Phase 6.2 TFT Testing')
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_dir = os.path.join(base_dir, f"checkpoints/tft_epochs_{args.epochs}")
    output_dir = os.path.join(base_dir, f"outputs/tft_epochs_{args.epochs}")
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
    print(f"  Loaded from epoch {checkpoint['epoch']} (val_loss={checkpoint['val_loss']:.6f})")

    print("Running inference...")
    logits, targets = run_inference(model, test_loader, device)
    print(f"  Logits shape: {logits.shape}")
    print(f"  Targets shape: {targets.shape}")

    print("Computing metrics...")
    metrics, predictions = compute_metrics(logits, targets)

    # Print results
    print(f"\n{'='*70}")
    print(f"TEST RESULTS — Phase 6.2 TFT (Cluster Classification)")
    print(f"{'='*70}")
    print(f"  Overall accuracy:  {metrics['overall']['accuracy']:.4f}")
    print(f"  Macro F1:          {metrics['overall']['f1_macro']:.4f}")
    print(f"  Weighted F1:       {metrics['overall']['f1_weighted']:.4f}")
    print(f"\n  Per-class metrics:")
    print(f"  {'Class':<12} {'Precision':<11} {'Recall':<9} {'F1':<8} {'Support':<10}")
    print(f"  {'-'*50}")
    for c in range(NUM_CLASSES):
        m = metrics['per_class'][f'cluster_{c}']
        print(f"  Cluster {c:<3} {m['precision']:<11.4f} {m['recall']:<9.4f} "
              f"{m['f1']:<8.4f} {m['support']:<10,}")
    print(f"{'='*70}")

    # Save
    print(f"\nSaving results to {output_dir}/...")
    np.save(os.path.join(output_dir, "test_predictions.npy"), predictions)
    np.save(os.path.join(output_dir, "test_targets.npy"), targets)
    np.save(os.path.join(output_dir, "test_logits.npy"), logits)

    with open(os.path.join(output_dir, "test_metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=2)

    # Plots
    print("\nGenerating plots...")
    cm = np.array(metrics['confusion_matrix'])
    plot_confusion_matrix(cm, output_dir)
    plot_accuracy_by_horizon(metrics['per_horizon_accuracy'], output_dir)
    plot_per_class_metrics(metrics, output_dir)

    print("Done!")


if __name__ == "__main__":
    main()
