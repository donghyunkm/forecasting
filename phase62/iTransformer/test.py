"""
Test script for Phase 6.2 iTransformer — Cluster Label Forecasting.

Metrics: accuracy, macro/weighted F1, per-class precision/recall/F1, confusion matrix.
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
from sklearn.metrics import (accuracy_score, f1_score, precision_recall_fscore_support,
                             confusion_matrix)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from model import iTransformer, build_model
from preprocess import create_dataloaders, NUM_CLASSES


@torch.no_grad()
def run_inference(model, test_loader, device):
    model.eval()
    all_logits = []
    all_targets = []

    for batch in test_loader:
        historical = batch['historical'].to(device)
        target = batch['target']

        logits = model(historical)  # (B, 12, 7)
        all_logits.append(logits.cpu().numpy())
        all_targets.append(target.numpy())

    return np.concatenate(all_logits, axis=0), np.concatenate(all_targets, axis=0)


def compute_metrics(logits, targets):
    predictions = logits.argmax(axis=-1)
    pred_flat = predictions.flatten()
    tgt_flat = targets.flatten()

    accuracy = accuracy_score(tgt_flat, pred_flat)
    f1_macro = f1_score(tgt_flat, pred_flat, average='macro')
    f1_weighted = f1_score(tgt_flat, pred_flat, average='weighted')

    precision, recall, f1, support = precision_recall_fscore_support(
        tgt_flat, pred_flat, labels=list(range(NUM_CLASSES)), zero_division=0)

    cm = confusion_matrix(tgt_flat, pred_flat, labels=list(range(NUM_CLASSES)))

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
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', ax=ax,
                xticklabels=[f'Pred {i}' for i in range(NUM_CLASSES)],
                yticklabels=[f'True {i}' for i in range(NUM_CLASSES)])
    ax.set_xlabel('Predicted Cluster')
    ax.set_ylabel('True Cluster')
    ax.set_title('iTransformer (Phase 6.2) — Normalized Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: confusion_matrix.png")


def plot_accuracy_by_horizon(horizon_accs, output_dir):
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    time_steps = np.arange(len(horizon_accs)) * 2.5

    ax.plot(time_steps, horizon_accs, 'o-', color='steelblue', markersize=6, linewidth=2)
    ax.axhline(np.mean(horizon_accs), color='red', linestyle='--', alpha=0.7,
               label=f'Mean: {np.mean(horizon_accs):.4f}')
    ax.axhline(1.0 / NUM_CLASSES, color='gray', linestyle=':', alpha=0.5,
               label=f'Random: {1.0/NUM_CLASSES:.4f}')

    ax.set_xlabel('Forecast Horizon (minutes)')
    ax.set_ylabel('Accuracy')
    ax.set_title('iTransformer (Phase 6.2) — Accuracy by Forecast Horizon')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "accuracy_by_horizon.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: accuracy_by_horizon.png")


def plot_per_class_metrics(metrics, output_dir):
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    classes = [f'Cluster {i}' for i in range(NUM_CLASSES)]
    f1_scores = [metrics['per_class'][f'cluster_{i}']['f1'] for i in range(NUM_CLASSES)]
    supports = [metrics['per_class'][f'cluster_{i}']['support'] for i in range(NUM_CLASSES)]

    x = np.arange(NUM_CLASSES)
    bars = ax.bar(x, f1_scores, color='steelblue', alpha=0.8)
    for i, (bar, sup) in enumerate(zip(bars, supports)):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'n={sup:,}', ha='center', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylabel('F1 Score')
    ax.set_title('iTransformer (Phase 6.2) — Per-Class F1 Score')
    ax.set_ylim(0, 1)
    ax.axhline(metrics['overall']['f1_macro'], color='red', linestyle='--',
               label=f'Macro F1: {metrics["overall"]["f1_macro"]:.4f}')
    ax.legend()
    ax.grid(True, alpha=0.2, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "per_class_f1.png"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: per_class_f1.png")


def test(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    output_dir = f"outputs/itransformer_epochs_{args.epochs}"
    model_path = os.path.join(output_dir, "best_model.pt")

    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        sys.exit(1)

    print("\n--- Loading Data ---")
    _, _, test_loader, norm_params = create_dataloaders(batch_size=64, num_workers=4)

    print("\n--- Loading Model ---")
    model = build_model(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    print("\n--- Running Inference ---")
    logits, targets = run_inference(model, test_loader, device)
    print(f"  Logits shape: {logits.shape}")
    print(f"  Targets shape: {targets.shape}")

    print("\n--- Computing Metrics ---")
    metrics, predictions = compute_metrics(logits, targets)

    print(f"\n{'='*70}")
    print(f"  iTransformer Test Results (Phase 6.2 — Cluster Classification)")
    print(f"{'='*70}")
    print(f"  Overall accuracy:  {metrics['overall']['accuracy']:.4f}")
    print(f"  Macro F1:          {metrics['overall']['f1_macro']:.4f}")
    print(f"  Weighted F1:       {metrics['overall']['f1_weighted']:.4f}")
    print(f"\n  {'Class':<12} {'Precision':<11} {'Recall':<9} {'F1':<8} {'Support':<10}")
    print(f"  {'-'*50}")
    for c in range(NUM_CLASSES):
        m = metrics['per_class'][f'cluster_{c}']
        print(f"  Cluster {c:<3} {m['precision']:<11.4f} {m['recall']:<9.4f} "
              f"{m['f1']:<8.4f} {m['support']:<10,}")
    print(f"{'='*70}")

    print(f"\n--- Saving Results ---")
    np.save(os.path.join(output_dir, "test_predictions.npy"), predictions)
    np.save(os.path.join(output_dir, "test_targets.npy"), targets)
    np.save(os.path.join(output_dir, "test_logits.npy"), logits)
    with open(os.path.join(output_dir, "test_metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\n--- Generating Plots ---")
    cm = np.array(metrics['confusion_matrix'])
    plot_confusion_matrix(cm, output_dir)
    plot_accuracy_by_horizon(metrics['per_horizon_accuracy'], output_dir)
    plot_per_class_metrics(metrics, output_dir)
    print(f"\n  Output dir: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()
    test(args)
