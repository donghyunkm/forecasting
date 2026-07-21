"""
iTransformer — Vital Sign Forecasting from Waveforms (Phase 4.1)
Training script for the iTransformer model.
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model import iTransformer
from preprocess import create_dataloaders

# ==============================================================================
# Constants
# ==============================================================================
NUM_SIGNALS = 4
FUTURE_MONTHS = 25
SIGNAL_NAMES = ['mean_bp', 'pulse', 'spo2', 'respiratory_rate']
OUTPUT_QUANTILES = [0.1, 0.5, 0.9]


# ==============================================================================
# Loss Function (identical to TFT)
# ==============================================================================
def get_quantiles_loss_and_q_risk(outputs, targets, masks, desired_quantiles):
    """
    Quantile loss with masking.
    
    Args:
        outputs: (batch, 25, 12) -> reshaped to (batch, 25, 4, 3)
        targets: (batch, 25, 4)
        masks: (batch, 25, 4)
        desired_quantiles: tensor of quantile values [0.1, 0.5, 0.9]
    
    Returns:
        q_loss: scalar quantile loss
        q_risk: quantile risk per signal per quantile
        losses: elementwise losses
    """
    outputs = outputs.reshape((outputs.shape[0], FUTURE_MONTHS, NUM_SIGNALS, len(desired_quantiles)))
    errors = targets.unsqueeze(-1) - outputs
    for i in range(masks.shape[-1]):
        for j in range(len(desired_quantiles)):
            errors[..., i, j] = errors[..., i, j] * masks[..., i]
    losses = torch.max((desired_quantiles - 1) * errors, desired_quantiles * errors)
    q_loss = losses.sum(dim=-1).sum(dim=-1).mean(dim=-1).mean()
    q_risk = 2 * (losses.sum(dim=1).sum(dim=0)) / (targets.abs().sum().unsqueeze(-1))
    q_risk = q_risk.sum(dim=0)
    return q_loss, q_risk, losses


# ==============================================================================
# Training & Validation
# ==============================================================================
def train_one_epoch(model, train_loader, optimizer, device, desired_quantiles):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in train_loader:
        historical = batch['historical_ts_numeric'].to(device)
        future = batch['future_ts_numeric'].to(device)
        static = batch['static_feats_numeric'].to(device)
        targets = batch['target'].to(device)
        masks = batch['target_mask'].to(device)

        # Forward pass
        input_dict = {
            'historical_ts_numeric': historical,
            'future_ts_numeric': future,
            'static_feats_numeric': static,
        }
        output = model(input_dict)
        predicted_quantiles = output['predicted_quantiles']  # (batch, 25, 12)

        # Loss
        q_loss, q_risk, _ = get_quantiles_loss_and_q_risk(
            predicted_quantiles, targets, masks, desired_quantiles
        )

        # Backward
        optimizer.zero_grad()
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += q_loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def validate(model, val_loader, device, desired_quantiles):
    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            historical = batch['historical_ts_numeric'].to(device)
            future = batch['future_ts_numeric'].to(device)
            static = batch['static_feats_numeric'].to(device)
            targets = batch['target'].to(device)
            masks = batch['target_mask'].to(device)

            input_dict = {
                'historical_ts_numeric': historical,
                'future_ts_numeric': future,
                'static_feats_numeric': static,
            }
            output = model(input_dict)
            predicted_quantiles = output['predicted_quantiles']

            q_loss, q_risk, _ = get_quantiles_loss_and_q_risk(
                predicted_quantiles, targets, masks, desired_quantiles
            )

            total_loss += q_loss.item()
            n_batches += 1

    return total_loss / max(n_batches, 1)


# ==============================================================================
# Plot Training Curves
# ==============================================================================
def plot_training_curves(train_losses, val_losses, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    ax.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Quantile Loss', fontsize=12)
    ax.set_title('iTransformer — Training Curves (Phase 4.1)', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, len(train_losses))
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'training_curves.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Training curves saved to: {save_path}')


# ==============================================================================
# Main
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description='Train iTransformer for Phase 4.1')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs (default: 100)')
    args = parser.parse_args()

    num_epochs = args.epochs

    print('=' * 80)
    print('iTransformer — Vital Sign Forecasting from Waveforms (Phase 4.1)')
    print('=' * 80)
    print()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'  Device: {device}')
    if device.type == 'cuda':
        print(f'  GPU: {torch.cuda.get_device_name(0)}')
        print(f'  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
    print()

    # Data
    print('Loading pre-processed data...')
    train_loader, val_loader, test_loader, norm_params = create_dataloaders(batch_size=64, num_workers=4)
    print()

    # Model config
    # Note: n_vars=4 means 4 vital signs; the model internally computes
    # n_input_vars = n_vars*2 + 1 = 9 (4 vitals + 4 masks + 1 time)
    model_config = {
        'seq_len': 75,
        'pred_len': 25,
        'n_vars': 4,
        'n_output_vars': 4,
        'n_quantiles': 3,
        'd_model': 256,
        'n_heads': 4,
        'd_ff': 512,
        'n_layers': 3,
        'dropout': 0.1,
    }

    print('Model Configuration:')
    for k, v in model_config.items():
        print(f'  {k}: {v}')
    print()

    # Build model
    model = iTransformer(**model_config).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  Trainable parameters: {n_params:,}')
    print()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    print(f'  Optimizer: Adam, LR=1e-4')
    print(f'  Gradient clipping: max_norm=1.0')
    print(f'  Early stopping patience: 20')
    print(f'  Epochs: {num_epochs}')
    print()

    # Quantiles on device
    desired_quantiles = torch.tensor(OUTPUT_QUANTILES, device=device)

    # Output directories
    checkpoint_dir = os.path.join('checkpoints', f'iTransformer_epochs_{num_epochs}')
    output_dir = os.path.join('outputs', f'iTransformer_epochs_{num_epochs}')
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Training loop
    print('=' * 80)
    print('Training...')
    print('=' * 80)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 20
    best_epoch = 0

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, device, desired_quantiles)
        val_loss = validate(model, val_loader, device, desired_quantiles)

        elapsed = time.time() - t0
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            # Save best model
            save_path = os.path.join(checkpoint_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'model_config': model_config,
            }, save_path)
            marker = ' *'
        else:
            patience_counter += 1
            marker = ''

        print(f'  Epoch {epoch:3d}/{num_epochs} | '
              f'Train: {train_loss:.6f} | Val: {val_loss:.6f} | '
              f'Time: {elapsed:.1f}s | Patience: {patience_counter}/{patience}{marker}')

        # Early stopping
        if patience_counter >= patience:
            print(f'\n  Early stopping at epoch {epoch}. Best epoch: {best_epoch} (val_loss={best_val_loss:.6f})')
            break

    print()
    print('=' * 80)
    print('Training Complete!')
    print('=' * 80)
    print(f'  Best epoch: {best_epoch}')
    print(f'  Best val loss: {best_val_loss:.6f}')
    print(f'  Checkpoint: {os.path.join(checkpoint_dir, "best_model.pt")}')
    print()

    # Plot training curves
    plot_training_curves(train_losses, val_losses, output_dir)

    # Final evaluation on test set
    print('Evaluating on test set...')
    checkpoint = torch.load(os.path.join(checkpoint_dir, 'best_model.pt'), weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    test_loss = validate(model, test_loader, device, desired_quantiles)
    print(f'  Test Loss: {test_loss:.6f}')
    print()
    print('Done.')


if __name__ == '__main__':
    main()
