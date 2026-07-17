#!/usr/bin/env python3
"""
model.py - LSTM models for MIMIC-III waveform forecasting (all 3 signals).

Trains 3 separate LSTM models, one per target signal (ABP, PLETH, II).
Each model uses all 3 signals as input (input_size=3) to predict the next
25 time steps of its target signal.

Best checkpoint per model is saved based on minimum validation loss.

Usage:
    python model.py    # Train all 3 models
"""

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from preprocess import create_dataloaders, FORECAST_HORIZON, INPUT_LENGTH, \
    NUM_SIGNALS, SIGNAL_NAMES


# Configuration
HIDDEN_SIZE = 64
NUM_LAYERS = 2
INPUT_SIZE = NUM_SIGNALS  # 3 signals as input
NUM_EPOCHS = 20
LEARNING_RATE = 0.001
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(BASE_DIR, 'checkpoints')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')


def get_checkpoint_dir(num_epochs=NUM_EPOCHS):
    """Get epoch-specific checkpoint directory."""
    return os.path.join(BASE_DIR, 'checkpoints', f'epochs_{num_epochs}')


def get_output_dir(num_epochs=NUM_EPOCHS):
    """Get epoch-specific output directory."""
    return os.path.join(BASE_DIR, 'outputs', f'epochs_{num_epochs}')


class LSTMForecaster(nn.Module):
    """
    LSTM-based model for time-series forecasting.

    Architecture:
        - 2-layer LSTM with hidden_size=64, input_size=3 (all signals)
        - Fully connected output layer predicting forecast_horizon steps
          of a single target signal
    """

    def __init__(self, input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE,
                 num_layers=NUM_LAYERS, forecast_horizon=FORECAST_HORIZON):
        super(LSTMForecaster, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1,
        )

        self.fc = nn.Linear(hidden_size, forecast_horizon)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Tensor of shape (batch, seq_len, input_size=3)

        Returns:
            Tensor of shape (batch, forecast_horizon)
        """
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_size)
        last_output = lstm_out[:, -1, :]  # (batch, hidden_size)
        prediction = self.fc(last_output)  # (batch, forecast_horizon)
        return prediction


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """Train model for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        predictions = model(x_batch)
        loss = criterion(predictions, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def evaluate(model, data_loader, criterion, device):
    """Evaluate model on a data loader."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            predictions = model(x_batch)
            loss = criterion(predictions, y_batch)

            total_loss += loss.item()
            num_batches += 1

    return total_loss / max(num_batches, 1)


def train_single_model(target_idx, device, num_epochs=NUM_EPOCHS):
    """
    Train a single LSTM model for a target signal.

    Args:
        target_idx: Index of target signal (0=ABP, 1=PLETH, 2=II).
        device: torch device.
        num_epochs: Number of training epochs.

    Returns:
        Tuple of (train_losses, val_losses, best_val_loss).
    """
    signal_name = SIGNAL_NAMES[target_idx]
    checkpoint_dir = get_checkpoint_dir(num_epochs)
    print(f"\n{'=' * 60}")
    print(f"Training model for: {signal_name} (target_idx={target_idx})")
    print(f"Input: all 3 signals ({', '.join(SIGNAL_NAMES)})")
    print(f"{'=' * 60}")

    # Load data for this target
    print(f"\n[INFO] Loading data (target: {signal_name})...")
    train_loader, val_loader, test_loader, norm_params = create_dataloaders(target_idx)

    # Create model
    model = LSTMForecaster().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Model: input_size={INPUT_SIZE}, hidden={HIDDEN_SIZE}, "
          f"layers={NUM_LAYERS}, output={FORECAST_HORIZON}, params={total_params:,}")

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    print(f"\n[INFO] Training for {num_epochs} epochs...")
    print("-" * 50)
    print(f"{'Epoch':>6} | {'Train Loss':>12} | {'Val Loss':>12} | {'Status':>10}")
    print("-" * 50)

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Save checkpoint at minimum validation loss
        status = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(
                checkpoint_dir, f'best_model_{signal_name.lower()}.pt'
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'norm_params': norm_params,
                'target_idx': target_idx,
                'signal_name': signal_name,
                'num_epochs': num_epochs,
            }, checkpoint_path)
            status = "* best *"

        print(f"{epoch:>6} | {train_loss:>12.6f} | {val_loss:>12.6f} | {status:>10}")

    print("-" * 50)
    print(f"[INFO] Best val loss for {signal_name}: {best_val_loss:.6f}")
    print(f"[SAVED] {os.path.join(checkpoint_dir, f'best_model_{signal_name.lower()}.pt')}")

    return train_losses, val_losses, best_val_loss


def plot_training_curves(all_train_losses, all_val_losses, output_dir):
    """Save training curves for all 3 models in one figure."""
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = ['tab:blue', 'tab:green', 'tab:red']

    for i, (ax, signal_name) in enumerate(zip(axes, SIGNAL_NAMES)):
        epochs = range(1, len(all_train_losses[i]) + 1)
        ax.plot(epochs, all_train_losses[i], f'-o', color=colors[i],
                markersize=3, label='Train', alpha=0.8)
        ax.plot(epochs, all_val_losses[i], f'--s', color=colors[i],
                markersize=3, label='Val', alpha=0.8)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MSE Loss')
        ax.set_title(f'{signal_name} Model')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle('Training Curves — All Signals', fontsize=13, fontweight='bold')
    plt.tight_layout()

    filepath = os.path.join(output_dir, 'training_curves.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] Training curves: {filepath}")


def main(num_epochs=None):
    """Train all 3 LSTM models (one per signal).
    
    Args:
        num_epochs: Override default NUM_EPOCHS if provided.
    """
    epochs = num_epochs if num_epochs is not None else NUM_EPOCHS
    checkpoint_dir = get_checkpoint_dir(epochs)
    output_dir = get_output_dir(epochs)

    print("=" * 60)
    print("MIMIC-III Waveform Forecasting — Multi-Signal LSTM Training")
    print("=" * 60)
    print(f"Signals: {SIGNAL_NAMES}")
    print(f"Architecture: 3 separate LSTMs, input_size={INPUT_SIZE}")
    print(f"Each model uses all 3 signals as input, predicts 1 target signal")
    print(f"Epochs: {epochs}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Device: {device}")

    all_train_losses = []
    all_val_losses = []
    all_best_val = []

    for target_idx in range(NUM_SIGNALS):
        train_losses, val_losses, best_val = train_single_model(target_idx, device, epochs)
        all_train_losses.append(train_losses)
        all_val_losses.append(val_losses)
        all_best_val.append(best_val)

    # Plot all training curves
    plot_training_curves(all_train_losses, all_val_losses, output_dir)

    # Summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE — All Models")
    print("=" * 60)
    for i, name in enumerate(SIGNAL_NAMES):
        print(f"  {name:>5}: best_val_loss = {all_best_val[i]:.6f} "
              f"| final_train = {all_train_losses[i][-1]:.6f}")
    print(f"\n  Checkpoints: {checkpoint_dir}/best_model_{{signal}}.pt")
    print(f"  Training plot: {output_dir}/training_curves.png")
    print("=" * 60)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train LSTM models for waveform forecasting')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS,
                        help=f'Number of training epochs (default: {NUM_EPOCHS})')
    args = parser.parse_args()
    main(num_epochs=args.epochs)
