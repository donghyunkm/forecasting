#!/usr/bin/env python3
"""
model.py - LSTM model for multivariate waveform forecasting (Phase 3).

Trains a separate LSTM model for each target signal. Each model takes 75 time
points of aggregated features from all 4 signals (24 features total) and predicts
the next 25 time points of the target signal's mean value.

Architecture:
    - 2-layer Bidirectional LSTM (hidden=128)
    - Fully connected decoder: project final hidden → 25-step forecast

Best checkpoint is saved based on minimum validation loss.

Usage:
    python model.py --target II         # Train II forecaster
    python model.py --target PLETH      # Train PLETH forecaster
    python model.py --target RESP --epochs 50
    python model.py --target ABP
"""

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from preprocess import (create_dataloaders, INPUT_LENGTH, OUTPUT_LENGTH,
                        NUM_SIGNALS, NUM_FEATURES, SIGNAL_NAMES, VALID_TARGETS,
                        INTERVAL_MINUTES, FEATURE_NAMES)


# Configuration
HIDDEN_SIZE = 64
NUM_LAYERS = 2
INPUT_SIZE = NUM_SIGNALS * NUM_FEATURES  # 4 signals × 6 features = 24
NUM_EPOCHS = 100
LEARNING_RATE = 0.001
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def get_checkpoint_dir(target_signal, num_epochs=NUM_EPOCHS):
    """Get checkpoint directory for a specific target signal."""
    return os.path.join(BASE_DIR, 'checkpoints', f'{target_signal}_epochs_{num_epochs}')


def get_output_dir(target_signal, num_epochs=NUM_EPOCHS):
    """Get output directory for a specific target signal."""
    return os.path.join(BASE_DIR, 'outputs', f'{target_signal}_epochs_{num_epochs}')


class LSTMForecaster(nn.Module):
    """
    Bidirectional LSTM for multivariate waveform forecasting.

    Architecture:
        - Input: (batch, 75, 24) — 75 time steps, 24 features per step
        - 2-layer Bidirectional LSTM, hidden=128
        - FC decoder: hidden*2 → 128 → 64 → 150 (25 steps × 6 features)

    The model processes the full 75-step input sequence and outputs a 25-step
    forecast of all 6 aggregated features for the target signal.
    """

    def __init__(self, input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE,
                 num_layers=NUM_LAYERS, output_length=OUTPUT_LENGTH,
                 num_features=NUM_FEATURES, dropout=0.2):
        super(LSTMForecaster, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_length = output_length
        self.num_features = num_features
        self.output_size = output_length * num_features  # 25 × 6 = 150

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )

        # Attention mechanism to weight temporal positions
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

        # Decoder: attention-weighted context → forecast (25 × 6 = 150 outputs)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, self.output_size),
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Tensor of shape (batch, input_length=75, input_size=24)

        Returns:
            Tensor of shape (batch, output_length=25, num_features=6)
        """
        batch_size = x.shape[0]

        # Project input features
        x = self.input_proj(x)  # (batch, 75, hidden_size)

        # LSTM encoding
        lstm_out, _ = self.lstm(x)  # (batch, 75, hidden_size*2)

        # Attention over time steps
        attn_weights = self.attention(lstm_out)  # (batch, 75, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)
        context = torch.sum(lstm_out * attn_weights, dim=1)  # (batch, hidden_size*2)

        # Decode to forecast
        output = self.decoder(context)  # (batch, 150)
        output = output.view(batch_size, self.output_length, self.num_features)  # (batch, 25, 6)

        return output


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
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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


def train_model(device, target_signal='II', num_epochs=NUM_EPOCHS):
    """
    Train the LSTM forecasting model for a specific target signal.

    Args:
        device: torch device.
        target_signal: Signal to forecast ('II', 'PLETH', 'RESP', 'ABP').
        num_epochs: Number of training epochs.

    Returns:
        Tuple of (train_losses, val_losses, best_val_loss).
    """
    checkpoint_dir = get_checkpoint_dir(target_signal, num_epochs)
    print(f"\n{'=' * 60}")
    print(f"Training LSTM Forecaster — Target: {target_signal}")
    print(f"Input: {INPUT_LENGTH} intervals × {INPUT_SIZE} features "
          f"({INPUT_LENGTH * INTERVAL_MINUTES / 60:.1f} hours)")
    print(f"Output: {OUTPUT_LENGTH} intervals of {target_signal} mean "
          f"({OUTPUT_LENGTH * INTERVAL_MINUTES / 60:.1f} hours)")
    print(f"{'=' * 60}")

    # Load data
    print(f"\n[INFO] Loading data for target: {target_signal}...")
    train_loader, val_loader, test_loader, norm_params = create_dataloaders(
        target_signal=target_signal
    )

    # Create model
    model = LSTMForecaster().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Model: input={INPUT_SIZE}, hidden={HIDDEN_SIZE}, "
          f"layers={NUM_LAYERS}, output={OUTPUT_LENGTH}, params={total_params:,}")

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    print(f"\n[INFO] Training for {num_epochs} epochs...")
    print("-" * 55)
    print(f"{'Epoch':>6} | {'Train Loss':>12} | {'Val Loss':>12} | {'LR':>10} | {'Status':>8}")
    print("-" * 55)

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]['lr']
        status = ""

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'norm_params': norm_params,
                'num_epochs': num_epochs,
                'target_signal': target_signal,
            }, checkpoint_path)
            status = "* best *"

        if epoch <= 5 or epoch % 5 == 0 or status:
            print(f"{epoch:>6} | {train_loss:>12.6f} | {val_loss:>12.6f} | "
                  f"{current_lr:>10.6f} | {status:>8}")

    print("-" * 55)
    print(f"[INFO] Best val loss: {best_val_loss:.6f}")
    print(f"[SAVED] {os.path.join(checkpoint_dir, 'best_model.pt')}")

    return train_losses, val_losses, best_val_loss


def plot_training_curves(train_losses, val_losses, target_signal, output_dir):
    """Save training curves."""
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, '-o', color='tab:blue',
            markersize=2, label='Train', alpha=0.8)
    ax.plot(epochs, val_losses, '--s', color='tab:red',
            markersize=2, label='Val', alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss (normalized)')
    ax.set_title(f'Phase 3 LSTM — {target_signal} Forecasting — Training Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    filepath = os.path.join(output_dir, 'training_curves.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] Training curves: {filepath}")


def main(target_signal=None, num_epochs=None):
    """Train the LSTM forecasting model.

    Args:
        target_signal: Signal to forecast.
        num_epochs: Override default epochs.
    """
    target = target_signal if target_signal is not None else 'II'
    epochs = num_epochs if num_epochs is not None else NUM_EPOCHS
    output_dir = get_output_dir(target, epochs)

    print("=" * 60)
    print(f"Phase 3 — Multivariate Waveform Forecasting (LSTM)")
    print("=" * 60)
    print(f"Target signal: {target}")
    print(f"Input: {INPUT_LENGTH} intervals ({INPUT_LENGTH * INTERVAL_MINUTES / 60:.1f} hrs) "
          f"of all {SIGNAL_NAMES}")
    print(f"Output: {OUTPUT_LENGTH} intervals ({OUTPUT_LENGTH * INTERVAL_MINUTES / 60:.1f} hrs) "
          f"of {target} mean")
    print(f"Epochs: {epochs}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Device: {device}")

    train_losses, val_losses, best_val = train_model(device, target, epochs)

    # Plot training curves
    plot_training_curves(train_losses, val_losses, target, output_dir)

    # Summary
    checkpoint_dir = get_checkpoint_dir(target, epochs)
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Target signal:  {target}")
    print(f"  Best val loss:  {best_val:.6f}")
    print(f"  Final train:    {train_losses[-1]:.6f}")
    print(f"  Checkpoint:     {checkpoint_dir}/best_model.pt")
    print(f"  Training plot:  {output_dir}/training_curves.png")
    print("=" * 60)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train LSTM forecaster for waveform prediction')
    parser.add_argument('--target', type=str, default='II', choices=VALID_TARGETS,
                        help=f'Target signal to forecast (default: II)')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS,
                        help=f'Number of training epochs (default: {NUM_EPOCHS})')
    args = parser.parse_args()
    main(target_signal=args.target, num_epochs=args.epochs)
