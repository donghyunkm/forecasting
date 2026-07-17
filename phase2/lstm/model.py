#!/usr/bin/env python3
"""
model.py - LSTM model for heart rate prediction from waveforms.

Trains a single LSTM model that uses all 3 signals (ABP, PLETH, II) as input
to predict the heart rate (BPM) derived from the upcoming waveform window.

Best checkpoint is saved based on minimum validation loss.

Usage:
    python model.py              # Train with default 20 epochs
    python model.py --epochs 50  # Custom epoch count
"""

import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from preprocess import create_dataloaders, INPUT_LENGTH, TARGET_LENGTH, NUM_SIGNALS, SIGNAL_NAMES


# Configuration
HIDDEN_SIZE = 64
NUM_LAYERS = 2
INPUT_SIZE = NUM_SIGNALS  # 4 signals as input
OUTPUT_SIZE = 1           # Single HR value
NUM_EPOCHS = 20
LEARNING_RATE = 0.001
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(BASE_DIR, 'checkpoints')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')


def get_checkpoint_dir(num_epochs=NUM_EPOCHS, input_length=INPUT_LENGTH, target_length=TARGET_LENGTH):
    """Get checkpoint directory with input/target length and epochs in name."""
    return os.path.join(BASE_DIR, 'checkpoints', f'in{input_length}_tgt{target_length}_epochs_{num_epochs}')


def get_output_dir(num_epochs=NUM_EPOCHS, input_length=INPUT_LENGTH, target_length=TARGET_LENGTH):
    """Get output directory with input/target length and epochs in name."""
    return os.path.join(BASE_DIR, 'outputs', f'in{input_length}_tgt{target_length}_epochs_{num_epochs}')


class LSTMHeartRate(nn.Module):
    """
    LSTM-based model for heart rate prediction from physiological waveforms.

    Architecture:
        - 2-layer LSTM with hidden_size=64, input_size=4 (all signals)
        - Fully connected output layer predicting a single HR value
    """

    def __init__(self, input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE,
                 num_layers=NUM_LAYERS, output_size=OUTPUT_SIZE):
        super(LSTMHeartRate, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1,
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, output_size),
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Tensor of shape (batch, seq_len, input_size=3)

        Returns:
            Tensor of shape (batch, 1) — predicted heart rate (normalized)
        """
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_size)
        last_output = lstm_out[:, -1, :]  # (batch, hidden_size)
        prediction = self.fc(last_output)  # (batch, 1)
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


def train_model(device, num_epochs=NUM_EPOCHS, input_length=INPUT_LENGTH, target_length=TARGET_LENGTH):
    """
    Train the LSTM heart rate prediction model.

    Args:
        device: torch device.
        num_epochs: Number of training epochs.
        input_length: Input window length in samples.
        target_length: Target window length in samples.

    Returns:
        Tuple of (train_losses, val_losses, best_val_loss).
    """
    checkpoint_dir = get_checkpoint_dir(num_epochs, input_length, target_length)
    print(f"\n{'=' * 60}")
    print(f"Training LSTM for Heart Rate Prediction")
    print(f"Input: {input_length} samples ({input_length/125:.1f}s) of all 3 signals ({', '.join(SIGNAL_NAMES)})")
    print(f"Target: HR from next {target_length} samples ({target_length/125:.1f}s)")
    print(f"Output: Heart Rate (BPM)")
    print(f"{'=' * 60}")

    # Load data
    print(f"\n[INFO] Loading data...")
    train_loader, val_loader, test_loader, norm_params = create_dataloaders(
        input_length=input_length, target_length=target_length
    )

    # Create model
    model = LSTMHeartRate().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Model: input_size={INPUT_SIZE}, hidden={HIDDEN_SIZE}, "
          f"layers={NUM_LAYERS}, output={OUTPUT_SIZE}, params={total_params:,}")

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
            checkpoint_path = os.path.join(checkpoint_dir, 'best_model_hr.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'norm_params': norm_params,
                'num_epochs': num_epochs,
            }, checkpoint_path)
            status = "* best *"

        print(f"{epoch:>6} | {train_loss:>12.6f} | {val_loss:>12.6f} | {status:>10}")

    print("-" * 50)
    print(f"[INFO] Best val loss: {best_val_loss:.6f}")
    print(f"[SAVED] {os.path.join(checkpoint_dir, 'best_model_hr.pt')}")

    return train_losses, val_losses, best_val_loss


def plot_training_curves(train_losses, val_losses, output_dir):
    """Save training curves."""
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, '-o', color='tab:blue',
            markersize=3, label='Train', alpha=0.8)
    ax.plot(epochs, val_losses, '--s', color='tab:red',
            markersize=3, label='Val', alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss (normalized HR)')
    ax.set_title('Heart Rate Prediction — Training Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    filepath = os.path.join(output_dir, 'training_curves.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] Training curves: {filepath}")


def main(num_epochs=None, input_length=None, target_length=None):
    """Train the LSTM heart rate model.

    Args:
        num_epochs: Override default NUM_EPOCHS if provided.
        input_length: Override default INPUT_LENGTH if provided.
        target_length: Override default TARGET_LENGTH if provided.
    """
    epochs = num_epochs if num_epochs is not None else NUM_EPOCHS
    in_len = input_length if input_length is not None else INPUT_LENGTH
    tgt_len = target_length if target_length is not None else TARGET_LENGTH
    output_dir = get_output_dir(epochs, in_len, tgt_len)

    print("=" * 60)
    print("MIMIC-III Heart Rate Prediction — LSTM Training")
    print("=" * 60)
    print(f"Input: {in_len} samples ({in_len/125:.1f}s) of {SIGNAL_NAMES}")
    print(f"Target: HR from next {tgt_len} samples ({tgt_len/125:.1f}s)")
    print(f"Output: Heart Rate (BPM)")
    print(f"Epochs: {epochs}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Device: {device}")

    train_losses, val_losses, best_val = train_model(device, epochs, in_len, tgt_len)

    # Plot training curves
    plot_training_curves(train_losses, val_losses, output_dir)

    # Summary
    checkpoint_dir = get_checkpoint_dir(epochs, in_len, tgt_len)
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Best val loss: {best_val:.6f}")
    print(f"  Final train loss: {train_losses[-1]:.6f}")
    print(f"  Checkpoint: {checkpoint_dir}/best_model_hr.pt")
    print(f"  Training plot: {output_dir}/training_curves.png")
    print("=" * 60)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train LSTM model for heart rate prediction')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS,
                        help=f'Number of training epochs (default: {NUM_EPOCHS})')
    parser.add_argument('--input-length', type=int, default=INPUT_LENGTH,
                        help=f'Input window length in samples (default: {INPUT_LENGTH})')
    parser.add_argument('--target-length', type=int, default=TARGET_LENGTH,
                        help=f'Target window length in samples (default: {TARGET_LENGTH})')
    args = parser.parse_args()
    main(num_epochs=args.epochs, input_length=args.input_length, target_length=args.target_length)
