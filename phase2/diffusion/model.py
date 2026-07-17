#!/usr/bin/env python3
"""
model.py - Diffusion model (DDPM) for heart rate prediction from waveforms.

Trains a single conditional DDPM model:
  - Condition: INPUT_LENGTH-step window of all 3 signals (input_size=3)
  - Target: denoises a scalar HR prediction
  - Architecture: LSTM condition encoder + MLP denoiser

Best checkpoint is saved based on minimum validation loss.

Usage:
    python model.py              # Train with default 20 epochs
    python model.py --epochs 100 # Custom epoch count
"""

import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from preprocess import create_dataloaders, INPUT_LENGTH, TARGET_LENGTH, NUM_SIGNALS, SIGNAL_NAMES


# Configuration
NUM_TIMESTEPS = 200       # Diffusion steps (T)
NUM_EPOCHS = 20
LEARNING_RATE = 0.001
BETA_START = 1e-4
BETA_END = 0.02
OUTPUT_SIZE = 1           # Single HR value
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(BASE_DIR, 'checkpoints')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')


def get_checkpoint_dir(num_epochs=NUM_EPOCHS, input_length=INPUT_LENGTH, target_length=TARGET_LENGTH):
    """Get checkpoint directory with input/target length and epochs in name."""
    return os.path.join(BASE_DIR, 'checkpoints', f'in{input_length}_tgt{target_length}_epochs_{num_epochs}')


def get_output_dir(num_epochs=NUM_EPOCHS, input_length=INPUT_LENGTH, target_length=TARGET_LENGTH):
    """Get output directory with input/target length and epochs in name."""
    return os.path.join(BASE_DIR, 'outputs', f'in{input_length}_tgt{target_length}_epochs_{num_epochs}')


# ============================================================
# Diffusion Schedule
# ============================================================

def get_beta_schedule(num_timesteps=NUM_TIMESTEPS, beta_start=BETA_START, beta_end=BETA_END):
    """Linear beta schedule."""
    return torch.linspace(beta_start, beta_end, num_timesteps)


def get_diffusion_params(betas):
    """Precompute all diffusion parameters from beta schedule."""
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

    return {
        'betas': betas,
        'alphas': alphas,
        'alphas_cumprod': alphas_cumprod,
        'sqrt_alphas_cumprod': sqrt_alphas_cumprod,
        'sqrt_one_minus_alphas_cumprod': sqrt_one_minus_alphas_cumprod,
        'sqrt_recip_alphas': sqrt_recip_alphas,
        'posterior_variance': posterior_variance,
    }


# ============================================================
# Sinusoidal Time Embedding
# ============================================================

class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional embedding for diffusion timestep."""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


# ============================================================
# Conditional Denoiser for Heart Rate
# ============================================================

class ConditionalDenoiser(nn.Module):
    """
    Denoiser network conditioned on input waveform window.

    Uses a strided 1D CNN to compress the long input sequence (e.g., 37,500 steps)
    into a short representation (~150 tokens), then applies a bidirectional LSTM
    to capture temporal patterns before projecting to a condition vector.

    Compression stages (for 37,500 input @ 3 signals):
        Conv1d stride 5: 37,500 → 7,500  (channels: 3 → 32)
        Conv1d stride 5: 7,500 → 1,500   (channels: 32 → 64)
        Conv1d stride 5: 1,500 → 300     (channels: 64 → 128)
        Conv1d stride 2: 300 → 150       (channels: 128 → 128)
    Total compression: 250x (37,500 → 150 tokens)

    Takes:
        - x_noisy: noisy HR target (batch, 1) — the scalar being denoised
        - t: diffusion timestep (batch,)
        - condition: input window (batch, input_length, num_signals)

    Returns:
        - predicted noise (batch, 1)
    """

    def __init__(self, output_size=OUTPUT_SIZE, input_length=INPUT_LENGTH,
                 num_signals=NUM_SIGNALS, hidden_dim=128, time_dim=64):
        super().__init__()

        self.output_size = output_size
        self.hidden_dim = hidden_dim

        # Time embedding
        self.time_embed = SinusoidalTimeEmbedding(time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Condition encoder: strided 1D CNN for compression
        # Input shape: (batch, num_signals, input_length) after transpose
        self.cond_cnn = nn.Sequential(
            # Stage 1: stride 5, compress 5x
            nn.Conv1d(num_signals, 32, kernel_size=15, stride=5, padding=7),
            nn.BatchNorm1d(32),
            nn.SiLU(),
            # Stage 2: stride 5, compress 5x
            nn.Conv1d(32, 64, kernel_size=11, stride=5, padding=5),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            # Stage 3: stride 5, compress 5x
            nn.Conv1d(64, 128, kernel_size=7, stride=5, padding=3),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            # Stage 4: stride 2, compress 2x
            nn.Conv1d(128, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.SiLU(),
        )

        # Bidirectional LSTM on compressed sequence
        self.cond_lstm = nn.LSTM(
            input_size=128, hidden_size=hidden_dim,
            num_layers=2, batch_first=True, bidirectional=True,
            dropout=0.1,
        )
        # Project bidirectional output to condition vector
        self.cond_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # *2 for bidirectional
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Noisy HR projection
        self.input_proj = nn.Linear(output_size, hidden_dim)

        # Main denoising blocks (deeper with residual-style connections)
        self.blocks = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),  # x + cond + time
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_size),
        )

    def forward(self, x_noisy, t, condition):
        """
        Args:
            x_noisy: (batch, 1) - noisy HR target
            t: (batch,) - integer timesteps
            condition: (batch, input_length, num_signals) - input window

        Returns:
            predicted noise: (batch, 1)
        """
        # Time embedding
        t_emb = self.time_embed(t)  # (batch, time_dim)
        t_emb = self.time_mlp(t_emb)  # (batch, hidden_dim)

        # Condition encoding via CNN compression
        # Transpose to (batch, channels, time) for Conv1d
        cond = condition.transpose(1, 2)  # (batch, num_signals, input_length)
        cond = self.cond_cnn(cond)  # (batch, 128, compressed_len)
        # Transpose back to (batch, compressed_len, 128) for LSTM
        cond = cond.transpose(1, 2)  # (batch, compressed_len, 128)
        cond_out, _ = self.cond_lstm(cond)  # (batch, compressed_len, hidden_dim*2)
        # Use last hidden state from both directions
        cond_vec = self.cond_proj(cond_out[:, -1, :])  # (batch, hidden_dim)

        # Noisy HR projection
        x_proj = self.input_proj(x_noisy)  # (batch, hidden_dim)

        # Concatenate all information
        combined = torch.cat([x_proj, cond_vec, t_emb], dim=-1)  # (batch, hidden_dim*3)

        # Predict noise
        noise_pred = self.blocks(combined)  # (batch, 1)

        return noise_pred


# ============================================================
# DDPM Forward & Reverse Process
# ============================================================

def q_sample(x_start, t, diffusion_params, noise=None):
    """Forward diffusion: add noise to x_start at timestep t."""
    if noise is None:
        noise = torch.randn_like(x_start)

    sqrt_alpha_cumprod = diffusion_params['sqrt_alphas_cumprod'][t]  # (batch,)
    sqrt_one_minus = diffusion_params['sqrt_one_minus_alphas_cumprod'][t]  # (batch,)

    # Reshape for broadcasting: (batch, 1)
    sqrt_alpha_cumprod = sqrt_alpha_cumprod[:, None]
    sqrt_one_minus = sqrt_one_minus[:, None]

    return sqrt_alpha_cumprod * x_start + sqrt_one_minus * noise, noise


@torch.no_grad()
def p_sample(model, x_t, t, diffusion_params, condition):
    """Single reverse diffusion step: denoise from t to t-1."""
    betas = diffusion_params['betas']
    sqrt_one_minus = diffusion_params['sqrt_one_minus_alphas_cumprod']
    sqrt_alphas_cumprod = diffusion_params['sqrt_alphas_cumprod']
    posterior_var = diffusion_params['posterior_variance']

    # Predict noise
    noise_pred = model(x_t, t, condition)

    # Get schedule values
    beta_t = betas[t][:, None]
    sqrt_one_minus_t = sqrt_one_minus[t][:, None]
    sqrt_alpha_cumprod_t = sqrt_alphas_cumprod[t][:, None]

    # Predict x_0 from x_t and predicted noise, then clip
    pred_x0 = (x_t - sqrt_one_minus_t * noise_pred) / sqrt_alpha_cumprod_t
    pred_x0 = torch.clamp(pred_x0, -6.0, 6.0)  # Clip to ~6 std deviations

    # Recompute model mean from clipped x_0
    alphas_cumprod = diffusion_params['alphas_cumprod']
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    alpha_cumprod_t = alphas_cumprod[t][:, None]
    alpha_cumprod_prev_t = alphas_cumprod_prev[t][:, None]

    model_mean = (
        torch.sqrt(alpha_cumprod_prev_t) * beta_t / (1.0 - alpha_cumprod_t) * pred_x0
        + torch.sqrt(1.0 - beta_t) * (1.0 - alpha_cumprod_prev_t) / (1.0 - alpha_cumprod_t) * x_t
    )

    if t[0] == 0:
        return model_mean
    else:
        posterior_var_t = posterior_var[t][:, None]
        noise = torch.randn_like(x_t)
        return model_mean + torch.sqrt(posterior_var_t) * noise


@torch.no_grad()
def sample(model, condition, diffusion_params, device, num_timesteps=NUM_TIMESTEPS):
    """
    Full reverse diffusion: generate HR prediction from pure noise.

    Args:
        model: denoiser network
        condition: (batch, input_length, num_signals)
        diffusion_params: precomputed diffusion schedule
        device: torch device
        num_timesteps: number of diffusion steps

    Returns:
        (batch, 1) — denoised HR prediction (normalized)
    """
    batch_size = condition.shape[0]
    x = torch.randn(batch_size, OUTPUT_SIZE, device=device)

    for i in reversed(range(num_timesteps)):
        t = torch.full((batch_size,), i, device=device, dtype=torch.long)
        x = p_sample(model, x, t, diffusion_params, condition)

    return x


# ============================================================
# Training
# ============================================================

def train_one_epoch(model, train_loader, optimizer, diffusion_params, device):
    """Train for one epoch. Loss = MSE between predicted and actual noise."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for x_cond, y_target in train_loader:
        x_cond = x_cond.to(device)      # (batch, INPUT_LENGTH, 3)
        y_target = y_target.to(device)   # (batch, 1)

        # Sample random timestep for each example
        t = torch.randint(0, NUM_TIMESTEPS, (y_target.shape[0],), device=device)

        # Forward diffusion: add noise to target
        noise = torch.randn_like(y_target)
        y_noisy, _ = q_sample(y_target, t, diffusion_params, noise)

        # Predict the noise
        noise_pred = model(y_noisy, t, x_cond)

        # Loss: simple MSE between predicted and actual noise
        loss = F.mse_loss(noise_pred, noise)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def validate(model, val_loader, diffusion_params, device):
    """Validate: compute noise prediction loss on validation set."""
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for x_cond, y_target in val_loader:
            x_cond = x_cond.to(device)
            y_target = y_target.to(device)

            t = torch.randint(0, NUM_TIMESTEPS, (y_target.shape[0],), device=device)
            noise = torch.randn_like(y_target)
            y_noisy, _ = q_sample(y_target, t, diffusion_params, noise)

            noise_pred = model(y_noisy, t, x_cond)
            loss = F.mse_loss(noise_pred, noise)

            total_loss += loss.item()
            num_batches += 1

    return total_loss / max(num_batches, 1)


def train_model(device, num_epochs=NUM_EPOCHS, input_length=INPUT_LENGTH, target_length=TARGET_LENGTH):
    """
    Train the diffusion heart rate prediction model.

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
    print(f"Training DDPM for Heart Rate Prediction")
    print(f"Condition: {input_length} samples ({input_length/125:.1f}s) of all 3 signals ({', '.join(SIGNAL_NAMES)})")
    print(f"Target: HR from next {target_length} samples ({target_length/125:.1f}s)")
    print(f"Output: Heart Rate (1 value, normalized)")
    print(f"Diffusion timesteps: {NUM_TIMESTEPS}")
    print(f"{'=' * 60}")

    # Load data
    print(f"\n[INFO] Loading data...")
    train_loader, val_loader, test_loader, norm_params = create_dataloaders(
        input_length=input_length, target_length=target_length
    )

    # Diffusion schedule
    betas = get_beta_schedule().to(device)
    diffusion_params = get_diffusion_params(betas)
    diffusion_params = {k: v.to(device) for k, v in diffusion_params.items()}

    # Create model
    model = ConditionalDenoiser().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Model: ConditionalDenoiser, params={total_params:,}")

    # Optimizer
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
        train_loss = train_one_epoch(model, train_loader, optimizer, diffusion_params, device)
        val_loss = validate(model, val_loader, diffusion_params, device)

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
                'num_timesteps': NUM_TIMESTEPS,
                'beta_start': BETA_START,
                'beta_end': BETA_END,
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
    ax.set_ylabel('Noise Prediction Loss (MSE)')
    ax.set_title('Heart Rate DDPM — Training Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    filepath = os.path.join(output_dir, 'training_curves.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] Training curves: {filepath}")


def main(num_epochs=None, input_length=None, target_length=None):
    """Train the DDPM heart rate model.

    Args:
        num_epochs: Override default NUM_EPOCHS if provided.
        input_length: Override default INPUT_LENGTH if provided.
        target_length: Override default TARGET_LENGTH if provided.
    """
    epochs = num_epochs if num_epochs is not None else NUM_EPOCHS
    in_len = input_length if input_length is not None else INPUT_LENGTH
    tgt_len = target_length if target_length is not None else TARGET_LENGTH

    print("=" * 60)
    print("Heart Rate Prediction — Diffusion (DDPM) Training")
    print("=" * 60)
    print(f"Input: {in_len} samples ({in_len/125:.1f}s) of {SIGNAL_NAMES}")
    print(f"Target: HR from next {tgt_len} samples ({tgt_len/125:.1f}s)")
    print(f"Output: Heart Rate (BPM)")
    print(f"Diffusion steps: {NUM_TIMESTEPS}, Beta: [{BETA_START}, {BETA_END}]")
    print(f"Epochs: {epochs}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Device: {device}")

    train_losses, val_losses, best_val = train_model(device, epochs, in_len, tgt_len)

    # Plot training curves
    output_dir = get_output_dir(epochs, in_len, tgt_len)
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
    parser = argparse.ArgumentParser(description='Train diffusion model for heart rate prediction')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS,
                        help=f'Number of training epochs (default: {NUM_EPOCHS})')
    parser.add_argument('--input-length', type=int, default=INPUT_LENGTH,
                        help=f'Input window length in samples (default: {INPUT_LENGTH})')
    parser.add_argument('--target-length', type=int, default=TARGET_LENGTH,
                        help=f'Target window length in samples (default: {TARGET_LENGTH})')
    args = parser.parse_args()
    main(num_epochs=args.epochs, input_length=args.input_length, target_length=args.target_length)
