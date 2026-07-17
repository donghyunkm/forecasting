#!/usr/bin/env python3
"""
model.py - Diffusion model (DDPM) for MIMIC-III waveform forecasting.

Trains 3 separate conditional DDPM models, one per target signal (ABP, PLETH, II).
Each model:
  - Condition: 125-step window of all 3 signals (input_size=3)
  - Target: denoises a 25-step forecast of the target signal
  - Architecture: 1D U-Net denoiser conditioned on the input window

Best checkpoint per model is saved based on minimum validation loss.

Usage:
    python model.py    # Train all 3 diffusion models
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

from preprocess import create_dataloaders, FORECAST_HORIZON, INPUT_LENGTH, \
    NUM_SIGNALS, SIGNAL_NAMES


# Configuration
NUM_TIMESTEPS = 200       # Diffusion steps (T)
NUM_EPOCHS = 20
LEARNING_RATE = 0.001
BETA_START = 1e-4
BETA_END = 0.02
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(BASE_DIR, 'checkpoints')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')


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
# 1D U-Net Denoiser (conditioned on input window)
# ============================================================

class ConditionalDenoiser(nn.Module):
    """
    1D denoiser network conditioned on input window.

    Takes:
        - x_noisy: noisy target (batch, forecast_horizon) — the signal being denoised
        - t: diffusion timestep (batch,)
        - condition: input window (batch, input_length, num_signals)

    Returns:
        - predicted noise (batch, forecast_horizon)
    """

    def __init__(self, forecast_horizon=FORECAST_HORIZON, input_length=INPUT_LENGTH,
                 num_signals=NUM_SIGNALS, hidden_dim=128, time_dim=64):
        super().__init__()

        self.forecast_horizon = forecast_horizon
        self.hidden_dim = hidden_dim

        # Time embedding
        self.time_embed = SinusoidalTimeEmbedding(time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Condition encoder: process the input window (125, 3) -> fixed-size embedding
        self.cond_encoder = nn.Sequential(
            nn.Linear(num_signals, 64),
            nn.SiLU(),
            nn.Linear(64, 64),
        )
        self.cond_lstm = nn.LSTM(
            input_size=64, hidden_size=hidden_dim,
            num_layers=1, batch_first=True, bidirectional=False,
        )
        # Project LSTM output to condition vector
        self.cond_proj = nn.Linear(hidden_dim, hidden_dim)

        # Denoiser: takes noisy signal + condition + time -> predicted noise
        # Input: forecast_horizon (noisy x) concatenated with condition info
        self.input_proj = nn.Linear(forecast_horizon, hidden_dim)

        # Main denoising blocks
        self.blocks = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),  # x + cond + time
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, forecast_horizon),
        )

    def forward(self, x_noisy, t, condition):
        """
        Args:
            x_noisy: (batch, forecast_horizon) - noisy target
            t: (batch,) - integer timesteps
            condition: (batch, input_length, num_signals) - input window

        Returns:
            predicted noise: (batch, forecast_horizon)
        """
        # Time embedding
        t_emb = self.time_embed(t)  # (batch, time_dim)
        t_emb = self.time_mlp(t_emb)  # (batch, hidden_dim)

        # Condition encoding
        cond = self.cond_encoder(condition)  # (batch, input_length, 64)
        cond_out, _ = self.cond_lstm(cond)  # (batch, input_length, hidden_dim)
        cond_vec = self.cond_proj(cond_out[:, -1, :])  # (batch, hidden_dim) — last step

        # Noisy signal projection
        x_proj = self.input_proj(x_noisy)  # (batch, hidden_dim)

        # Concatenate all information
        combined = torch.cat([x_proj, cond_vec, t_emb], dim=-1)  # (batch, hidden_dim*3)

        # Predict noise
        noise_pred = self.blocks(combined)  # (batch, forecast_horizon)

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
    sqrt_recip_alphas = diffusion_params['sqrt_recip_alphas']
    sqrt_one_minus = diffusion_params['sqrt_one_minus_alphas_cumprod']
    posterior_var = diffusion_params['posterior_variance']

    # Predict noise
    noise_pred = model(x_t, t, condition)

    # Compute x_{t-1}
    beta_t = betas[t][:, None]
    sqrt_recip_alpha_t = sqrt_recip_alphas[t][:, None]
    sqrt_one_minus_t = sqrt_one_minus[t][:, None]

    model_mean = sqrt_recip_alpha_t * (x_t - beta_t * noise_pred / sqrt_one_minus_t)

    if t[0] == 0:
        return model_mean
    else:
        posterior_var_t = posterior_var[t][:, None]
        noise = torch.randn_like(x_t)
        return model_mean + torch.sqrt(posterior_var_t) * noise


@torch.no_grad()
def sample(model, condition, diffusion_params, device):
    """
    Full reverse diffusion: generate forecast from pure noise.

    Args:
        model: denoiser network
        condition: (batch, input_length, num_signals)
        diffusion_params: precomputed diffusion schedule
        device: torch device

    Returns:
        (batch, forecast_horizon) — denoised prediction
    """
    batch_size = condition.shape[0]
    x = torch.randn(batch_size, FORECAST_HORIZON, device=device)

    for i in reversed(range(NUM_TIMESTEPS)):
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
        x_cond = x_cond.to(device)      # (batch, 125, 3)
        y_target = y_target.to(device)   # (batch, 25)

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


def train_single_model(target_idx, device, num_epochs=NUM_EPOCHS):
    """
    Train a single diffusion model for a target signal.

    Args:
        target_idx: Index of target signal (0=ABP, 1=PLETH, 2=II).
        device: torch device.
        num_epochs: Number of training epochs.

    Returns:
        Tuple of (train_losses, val_losses, best_val_loss).
    """
    signal_name = SIGNAL_NAMES[target_idx]
    print(f"\n{'=' * 60}")
    print(f"Training DDPM for: {signal_name} (target_idx={target_idx})")
    print(f"Condition: all 3 signals ({', '.join(SIGNAL_NAMES)})")
    print(f"Diffusion timesteps: {NUM_TIMESTEPS}")
    print(f"{'=' * 60}")

    # Load data
    print(f"\n[INFO] Loading data (target: {signal_name})...")
    train_loader, val_loader, test_loader, norm_params = create_dataloaders(target_idx)

    # Diffusion schedule
    betas = get_beta_schedule().to(device)
    diffusion_params = get_diffusion_params(betas)
    # Move all params to device
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
            os.makedirs(CHECKPOINT_DIR, exist_ok=True)
            checkpoint_path = os.path.join(
                CHECKPOINT_DIR, f'best_model_{signal_name.lower()}.pt'
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
                'num_timesteps': NUM_TIMESTEPS,
                'beta_start': BETA_START,
                'beta_end': BETA_END,
            }, checkpoint_path)
            status = "* best *"

        print(f"{epoch:>6} | {train_loss:>12.6f} | {val_loss:>12.6f} | {status:>10}")

    print("-" * 50)
    print(f"[INFO] Best val loss for {signal_name}: {best_val_loss:.6f}")
    print(f"[SAVED] {os.path.join(CHECKPOINT_DIR, f'best_model_{signal_name.lower()}.pt')}")

    return train_losses, val_losses, best_val_loss


def plot_training_curves(all_train_losses, all_val_losses, output_dir):
    """Save training curves for all 3 models."""
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = ['tab:blue', 'tab:green', 'tab:red']

    for i, (ax, signal_name) in enumerate(zip(axes, SIGNAL_NAMES)):
        epochs = range(1, len(all_train_losses[i]) + 1)
        ax.plot(epochs, all_train_losses[i], '-o', color=colors[i],
                markersize=3, label='Train', alpha=0.8)
        ax.plot(epochs, all_val_losses[i], '--s', color=colors[i],
                markersize=3, label='Val', alpha=0.8)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Noise Prediction Loss (MSE)')
        ax.set_title(f'{signal_name} DDPM')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle('Diffusion Training Curves — All Signals', fontsize=13, fontweight='bold')
    plt.tight_layout()

    filepath = os.path.join(output_dir, 'training_curves.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] Training curves: {filepath}")


def main(num_epochs=None):
    """Train all 3 DDPM models (one per signal).
    
    Args:
        num_epochs: Override default NUM_EPOCHS if provided.
    """
    epochs = num_epochs if num_epochs is not None else NUM_EPOCHS

    print("=" * 60)
    print("MIMIC-III Waveform Forecasting — Diffusion (DDPM) Training")
    print("=" * 60)
    print(f"Signals: {SIGNAL_NAMES}")
    print(f"Architecture: 3 separate conditional DDPMs")
    print(f"Each model conditioned on all 3 signals, denoises target signal forecast")
    print(f"Diffusion steps: {NUM_TIMESTEPS}, Beta: [{BETA_START}, {BETA_END}]")
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

    # Plot training curves
    plot_training_curves(all_train_losses, all_val_losses, OUTPUT_DIR)

    # Summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE — All DDPM Models")
    print("=" * 60)
    for i, name in enumerate(SIGNAL_NAMES):
        print(f"  {name:>5}: best_val_loss = {all_best_val[i]:.6f} "
              f"| final_train = {all_train_losses[i][-1]:.6f}")
    print(f"\n  Checkpoints: {CHECKPOINT_DIR}/best_model_{{signal}}.pt")
    print(f"  Training plot: {OUTPUT_DIR}/training_curves.png")
    print("=" * 60)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train diffusion models for waveform forecasting')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS,
                        help=f'Number of training epochs (default: {NUM_EPOCHS})')
    args = parser.parse_args()
    main(num_epochs=args.epochs)
