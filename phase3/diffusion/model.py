#!/usr/bin/env python3
"""
model.py - Conditional DDPM for multivariate waveform forecasting (Phase 3).

Trains a conditional diffusion model that denoises a 150-value forecast
(25 steps × 6 features), conditioned on 75 time steps of aggregated features
from all 4 signals.

Architecture:
    - Condition Encoder: MLP + Transformer on input sequence (75 × 24 features)
    - Denoiser: 1D U-Net with cross-attention to condition tokens
    - Diffusion: T=200 steps, linear beta schedule

Usage:
    python model.py --target II
    python model.py --target PLETH --epochs 100
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

from preprocess import (create_dataloaders, INPUT_LENGTH, OUTPUT_LENGTH,
                        NUM_SIGNALS, NUM_FEATURES, SIGNAL_NAMES, VALID_TARGETS,
                        INTERVAL_MINUTES, FEATURE_NAMES)


# Configuration
NUM_TIMESTEPS = 200       # Diffusion steps (T)
NUM_EPOCHS = 100
LEARNING_RATE = 0.0002
BETA_START = 1e-4
BETA_END = 0.02
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_SIZE = NUM_SIGNALS * NUM_FEATURES  # 24
OUTPUT_FLAT_SIZE = OUTPUT_LENGTH * NUM_FEATURES  # 25 × 6 = 150


def get_checkpoint_dir(target_signal, num_epochs=NUM_EPOCHS):
    return os.path.join(BASE_DIR, 'checkpoints', f'{target_signal}_epochs_{num_epochs}')


def get_output_dir(target_signal, num_epochs=NUM_EPOCHS):
    return os.path.join(BASE_DIR, 'outputs', f'{target_signal}_epochs_{num_epochs}')


# ============================================================
# Diffusion Schedule
# ============================================================

def get_beta_schedule(num_timesteps=NUM_TIMESTEPS):
    return torch.linspace(BETA_START, BETA_END, num_timesteps)


def get_diffusion_params(betas):
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
        'alphas_cumprod_prev': alphas_cumprod_prev,
        'sqrt_alphas_cumprod': sqrt_alphas_cumprod,
        'sqrt_one_minus_alphas_cumprod': sqrt_one_minus_alphas_cumprod,
        'sqrt_recip_alphas': sqrt_recip_alphas,
        'posterior_variance': posterior_variance,
    }


# ============================================================
# Sinusoidal Time Embedding
# ============================================================

class SinusoidalTimeEmbedding(nn.Module):
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
# Building Blocks
# ============================================================

class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.GroupNorm(min(8, in_channels), in_channels),
            nn.SiLU(),
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
        )
        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_channels),
        )
        self.conv2 = nn.Sequential(
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
        )
        self.skip = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(x)
        h = h + self.time_proj(t_emb)[:, :, None]
        h = self.conv2(h)
        return h + self.skip(x)


class CrossAttention1D(nn.Module):
    def __init__(self, channels, context_dim, num_heads=4):
        super().__init__()
        self.norm_x = nn.GroupNorm(min(8, channels), channels)
        self.norm_ctx = nn.LayerNorm(context_dim)
        self.proj_ctx = nn.Linear(context_dim, channels) if context_dim != channels else nn.Identity()
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True, dropout=0.1)

    def forward(self, x, context):
        b, c, s = x.shape
        h = self.norm_x(x).transpose(1, 2)
        ctx = self.proj_ctx(self.norm_ctx(context))
        h, _ = self.attn(h, ctx, ctx)
        h = h.transpose(1, 2)
        return x + h


class SelfAttention1D(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True, dropout=0.1)

    def forward(self, x):
        b, c, s = x.shape
        h = self.norm(x).transpose(1, 2)
        h, _ = self.attn(h, h, h)
        h = h.transpose(1, 2)
        return x + h


# ============================================================
# Condition Encoder
# ============================================================

class ConditionEncoder(nn.Module):
    """
    Encodes 75 time steps of 24 features into a sequence of context tokens.
    MLP projection + Transformer self-attention.
    """

    def __init__(self, input_size=INPUT_SIZE, context_dim=128, num_layers=3, num_heads=4):
        super().__init__()
        self.context_dim = context_dim

        # Project input features to context_dim
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, context_dim),
            nn.SiLU(),
            nn.Linear(context_dim, context_dim),
        )

        # Positional embedding for 75 time steps
        self.pos_embed = nn.Parameter(torch.randn(1, INPUT_LENGTH, context_dim) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=context_dim,
            nhead=num_heads,
            dim_feedforward=context_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.final_norm = nn.LayerNorm(context_dim)

    def forward(self, condition):
        """
        Args:
            condition: (batch, 75, 24)
        Returns:
            context: (batch, 75, context_dim)
        """
        x = self.input_proj(condition)
        x = x + self.pos_embed[:, :x.shape[1], :]
        x = self.transformer(x)
        x = self.final_norm(x)
        return x


# ============================================================
# 1D U-Net Denoiser
# ============================================================

class ConditionalDenoiser(nn.Module):
    """
    1D U-Net denoiser with cross-attention to condition tokens.

    Operates on noisy 150-step flattened forecast sequence (25 steps × 6 features).
    Levels: 150 → 75 → 38 (bottleneck) → 75 → 150
    """

    def __init__(self, output_size=OUTPUT_FLAT_SIZE, context_dim=128, time_dim=128):
        super().__init__()
        self.output_size = output_size
        ch0, ch1, ch_bot = 128, 192, 192

        # Time embedding
        self.time_embed = SinusoidalTimeEmbedding(time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim),
        )

        # Condition encoder
        self.condition_encoder = ConditionEncoder(
            input_size=INPUT_SIZE, context_dim=context_dim,
            num_layers=3, num_heads=4,
        )

        # Input projection
        self.input_proj = nn.Conv1d(1, ch0, kernel_size=1)

        # Encoder
        self.enc0_res = ResidualBlock1D(ch0, ch0, time_dim)
        self.enc0_attn = CrossAttention1D(ch0, context_dim, num_heads=4)
        self.down0 = nn.Conv1d(ch0, ch1, kernel_size=3, stride=2, padding=1)  # 150→75

        self.enc1_res = ResidualBlock1D(ch1, ch1, time_dim)
        self.enc1_attn = CrossAttention1D(ch1, context_dim, num_heads=4)
        self.down1 = nn.Conv1d(ch1, ch_bot, kernel_size=3, stride=2, padding=1)  # 75→38

        # Bottleneck
        self.bot_res1 = ResidualBlock1D(ch_bot, ch_bot, time_dim)
        self.bot_self_attn = SelfAttention1D(ch_bot, num_heads=4)
        self.bot_cross_attn = CrossAttention1D(ch_bot, context_dim, num_heads=4)
        self.bot_res2 = ResidualBlock1D(ch_bot, ch_bot, time_dim)

        # Decoder
        self.up1 = nn.ConvTranspose1d(ch_bot, ch1, kernel_size=4, stride=2, padding=1)  # 38→75
        self.dec1_res = ResidualBlock1D(ch1 * 2, ch1, time_dim)
        self.dec1_attn = CrossAttention1D(ch1, context_dim, num_heads=4)

        self.up0 = nn.ConvTranspose1d(ch1, ch0, kernel_size=4, stride=2, padding=1)  # 75→150
        self.dec0_res = ResidualBlock1D(ch0 * 2, ch0, time_dim)
        self.dec0_attn = CrossAttention1D(ch0, context_dim, num_heads=4)

        # Output projection
        self.output_proj = nn.Sequential(
            nn.GroupNorm(8, ch0),
            nn.SiLU(),
            nn.Conv1d(ch0, 1, kernel_size=1),
        )

    def forward(self, x_noisy, t, condition):
        """
        Args:
            x_noisy: (batch, 150) — noisy flattened forecast (25 steps × 6 features)
            t: (batch,) — diffusion timesteps
            condition: (batch, 75, 24) — input features
        Returns:
            predicted noise: (batch, 150)
        """
        t_emb = self.time_mlp(self.time_embed(t))
        context = self.condition_encoder(condition)

        x = x_noisy.unsqueeze(1)  # (batch, 1, 150)
        x = self.input_proj(x)     # (batch, ch0, 150)

        # Encoder level 0
        h0 = self.enc0_res(x, t_emb)
        h0 = self.enc0_attn(h0, context)
        x = self.down0(h0)  # (batch, ch1, 75)

        # Encoder level 1
        h1 = self.enc1_res(x, t_emb)
        h1 = self.enc1_attn(h1, context)
        x = self.down1(h1)  # (batch, ch_bot, 38)

        # Bottleneck
        x = self.bot_res1(x, t_emb)
        x = self.bot_self_attn(x)
        x = self.bot_cross_attn(x, context)
        x = self.bot_res2(x, t_emb)

        # Decoder level 1
        x = self.up1(x)[:, :, :h1.shape[2]]  # trim to match h1
        x = torch.cat([x, h1], dim=1)
        x = self.dec1_res(x, t_emb)
        x = self.dec1_attn(x, context)

        # Decoder level 0
        x = self.up0(x)[:, :, :h0.shape[2]]  # trim to match h0
        x = torch.cat([x, h0], dim=1)
        x = self.dec0_res(x, t_emb)
        x = self.dec0_attn(x, context)

        # Output
        x = self.output_proj(x)  # (batch, 1, 150)
        x = x.squeeze(1)         # (batch, 150)
        return x


# ============================================================
# DDPM Forward & Reverse Process
# ============================================================

def q_sample(x_start, t, diffusion_params, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)
    sqrt_alpha_cumprod = diffusion_params['sqrt_alphas_cumprod'][t][:, None]
    sqrt_one_minus = diffusion_params['sqrt_one_minus_alphas_cumprod'][t][:, None]
    return sqrt_alpha_cumprod * x_start + sqrt_one_minus * noise, noise


@torch.no_grad()
def p_sample(model, x_t, t, diffusion_params, condition):
    betas = diffusion_params['betas']
    sqrt_one_minus = diffusion_params['sqrt_one_minus_alphas_cumprod']
    sqrt_alphas_cumprod = diffusion_params['sqrt_alphas_cumprod']
    posterior_var = diffusion_params['posterior_variance']
    alphas_cumprod = diffusion_params['alphas_cumprod']
    alphas_cumprod_prev = diffusion_params['alphas_cumprod_prev']

    noise_pred = model(x_t, t, condition)

    beta_t = betas[t][:, None]
    sqrt_one_minus_t = sqrt_one_minus[t][:, None]
    sqrt_alpha_cumprod_t = sqrt_alphas_cumprod[t][:, None]

    pred_x0 = (x_t - sqrt_one_minus_t * noise_pred) / sqrt_alpha_cumprod_t
    pred_x0 = torch.clamp(pred_x0, -6.0, 6.0)

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
    batch_size = condition.shape[0]
    x = torch.randn(batch_size, OUTPUT_FLAT_SIZE, device=device)
    for i in reversed(range(num_timesteps)):
        t = torch.full((batch_size,), i, device=device, dtype=torch.long)
        x = p_sample(model, x, t, diffusion_params, condition)
    return x


# ============================================================
# Training & Validation
# ============================================================

def train_one_epoch(model, train_loader, optimizer, diffusion_params, device):
    model.train()
    total_loss = 0.0
    num_batches = 0

    for x_cond, y_target in train_loader:
        x_cond = x_cond.to(device)
        # Flatten target from (batch, 25, 6) to (batch, 150)
        y_target = y_target.to(device).view(y_target.shape[0], -1)

        t = torch.randint(0, NUM_TIMESTEPS, (y_target.shape[0],), device=device)
        noise = torch.randn_like(y_target)
        y_noisy, _ = q_sample(y_target, t, diffusion_params, noise)

        noise_pred = model(y_noisy, t, x_cond)
        loss = F.mse_loss(noise_pred, noise)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


def validate(model, val_loader, diffusion_params, device):
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for x_cond, y_target in val_loader:
            x_cond = x_cond.to(device)
            # Flatten target from (batch, 25, 6) to (batch, 150)
            y_target = y_target.to(device).view(y_target.shape[0], -1)

            t = torch.randint(0, NUM_TIMESTEPS, (y_target.shape[0],), device=device)
            noise = torch.randn_like(y_target)
            y_noisy, _ = q_sample(y_target, t, diffusion_params, noise)

            noise_pred = model(y_noisy, t, x_cond)
            loss = F.mse_loss(noise_pred, noise)

            total_loss += loss.item()
            num_batches += 1

    return total_loss / max(num_batches, 1)


# ============================================================
# Training Loop
# ============================================================

def train_model(device, target_signal='II', num_epochs=NUM_EPOCHS):
    """Train the diffusion forecasting model."""
    checkpoint_dir = get_checkpoint_dir(target_signal, num_epochs)

    print(f"\n{'=' * 60}")
    print(f"Training Diffusion Forecaster — Target: {target_signal}")
    print(f"Input: {INPUT_LENGTH} intervals × {INPUT_SIZE} features")
    print(f"Output: {OUTPUT_LENGTH} intervals of {target_signal} mean")
    print(f"Diffusion: T={NUM_TIMESTEPS}, beta=[{BETA_START}, {BETA_END}]")
    print(f"{'=' * 60}")

    # Load data
    train_loader, val_loader, test_loader, norm_params = create_dataloaders(
        target_signal=target_signal
    )

    # Diffusion schedule
    betas = get_beta_schedule()
    diffusion_params = get_diffusion_params(betas)
    # Move to device
    for k, v in diffusion_params.items():
        diffusion_params[k] = v.to(device)

    # Create model
    model = ConditionalDenoiser(
        output_size=OUTPUT_FLAT_SIZE, context_dim=128, time_dim=128
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Model params: {total_params:,}")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    print(f"\n[INFO] Training for {num_epochs} epochs...")
    print("-" * 55)
    print(f"{'Epoch':>6} | {'Train Loss':>12} | {'Val Loss':>12} | {'LR':>10} | {'Status':>8}")
    print("-" * 55)

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, diffusion_params, device)
        val_loss = validate(model, val_loader, diffusion_params, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step()

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
                'num_timesteps': NUM_TIMESTEPS,
            }, checkpoint_path)
            status = "* best *"

        if epoch <= 5 or epoch % 10 == 0 or status:
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
    ax.plot(epochs, train_losses, '-', color='tab:blue', linewidth=1, label='Train', alpha=0.8)
    ax.plot(epochs, val_losses, '--', color='tab:red', linewidth=1, label='Val', alpha=0.8)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Noise Prediction Loss (MSE)')
    ax.set_title(f'Phase 3 Diffusion — {target_signal} Forecasting — Training Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    filepath = os.path.join(output_dir, 'training_curves.png')
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {filepath}")


def main(target_signal=None, num_epochs=None):
    """Train the diffusion forecasting model."""
    target = target_signal if target_signal is not None else 'II'
    epochs = num_epochs if num_epochs is not None else NUM_EPOCHS
    output_dir = get_output_dir(target, epochs)

    print("=" * 60)
    print(f"Phase 3 — Multivariate Waveform Forecasting (Diffusion)")
    print("=" * 60)
    print(f"Target signal: {target}")
    print(f"Input: {INPUT_LENGTH} intervals ({INPUT_LENGTH * INTERVAL_MINUTES / 60:.1f} hrs)")
    print(f"Output: {OUTPUT_LENGTH} intervals ({OUTPUT_LENGTH * INTERVAL_MINUTES / 60:.1f} hrs)")
    print(f"Epochs: {epochs}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] Device: {device}")

    train_losses, val_losses, best_val = train_model(device, target, epochs)
    plot_training_curves(train_losses, val_losses, target, output_dir)

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
    parser = argparse.ArgumentParser(description='Train Diffusion forecaster')
    parser.add_argument('--target', type=str, default='II', choices=VALID_TARGETS,
                        help='Target signal to forecast (default: II)')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS,
                        help=f'Number of training epochs (default: {NUM_EPOCHS})')
    args = parser.parse_args()
    main(target_signal=args.target, num_epochs=args.epochs)
