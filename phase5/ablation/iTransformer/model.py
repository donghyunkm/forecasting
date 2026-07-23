"""
iTransformer Ablation: Vitals-Only (Phase 5)

Same architecture as the full Phase 5 iTransformer but with only 5 input features
(4 vitals + 1 time position) instead of 12 (7 correlations + 4 vitals + 1 time).

This ablation measures the contribution of waveform correlation features by removing them.

Architecture: Each variate's full lookback sequence is treated as a single token.
Attention is applied across variates (not across time steps).
Output projects vital sign tokens to quantile predictions.
"""

import torch
import torch.nn as nn
import math


class DataEmbedding_inverted(nn.Module):
    """Projects each variate's full lookback (seq_len) into d_model via Linear + learned positional embedding."""

    def __init__(self, seq_len: int, d_model: int, n_input_vars: int, dropout: float = 0.1):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.n_input_vars = n_input_vars

        # Linear projection: each variate's seq_len history -> d_model
        self.value_embedding = nn.Linear(seq_len, d_model)

        # Learned positional embedding for each variate token
        self.position_embedding = nn.Parameter(torch.randn(1, n_input_vars, d_model) * 0.02)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        """
        Args:
            x: (B, seq_len, n_input_vars)
        Returns:
            (B, n_input_vars, d_model) - one token per variate
        """
        # Transpose to (B, n_input_vars, seq_len) so each variate has its full history
        x = x.permute(0, 2, 1)  # (B, n_input_vars, seq_len)

        # Project each variate's history to d_model
        x = self.value_embedding(x)  # (B, n_input_vars, d_model)

        # Add learned positional embedding
        x = x + self.position_embedding

        return self.dropout(x)


class EncoderLayer(nn.Module):
    """Pre-LayerNorm Transformer encoder layer with MultiheadAttention + GELU FFN."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        # Pre-LayerNorm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Multi-head self-attention
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Feed-forward network with GELU
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: (B, n_vars, d_model)
        Returns:
            (B, n_vars, d_model)
        """
        # Pre-norm self-attention
        residual = x
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = residual + self.dropout(attn_out)

        # Pre-norm FFN
        residual = x
        x_norm = self.norm2(x)
        ffn_out = self.ffn(x_norm)
        x = residual + ffn_out

        return x


class Encoder(nn.Module):
    """Stack of N encoder layers + final LayerNorm."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, n_layers: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        Args:
            x: (B, n_vars, d_model)
        Returns:
            (B, n_vars, d_model)
        """
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class iTransformer(nn.Module):
    """
    iTransformer for quantile vital sign forecasting (Vitals-Only Ablation).

    Input: (B, seq_len=72, n_input_vars=5)
        - indices 0-3: vital signs (ABPMean, PULSE, SpO2, RESP)
        - index 4: time feature

    Output: (B, pred_len=24, n_output_vars=4, n_quantiles=3)
        - Quantile predictions for the 4 vital signs
    """

    def __init__(
        self,
        seq_len: int = 72,
        pred_len: int = 24,
        n_vars: int = 4,           # 4 vitals (excluding time)
        n_input_vars: int = 5,     # 4 vitals + 1 time
        n_output_vars: int = 4,    # 4 vital signs to predict
        n_quantiles: int = 3,      # quantiles: 0.1, 0.5, 0.9
        d_model: int = 256,
        n_heads: int = 4,
        d_ff: int = 512,
        n_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_vars = n_vars
        self.n_input_vars = n_input_vars
        self.n_output_vars = n_output_vars
        self.n_quantiles = n_quantiles
        self.d_model = d_model

        # Indices of the vital sign variates in the input (0, 1, 2, 3)
        self.vital_indices = [0, 1, 2, 3]

        # Embedding: project each variate's history into d_model
        self.embedding = DataEmbedding_inverted(
            seq_len=seq_len,
            d_model=d_model,
            n_input_vars=n_input_vars,
            dropout=dropout,
        )

        # Transformer encoder (attention across variates)
        self.encoder = Encoder(
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            n_layers=n_layers,
            dropout=dropout,
        )

        # Output projection: for each vital sign token -> (pred_len * n_quantiles)
        self.output_projection = nn.Linear(d_model, pred_len * n_quantiles)

    def forward(self, x):
        """
        Args:
            x: (B, seq_len, n_input_vars) = (B, 72, 5)

        Returns:
            (B, pred_len, n_output_vars, n_quantiles) = (B, 24, 4, 3)
        """
        B = x.shape[0]

        # Embed: (B, 72, 5) -> (B, 5, d_model)
        enc_in = self.embedding(x)

        # Encode with transformer (attention across 5 variate tokens)
        enc_out = self.encoder(enc_in)  # (B, 5, d_model)

        # Extract vital sign tokens (indices 0, 1, 2, 3)
        vital_tokens = enc_out[:, self.vital_indices, :]  # (B, 4, d_model)

        # Project to predictions: (B, 4, d_model) -> (B, 4, pred_len * n_quantiles)
        out = self.output_projection(vital_tokens)  # (B, 4, pred_len * n_quantiles)

        # Reshape to (B, 4, pred_len, n_quantiles)
        out = out.view(B, self.n_output_vars, self.pred_len, self.n_quantiles)

        # Permute to (B, pred_len, n_output_vars, n_quantiles) = (B, 24, 4, 3)
        out = out.permute(0, 2, 1, 3)

        return out

    def count_parameters(self):
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(device='cuda'):
    """Build iTransformer with ablation config (vitals-only)."""
    model = iTransformer(
        seq_len=72,
        pred_len=24,
        n_vars=4,
        n_input_vars=5,
        n_output_vars=4,
        n_quantiles=3,
        d_model=256,
        n_heads=4,
        d_ff=512,
        n_layers=3,
        dropout=0.1,
    )
    model = model.to(device)
    print(f"iTransformer (vitals-only ablation) parameters: {model.count_parameters():,}")
    return model


if __name__ == "__main__":
    # Quick test
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = build_model(device)

    # Test forward pass
    x = torch.randn(4, 72, 5).to(device)
    out = model(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out.shape}")
    assert out.shape == (4, 24, 4, 3), f"Expected (4, 24, 4, 3), got {out.shape}"
    print("✓ Model test passed")
