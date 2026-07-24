"""
iTransformer for Phase 6.1: Correlation Forecasting with Physiological Features.

Architecture: Each variate's full lookback sequence is treated as a single token.
Attention is applied across variates (not across time steps).
Correlation tokens (indices 0-6) project to point predictions.
Physio stats and time tokens participate in attention but have no output head.

Input: (B, 48, 46) = 7 correlations + 38 physio stats + 1 time position
Output: (B, 12, 7) = 7 correlation forecasts (single value each)
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
    iTransformer for point-prediction correlation forecasting (Phase 6.1).

    Input: (B, seq_len=48, n_input_vars=46)
        - indices 0-6: 7 correlation features
        - indices 7-44: 38 physiological stats (19 mean + 19 std)
        - index 45: time feature

    Output: (B, pred_len=12, n_output_vars=7)
        - Point predictions for all 7 correlations
    """

    def __init__(
        self,
        seq_len: int = 48,
        pred_len: int = 12,
        n_vars: int = 7,           # 7 correlations (output variates)
        n_input_vars: int = 46,    # 7 correlations + 38 physio stats + 1 time
        n_output_vars: int = 7,    # 7 correlations to predict
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
        self.d_model = d_model

        # Indices of the correlation variates in the input (0, 1, 2, 3, 4, 5, 6)
        self.corr_indices = list(range(n_output_vars))

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

        # Output projection: for each correlation token -> pred_len (single value per step)
        self.output_projection = nn.Linear(d_model, pred_len)

    def forward(self, x):
        """
        Args:
            x: (B, seq_len, n_input_vars) = (B, 48, 46)

        Returns:
            (B, pred_len, n_output_vars) = (B, 12, 7)
        """
        B = x.shape[0]

        # Embed: (B, 48, 46) -> (B, 46, d_model)
        enc_in = self.embedding(x)

        # Encode with transformer (attention across 46 variate tokens)
        enc_out = self.encoder(enc_in)  # (B, 46, d_model)

        # Extract correlation tokens (indices 0-6)
        corr_tokens = enc_out[:, self.corr_indices, :]  # (B, 7, d_model)

        # Project to predictions: (B, 7, d_model) -> (B, 7, pred_len)
        out = self.output_projection(corr_tokens)  # (B, 7, pred_len)

        # Permute to (B, pred_len, n_output_vars) = (B, 12, 7)
        out = out.permute(0, 2, 1)

        return out

    def count_parameters(self):
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(device='cuda'):
    """Build iTransformer with default config for Phase 6.1."""
    model = iTransformer(
        seq_len=48,
        pred_len=12,
        n_vars=7,
        n_input_vars=46,
        n_output_vars=7,
        d_model=256,
        n_heads=4,
        d_ff=512,
        n_layers=3,
        dropout=0.1,
    )
    model = model.to(device)
    print(f"iTransformer parameters: {model.count_parameters():,}")
    return model


if __name__ == "__main__":
    # Quick test
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = build_model(device)

    # Test forward pass
    x = torch.randn(4, 48, 46).to(device)
    out = model(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out.shape}")
    assert out.shape == (4, 12, 7), f"Expected (4, 12, 7), got {out.shape}"
    print("✓ Model test passed")
