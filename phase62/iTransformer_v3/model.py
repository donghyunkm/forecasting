"""
iTransformer for Phase 6.2 v3: Cluster Label Forecasting (Classification).

Architecture: Each variate's full lookback sequence is treated as a single token.
Attention is applied across variates (not across time steps).
Output is a classification head: (B, pred_len, num_classes) logits.

Input: (B, 48, 47) = 7 corr + 38 physio stats + 1 time + 1 label history
Output: (B, 12, 7) = logits for 7 cluster classes at each of 12 future steps
"""

import torch
import torch.nn as nn
import math


class DataEmbedding_inverted(nn.Module):
    """Projects each variate's full lookback (seq_len) into d_model."""

    def __init__(self, seq_len: int, d_model: int, n_input_vars: int, dropout: float = 0.1):
        super().__init__()
        self.value_embedding = nn.Linear(seq_len, d_model)
        self.position_embedding = nn.Parameter(torch.randn(1, n_input_vars, d_model) * 0.02)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B, n_input_vars, seq_len)
        x = self.value_embedding(x)  # (B, n_input_vars, d_model)
        x = x + self.position_embedding
        return self.dropout(x)


class EncoderLayer(nn.Module):
    """Pre-LayerNorm Transformer encoder layer."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads,
                                          dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = residual + self.dropout(attn_out)
        residual = x
        x_norm = self.norm2(x)
        x = residual + self.ffn(x_norm)
        return x


class Encoder(nn.Module):
    """Stack of N encoder layers + final LayerNorm."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, n_layers: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class iTransformer(nn.Module):
    """
    iTransformer for cluster label classification (Phase 6.2 v3).

    Input: (B, seq_len=48, n_input_vars=47)
    Output: (B, pred_len=12, num_classes=7) — logits for classification
    """

    def __init__(
        self,
        seq_len: int = 48,
        pred_len: int = 12,
        n_input_vars: int = 47,
        num_classes: int = 7,
        d_model: int = 256,
        n_heads: int = 4,
        d_ff: int = 512,
        n_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_input_vars = n_input_vars
        self.num_classes = num_classes
        self.d_model = d_model

        # Embedding
        self.embedding = DataEmbedding_inverted(
            seq_len=seq_len, d_model=d_model, n_input_vars=n_input_vars, dropout=dropout)

        # Encoder
        self.encoder = Encoder(d_model=d_model, n_heads=n_heads, d_ff=d_ff,
                               n_layers=n_layers, dropout=dropout)

        # Pool all variate tokens and project to classification output
        self.pool_proj = nn.Sequential(
            nn.Linear(d_model * n_input_vars, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, pred_len * num_classes),
        )

    def forward(self, x):
        """
        Args:
            x: (B, seq_len, n_input_vars) = (B, 48, 47)
        Returns:
            (B, pred_len, num_classes) = (B, 12, 7) logits
        """
        B = x.shape[0]

        # Embed: (B, 48, 47) -> (B, 47, d_model)
        enc_in = self.embedding(x)

        # Encode: attention across variate tokens
        enc_out = self.encoder(enc_in)  # (B, 47, d_model)

        # Flatten all variate tokens
        pooled = enc_out.reshape(B, -1)  # (B, 47 * d_model)

        # Project to classification logits
        out = self.pool_proj(pooled)  # (B, pred_len * num_classes)
        out = out.view(B, self.pred_len, self.num_classes)  # (B, 12, 7)

        return out

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(device='cuda'):
    """Build iTransformer with default config for Phase 6.2 v3."""
    model = iTransformer(
        seq_len=48,
        pred_len=12,
        n_input_vars=47,      # 7 corr + 38 physio + 1 time + 1 label history
        num_classes=7,
        d_model=256,
        n_heads=4,
        d_ff=512,
        n_layers=3,
        dropout=0.1,
    )
    model = model.to(device)
    print(f"iTransformer v3 parameters: {model.count_parameters():,}")
    return model


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = build_model(device)

    x = torch.randn(4, 48, 47).to(device)
    out = model(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out.shape}")
    assert out.shape == (4, 12, 7), f"Expected (4, 12, 7), got {out.shape}"
    print("✓ Model test passed")
