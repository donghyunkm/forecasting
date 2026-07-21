"""
iTransformer: Inverted Transformers Are Effective for Time Series Forecasting (ICLR 2024)

Key insight: Invert the transformer so that each VARIATE is a token (not each time step).
- Attention is applied ACROSS variates to capture multivariate correlations.
- FFN processes each variate token to learn temporal representations.
- A linear projection head maps variate tokens to forecasts.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DataEmbedding_inverted(nn.Module):
    """
    Embeds each variate's full lookback window into a d_model-dimensional token.
    
    Input: (batch, seq_len, n_vars)
    Output: (batch, n_vars, d_model)
    """

    def __init__(self, seq_len: int, d_model: int, n_vars: int, dropout: float = 0.1):
        super().__init__()
        self.seq_len = seq_len
        self.d_model = d_model
        self.n_vars = n_vars
        self.value_embedding = nn.Linear(seq_len, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, n_vars, d_model))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        self.dropout = nn.Dropout(p=dropout)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.value_embedding.weight)
        if self.value_embedding.bias is not None:
            nn.init.zeros_(self.value_embedding.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = self.value_embedding(x)
        x = x + self.pos_embedding
        x = self.dropout(x)
        return x


class EncoderLayer(nn.Module):
    """Standard transformer encoder layer with pre-LayerNorm."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model))
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        for module in self.ffn:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.norm1(x)
        attn_out, _ = self.attention(x_norm, x_norm, x_norm)
        x = x + self.dropout1(attn_out)
        x_norm = self.norm2(x)
        ffn_out = self.ffn(x_norm)
        x = x + self.dropout2(ffn_out)
        return x


class Encoder(nn.Module):
    """Stack of N EncoderLayers with a final LayerNorm."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, n_layers: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout)
            for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x


class iTransformer(nn.Module):
    """
    iTransformer for multivariate time series forecasting.
    
    Args:
        seq_len: Input sequence length (lookback window)
        pred_len: Prediction horizon length
        n_vars: Number of base variates (vitals)
        n_output_vars: Number of output variates to predict
        n_quantiles: Number of quantiles to predict per variate
        d_model: Transformer hidden dimension
        n_heads: Number of attention heads
        d_ff: Feed-forward hidden dimension
        n_layers: Number of encoder layers
        dropout: Dropout rate
        use_mask_input: Whether mask channels are included in input
    """

    def __init__(self, seq_len: int = 75, pred_len: int = 25, n_vars: int = 4,
                 n_output_vars: int = 4, n_quantiles: int = 3, d_model: int = 256,
                 n_heads: int = 4, d_ff: int = 512, n_layers: int = 3,
                 dropout: float = 0.1, use_mask_input: bool = False):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_vars = n_vars
        self.n_output_vars = n_output_vars
        self.n_quantiles = n_quantiles
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.n_layers = n_layers
        self.dropout = dropout
        self.use_mask_input = use_mask_input

        if use_mask_input:
            self.n_input_vars = n_vars * 2 + 1
        else:
            self.n_input_vars = n_vars + 1

        self.embedding = DataEmbedding_inverted(
            seq_len=seq_len, d_model=d_model, n_vars=self.n_input_vars, dropout=dropout)
        self.encoder = Encoder(
            d_model=d_model, n_heads=n_heads, d_ff=d_ff, n_layers=n_layers, dropout=dropout)
        self.projection = nn.Linear(d_model, pred_len * n_quantiles)
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.projection.weight)
        if self.projection.bias is not None:
            nn.init.zeros_(self.projection.bias)

    def forward(self, batch: dict) -> dict:
        x = batch['historical_ts_numeric']
        enc_input = self.embedding(x)
        enc_output = self.encoder(enc_input)
        vital_tokens = enc_output[:, :self.n_output_vars, :]
        projected = self.projection(vital_tokens)
        B = projected.shape[0]
        projected = projected.view(B, self.n_output_vars, self.pred_len, self.n_quantiles)
        projected = projected.permute(0, 2, 1, 3)
        output = projected.reshape(B, self.pred_len, self.n_output_vars * self.n_quantiles)
        return {'predicted_quantiles': output}


def _test():
    """Quick sanity check."""
    device = torch.device('cpu')
    model = iTransformer(
        seq_len=75, pred_len=25, n_vars=4, n_output_vars=4, n_quantiles=3,
        d_model=256, n_heads=4, d_ff=512, n_layers=3, dropout=0.1,
        use_mask_input=False).to(device)
    batch = {
        'historical_ts_numeric': torch.randn(8, 75, 5, device=device),
        'future_ts_numeric': torch.randn(8, 25, 1, device=device),
        'static_feats_numeric': torch.randn(8, 1, device=device),
    }
    output = model(batch)
    pred = output['predicted_quantiles']
    print(f"Input shape:  {batch['historical_ts_numeric'].shape}")
    print(f"Output shape: {pred.shape}")
    assert pred.shape == (8, 25, 12), f"Expected (8, 25, 12), got {pred.shape}"
    print("iTransformer test passed (use_mask_input=False, n_input_vars=5)!")
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {n_params:,}")


if __name__ == '__main__':
    _test()
