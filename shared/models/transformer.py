"""Reusable transformer components for diffusion language models."""

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence positions."""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input tensor.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).
        """
        return x + self.pe[:, : x.size(1)]


class TimeEmbedding(nn.Module):
    """Sinusoidal time step embedding, used to condition on diffusion timestep.

    Maps scalar timestep t to a d_model-dimensional vector using sinusoidal
    encoding followed by a two-layer MLP.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Embed timestep into d_model dimensions.

        Args:
            t: Timestep tensor of shape (batch,) with values in [0, 1].

        Returns:
            Embedding of shape (batch, d_model).
        """
        half_dim = self.d_model // 2
        emb = math.log(10000.0) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=torch.float) * -emb)
        emb = t.unsqueeze(1) * emb.unsqueeze(0)  # (batch, half_dim)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)  # (batch, d_model)
        return self.mlp(emb)


class TransformerBlock(nn.Module):
    """Standard transformer encoder block with pre-norm architecture.

    Used as a building block for diffusion model denoisers.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int = 0, dropout: float = 0.1):
        """Initialize transformer block.

        Args:
            d_model: Model dimension.
            n_heads: Number of attention heads.
            d_ff: Feedforward hidden dimension. Defaults to 4 * d_model.
            dropout: Dropout rate.
        """
        super().__init__()
        d_ff = d_ff or d_model * 4

        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with pre-norm residual connections.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model).
        """
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        x = x + self.ff(self.norm2(x))
        return x
