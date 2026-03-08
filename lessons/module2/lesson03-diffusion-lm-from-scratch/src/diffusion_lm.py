"""Diffusion-LM: Continuous diffusion language model.

Implements the core ideas from Li et al. (2022),
"Diffusion-LM Improves Controllable Text Generation."

Architecture:
  1. Embed discrete tokens into continuous space.
  2. Add Gaussian noise via the VP-SDE forward process.
  3. Train a transformer denoiser to predict the clean embedding (x_0-prediction).
  4. Generate by reverse SDE sampling, then round back to tokens.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiffusionLM(nn.Module):
    """Full Diffusion-LM model.

    Embeds tokens, trains with simple MSE denoising loss, and generates
    by iterative reverse-SDE sampling followed by nearest-neighbor rounding.

    Args:
        vocab_size: Vocabulary size.
        embed_dim: Embedding / model dimension.
        n_heads: Number of attention heads.
        n_layers: Number of transformer layers.
        seq_len: Maximum sequence length.
        beta_min: VP-SDE minimum noise rate.
        beta_max: VP-SDE maximum noise rate.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        n_heads: int = 4,
        n_layers: int = 4,
        seq_len: int = 64,
        beta_min: float = 0.1,
        beta_max: float = 20.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.beta_min = beta_min
        self.beta_max = beta_max

        # Token embedding (shared for embed and round-back)
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        nn.init.normal_(self.token_embedding.weight, std=1.0 / math.sqrt(embed_dim))

        # Positional encoding
        pe = torch.zeros(seq_len, embed_dim)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, seq_len, embed_dim)

        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

        # Input projection (noisy embedding -> model dimension)
        self.input_proj = nn.Linear(embed_dim, embed_dim)

        # Transformer layers
        self.layers = nn.ModuleList(
            [
                TransformerDenoiserBlock(embed_dim, n_heads, dropout=dropout)
                for _ in range(n_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(embed_dim)

        # Output projection (predicts clean embedding)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

    def embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Map token IDs to continuous embeddings.

        Args:
            token_ids: (batch, seq_len) integer tensor.

        Returns:
            Embeddings of shape (batch, seq_len, embed_dim).
        """
        return self.token_embedding(token_ids)

    def _time_embedding(self, t: torch.Tensor) -> torch.Tensor:
        """Sinusoidal time embedding.

        Args:
            t: (batch,) tensor with values in [0, 1].

        Returns:
            Time embedding of shape (batch, embed_dim).
        """
        half = self.embed_dim // 2
        freqs = torch.exp(
            torch.arange(half, device=t.device, dtype=torch.float)
            * (-math.log(10000.0) / half)
        )
        args = t.unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.time_mlp(emb)

    def _alpha_bar(self, t: torch.Tensor) -> torch.Tensor:
        """VP-SDE cumulative signal retention."""
        integral = self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * t ** 2
        return torch.exp(-0.5 * integral)

    def _forward_diffuse(
        self, x_0: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Add noise to clean embeddings via VP-SDE forward marginal.

        Args:
            x_0: Clean embeddings (batch, seq_len, embed_dim).
            t: Timestep (batch,) in [0, 1].

        Returns:
            (x_t, noise) tuple.
        """
        ab = self._alpha_bar(t).view(-1, 1, 1)  # (B, 1, 1)
        noise = torch.randn_like(x_0)
        x_t = torch.sqrt(ab) * x_0 + torch.sqrt(1.0 - ab) * noise
        return x_t, noise

    def denoise(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict clean embedding x_0 from noisy x_t and timestep t.

        Args:
            x_t: Noisy embeddings (batch, seq_len, embed_dim).
            t: Timestep (batch,) in [0, 1].

        Returns:
            Predicted clean embeddings (batch, seq_len, embed_dim).
        """
        B, S, D = x_t.shape
        # Project input
        h = self.input_proj(x_t) + self.pe[:, :S, :]

        # Add time conditioning
        t_emb = self._time_embedding(t)  # (B, D)
        h = h + t_emb.unsqueeze(1)  # broadcast over sequence

        # Transformer layers
        for layer in self.layers:
            h = layer(h)

        h = self.final_norm(h)
        return self.output_proj(h)

    def train_loss(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Compute the simplified MSE denoising loss.

        Samples a random timestep, adds noise, and computes MSE between
        the model's x_0-prediction and the actual clean embeddings.

        Args:
            token_ids: (batch, seq_len) integer tensor.

        Returns:
            Scalar loss tensor.
        """
        x_0 = self.embed(token_ids)  # (B, S, D)
        B = x_0.shape[0]

        # Sample random timesteps in (0, 1)
        t = torch.rand(B, device=x_0.device) * 0.999 + 0.001

        # Forward diffuse
        x_t, noise = self._forward_diffuse(x_0, t)

        # Predict clean embedding
        x_0_pred = self.denoise(x_t, t)

        # Simple MSE loss (x_0-prediction formulation)
        loss = F.mse_loss(x_0_pred, x_0)
        return loss

    @torch.no_grad()
    def sample(
        self,
        batch_size: int = 1,
        seq_len: Optional[int] = None,
        n_steps: int = 100,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Generate token sequences by reverse SDE sampling.

        Args:
            batch_size: Number of sequences to generate.
            seq_len: Sequence length (defaults to self.seq_len).
            n_steps: Number of reverse diffusion steps.
            device: Device to generate on.

        Returns:
            Token IDs of shape (batch_size, seq_len).
        """
        if seq_len is None:
            seq_len = self.seq_len
        if device is None:
            device = next(self.parameters()).device

        # Start from pure noise
        x = torch.randn(batch_size, seq_len, self.embed_dim, device=device)

        dt = 1.0 / n_steps
        for i in range(n_steps):
            t_val = 1.0 - i * dt
            t = torch.full((batch_size,), t_val, device=device)

            # Predict clean embedding
            x_0_pred = self.denoise(x, t)

            # Compute score from x_0-prediction:
            # score = (sqrt(ab) * x_0_pred - x_t) / (1 - ab)
            ab = self._alpha_bar(t).view(-1, 1, 1)
            score = (torch.sqrt(ab) * x_0_pred - x) / (1.0 - ab + 1e-8)

            # Reverse SDE step
            beta_t = (self.beta_min + t_val * (self.beta_max - self.beta_min))
            drift = -0.5 * beta_t * x - beta_t * score
            x = x + drift * (-dt)

            # Add stochastic noise (except at the last step)
            if i < n_steps - 1:
                noise = torch.randn_like(x)
                x = x + math.sqrt(beta_t * dt) * noise

        # Final denoising step
        t_final = torch.full((batch_size,), 0.001, device=device)
        x = self.denoise(x, t_final)

        # Round to nearest tokens
        return self.round_to_tokens(x)

    def round_to_tokens(self, continuous: torch.Tensor) -> torch.Tensor:
        """Map continuous embeddings to nearest token IDs.

        Args:
            continuous: (batch, seq_len, embed_dim) continuous vectors.

        Returns:
            Token IDs (batch, seq_len).
        """
        weight = self.token_embedding.weight  # (V, D)
        # Compute dot-product similarity (equivalent to min L2 for normalized)
        logits = torch.einsum("bsd,vd->bsv", continuous, weight)
        return logits.argmax(dim=-1)


class TransformerDenoiserBlock(nn.Module):
    """Pre-norm transformer block for the denoiser."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        x = x + self.ff(self.norm2(x))
        return x


def diffusion_lm_loss_fn(model: DiffusionLM, batch: torch.Tensor) -> torch.Tensor:
    """Loss function compatible with shared.utils.training.train_loop.

    Args:
        model: DiffusionLM instance.
        batch: Token IDs tensor of shape (batch, seq_len).

    Returns:
        Scalar loss.
    """
    return model.train_loss(batch)
