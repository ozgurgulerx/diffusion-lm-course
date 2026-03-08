"""CDCD: Continuous Diffusion for Categorical Data.

Implements key ideas from Dieleman et al. (2022),
"Continuous Diffusion for Categorical Data."

The main improvement over vanilla Diffusion-LM: CDCD adds an auxiliary loss
that encourages denoised predictions to land close to valid token embeddings,
not just anywhere in continuous space. This "categorical projection loss"
improves rounding accuracy at generation time.

The score interpolation idea: blend a learned continuous score with a
discrete-aware term that pulls predictions toward the nearest embedding.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CDCD(nn.Module):
    """Continuous Diffusion for Categorical Data.

    Extends the Diffusion-LM approach with:
    1. A categorical projection loss that penalizes distance to nearest embeddings.
    2. Score interpolation between continuous and discrete-aware scores.

    Args:
        vocab_size: Vocabulary size.
        embed_dim: Embedding dimension.
        n_heads: Number of attention heads.
        n_layers: Number of transformer layers.
        seq_len: Maximum sequence length.
        beta_min: VP-SDE min noise rate.
        beta_max: VP-SDE max noise rate.
        categorical_weight: Weight for the categorical projection loss.
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
        categorical_weight: float = 1.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.categorical_weight = categorical_weight

        # Token embedding
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
        self.register_buffer("pe", pe.unsqueeze(0))

        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

        # Denoiser
        self.input_proj = nn.Linear(embed_dim, embed_dim)
        self.layers = nn.ModuleList(
            [
                _TransformerBlock(embed_dim, n_heads, dropout=dropout)
                for _ in range(n_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

    def embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Map token IDs to continuous embeddings."""
        return self.token_embedding(token_ids)

    def _time_embedding(self, t: torch.Tensor) -> torch.Tensor:
        half = self.embed_dim // 2
        freqs = torch.exp(
            torch.arange(half, device=t.device, dtype=torch.float)
            * (-math.log(10000.0) / half)
        )
        args = t.unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.time_mlp(emb)

    def _alpha_bar(self, t: torch.Tensor) -> torch.Tensor:
        integral = self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * t ** 2
        return torch.exp(-0.5 * integral)

    def _forward_diffuse(
        self, x_0: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ab = self._alpha_bar(t).view(-1, 1, 1)
        noise = torch.randn_like(x_0)
        x_t = torch.sqrt(ab) * x_0 + torch.sqrt(1.0 - ab) * noise
        return x_t, noise

    def denoise(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict clean embedding x_0 from noisy x_t."""
        B, S, D = x_t.shape
        h = self.input_proj(x_t) + self.pe[:, :S, :]
        t_emb = self._time_embedding(t)
        h = h + t_emb.unsqueeze(1)
        for layer in self.layers:
            h = layer(h)
        h = self.final_norm(h)
        return self.output_proj(h)

    def categorical_projection_loss(
        self, x_0_pred: torch.Tensor, target_ids: torch.Tensor
    ) -> torch.Tensor:
        """Compute the categorical projection loss.

        This is the key CDCD contribution: the predicted clean embedding should
        produce high probability for the correct token when computing softmax
        similarity against the embedding table.

        Essentially cross-entropy: logits = x_0_pred @ E^T, target = true token ID.

        Args:
            x_0_pred: Predicted clean embeddings (batch, seq_len, embed_dim).
            target_ids: True token IDs (batch, seq_len).

        Returns:
            Scalar cross-entropy loss.
        """
        # Compute logits via dot product with embedding matrix
        weight = self.token_embedding.weight  # (V, D)
        logits = torch.einsum("bsd,vd->bsv", x_0_pred, weight)  # (B, S, V)

        # Cross-entropy loss
        logits_flat = logits.reshape(-1, self.vocab_size)
        target_flat = target_ids.reshape(-1)
        return F.cross_entropy(logits_flat, target_flat)

    def embedding_distance_loss(self, x_0_pred: torch.Tensor) -> torch.Tensor:
        """Penalize predictions that are far from any valid embedding.

        For each predicted vector, compute the minimum L2 distance to any
        token embedding. This encourages the model to output vectors that
        lie on or near the embedding manifold.

        Args:
            x_0_pred: Predicted clean embeddings (batch, seq_len, embed_dim).

        Returns:
            Scalar loss (mean min-distance).
        """
        weight = self.token_embedding.weight  # (V, D)
        # Dot product similarity
        dots = torch.einsum("bsd,vd->bsv", x_0_pred, weight)
        # Squared distances: ||x - e||^2 = ||x||^2 - 2*x.e + ||e||^2
        x_sq = (x_0_pred ** 2).sum(dim=-1, keepdim=True)  # (B, S, 1)
        e_sq = (weight ** 2).sum(dim=-1).view(1, 1, -1)  # (1, 1, V)
        dists_sq = x_sq - 2 * dots + e_sq  # (B, S, V)
        # Minimum distance per position
        min_dist = dists_sq.min(dim=-1).values  # (B, S)
        return min_dist.mean()

    def train_loss(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Compute the combined CDCD training loss.

        Loss = MSE(x_0_pred, x_0) + categorical_weight * CE(logits, tokens)

        The MSE term trains the denoiser. The categorical term ensures
        predictions are close to valid token embeddings, improving rounding.

        Args:
            token_ids: (batch, seq_len) integer tensor.

        Returns:
            Scalar combined loss.
        """
        x_0 = self.embed(token_ids)
        B = x_0.shape[0]

        # Sample random timesteps
        t = torch.rand(B, device=x_0.device) * 0.999 + 0.001

        # Forward diffuse
        x_t, noise = self._forward_diffuse(x_0, t)

        # Predict clean embedding
        x_0_pred = self.denoise(x_t, t)

        # MSE denoising loss
        mse_loss = F.mse_loss(x_0_pred, x_0)

        # Categorical projection loss
        cat_loss = self.categorical_projection_loss(x_0_pred, token_ids)

        return mse_loss + self.categorical_weight * cat_loss

    def round_to_tokens(self, continuous: torch.Tensor) -> torch.Tensor:
        """Map continuous embeddings to nearest token IDs."""
        weight = self.token_embedding.weight
        logits = torch.einsum("bsd,vd->bsv", continuous, weight)
        return logits.argmax(dim=-1)

    @torch.no_grad()
    def sample(
        self,
        batch_size: int = 1,
        seq_len: Optional[int] = None,
        n_steps: int = 100,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Generate token sequences by reverse SDE sampling."""
        if seq_len is None:
            seq_len = self.seq_len
        if device is None:
            device = next(self.parameters()).device

        x = torch.randn(batch_size, seq_len, self.embed_dim, device=device)
        dt = 1.0 / n_steps

        for i in range(n_steps):
            t_val = 1.0 - i * dt
            t = torch.full((batch_size,), t_val, device=device)

            x_0_pred = self.denoise(x, t)
            ab = self._alpha_bar(t).view(-1, 1, 1)
            score = (torch.sqrt(ab) * x_0_pred - x) / (1.0 - ab + 1e-8)

            beta_t = self.beta_min + t_val * (self.beta_max - self.beta_min)
            drift = -0.5 * beta_t * x - beta_t * score
            x = x + drift * (-dt)

            if i < n_steps - 1:
                noise = torch.randn_like(x)
                x = x + math.sqrt(beta_t * dt) * noise

        t_final = torch.full((batch_size,), 0.001, device=device)
        x = self.denoise(x, t_final)
        return self.round_to_tokens(x)


class _TransformerBlock(nn.Module):
    """Pre-norm transformer block."""

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


def cdcd_loss_fn(model: CDCD, batch: torch.Tensor) -> torch.Tensor:
    """Loss function compatible with shared.utils.training.train_loop."""
    return model.train_loss(batch)
