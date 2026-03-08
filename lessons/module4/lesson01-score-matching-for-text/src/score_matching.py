"""Score matching for text in continuous embedding space.

Score matching estimates the score function nabla_x log p(x) without requiring
the normalizing constant of the distribution. For text, we work in the
continuous embedding space (as in Diffusion-LM) and train with denoising score
matching (DSM).

Key insight: predicting the noise epsilon that was added to produce x_t is
mathematically equivalent to estimating the score. Specifically, if
    x_t = x_0 + sigma_t * epsilon,   epsilon ~ N(0, I)
then
    nabla_{x_t} log p(x_t | x_0) = -epsilon / sigma_t
so a model trained to predict epsilon implicitly learns the score (up to a
known scaling factor).

This module implements:
    - ContinuousScoreNet: a small Transformer that predicts the score (or
      equivalently, the noise) given noisy embeddings and a timestep.
    - Denoising score matching training loop.
    - Demonstration of the noise-prediction <-> score equivalence.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Noise schedule helpers
# ---------------------------------------------------------------------------

def linear_beta_schedule(num_timesteps: int, beta_min: float = 1e-4, beta_max: float = 0.02) -> torch.Tensor:
    """Linear variance schedule from beta_min to beta_max."""
    return torch.linspace(beta_min, beta_max, num_timesteps)


def compute_alpha_bars(betas: torch.Tensor) -> torch.Tensor:
    """Cumulative product of (1 - beta), giving the signal-retention factor."""
    return torch.cumprod(1.0 - betas, dim=0)


# ---------------------------------------------------------------------------
# Sinusoidal timestep embedding
# ---------------------------------------------------------------------------

class SinusoidalTimeEmbedding(nn.Module):
    """Maps scalar timestep t to a d-dimensional sinusoidal embedding."""

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (batch,) integer or float timesteps.
        Returns:
            (batch, d_model) embeddings.
        """
        half = self.d_model // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float) / half
        )
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


# ---------------------------------------------------------------------------
# Transformer block
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """Pre-norm Transformer encoder block."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
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
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        x = x + self.ff(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# ContinuousScoreNet
# ---------------------------------------------------------------------------

class ContinuousScoreNet(nn.Module):
    """Transformer-based score network for continuous text embeddings.

    Given noisy embeddings x_t and a timestep t, predicts either:
        - the noise epsilon that was added (noise prediction mode), or
        - the score nabla_{x_t} log p(x_t) directly (score prediction mode).

    These are equivalent: score = -epsilon / sigma_t.

    Architecture:
        1. Project input embeddings to d_model.
        2. Add sinusoidal timestep embedding (broadcast to all positions).
        3. Pass through a stack of Transformer encoder blocks.
        4. Linear projection to output (same dim as input embeddings).

    Args:
        embed_dim: Dimensionality of the input token embeddings.
        d_model: Internal Transformer width.
        n_heads: Number of attention heads.
        n_layers: Number of Transformer blocks.
        d_ff: Feed-forward hidden dimension.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(embed_dim, d_model)
        self.time_embed = SinusoidalTimeEmbedding(d_model)
        self.time_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)
        ])
        self.output_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, embed_dim)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_t: Noisy embeddings, shape (batch, seq_len, embed_dim).
            t: Timestep indices, shape (batch,).

        Returns:
            Predicted noise epsilon, shape (batch, seq_len, embed_dim).
        """
        h = self.input_proj(x_t)  # (B, L, d_model)
        t_emb = self.time_proj(self.time_embed(t))  # (B, d_model)
        h = h + t_emb.unsqueeze(1)  # broadcast time to all positions

        for block in self.blocks:
            h = block(h)

        h = self.output_norm(h)
        return self.output_proj(h)  # (B, L, embed_dim) -- predicted noise


# ---------------------------------------------------------------------------
# Denoising Score Matching trainer
# ---------------------------------------------------------------------------

class DenoisingScoreMatchingTrainer:
    """Trains a ContinuousScoreNet using denoising score matching.

    The training objective is:
        L_DSM = E_{t, x_0, epsilon} || model(x_t, t) - epsilon ||^2

    where x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon.

    This is equivalent to learning the score because:
        score(x_t) = -epsilon / sqrt(1 - alpha_bar_t)

    Args:
        model: The ContinuousScoreNet.
        num_timesteps: Number of diffusion timesteps.
        beta_min: Minimum beta for the linear schedule.
        beta_max: Maximum beta for the linear schedule.
        lr: Learning rate.
    """

    def __init__(
        self,
        model: ContinuousScoreNet,
        num_timesteps: int = 1000,
        beta_min: float = 1e-4,
        beta_max: float = 0.02,
        lr: float = 1e-4,
    ):
        self.model = model
        self.num_timesteps = num_timesteps

        betas = linear_beta_schedule(num_timesteps, beta_min, beta_max)
        self.alpha_bars = compute_alpha_bars(betas)  # (T,)

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    def to(self, device: torch.device) -> "DenoisingScoreMatchingTrainer":
        self.model.to(device)
        self.alpha_bars = self.alpha_bars.to(device)
        return self

    def add_noise(
        self, x_0: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward diffusion: add Gaussian noise to clean embeddings.

        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon

        Returns:
            (x_t, epsilon) -- the noisy input and the noise that was added.
        """
        alpha_bar_t = self.alpha_bars[t]  # (B,)
        # Reshape for broadcasting: (B, 1, 1) for (B, L, D)
        while alpha_bar_t.dim() < x_0.dim():
            alpha_bar_t = alpha_bar_t.unsqueeze(-1)

        epsilon = torch.randn_like(x_0)
        x_t = alpha_bar_t.sqrt() * x_0 + (1 - alpha_bar_t).sqrt() * epsilon
        return x_t, epsilon

    def train_step(self, x_0: torch.Tensor) -> float:
        """One training step of denoising score matching.

        Args:
            x_0: Clean embeddings, shape (batch, seq_len, embed_dim).

        Returns:
            Scalar loss value.
        """
        self.model.train()
        device = x_0.device
        batch_size = x_0.shape[0]

        # Sample random timesteps
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=device)

        # Forward diffusion
        x_t, epsilon = self.add_noise(x_0, t)

        # Predict noise
        epsilon_pred = self.model(x_t, t)

        # MSE loss on noise prediction = denoising score matching loss
        loss = F.mse_loss(epsilon_pred, epsilon)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def get_score(
        self, x_t: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Convert noise prediction to score estimate.

        score(x_t) = -epsilon_pred / sqrt(1 - alpha_bar_t)

        This demonstrates the key equivalence between noise prediction
        and score estimation.

        Args:
            x_t: Noisy embeddings.
            t: Timestep indices.

        Returns:
            Estimated score nabla_{x_t} log p(x_t).
        """
        self.model.eval()
        with torch.no_grad():
            epsilon_pred = self.model(x_t, t)

        alpha_bar_t = self.alpha_bars[t]
        while alpha_bar_t.dim() < x_t.dim():
            alpha_bar_t = alpha_bar_t.unsqueeze(-1)

        sigma_t = (1 - alpha_bar_t).sqrt()
        score = -epsilon_pred / sigma_t
        return score


# ---------------------------------------------------------------------------
# Discrete score (SEDD-style) -- conceptual sketch
# ---------------------------------------------------------------------------

def concrete_score_example(logits: torch.Tensor, x: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """Compute the concrete score for discrete tokens (SEDD concept).

    In the discrete setting, the "score" is not a gradient in embedding space
    but rather a ratio of transition probabilities. For a token x_i and a
    candidate replacement y != x_i, the concrete score is:

        s(x, y, t) = p(x_t = y | x_0) / p(x_t = x_i | x_0)

    which measures how likely a transition from x_i to y is under the forward
    corruption process.

    This function provides a simplified illustration for uniform corruption.

    Args:
        logits: Model output logits, shape (batch, seq_len, vocab_size).
        x: Current token IDs, shape (batch, seq_len).
        vocab_size: Size of vocabulary.

    Returns:
        Concrete score ratios, shape (batch, seq_len, vocab_size).
        Entry [b, i, v] is the score for transitioning position i to token v.
        The entry for the current token is set to 0 by convention.
    """
    # Under uniform corruption with rate beta:
    #   p(x_t = x_0 | x_0) = 1 - beta + beta/V
    #   p(x_t = y  | x_0) = beta/V   for y != x_0
    # So the concrete score ratio for y != x_i is:
    #   s = (beta/V) / (1 - beta + beta/V)
    # This is constant for all y != x_i under uniform corruption.
    # A learned model predicts these ratios from the corrupted input.

    probs = F.softmax(logits, dim=-1)  # (B, L, V)

    # Gather the probability assigned to the current token at each position
    x_expanded = x.unsqueeze(-1)  # (B, L, 1)
    p_current = probs.gather(-1, x_expanded)  # (B, L, 1)

    # Score ratio: p(y) / p(current) for each candidate y
    scores = probs / (p_current + 1e-8)

    # Zero out the self-transition (current token -> current token)
    scores.scatter_(-1, x_expanded, 0.0)

    return scores


# ---------------------------------------------------------------------------
# Demo / verification
# ---------------------------------------------------------------------------

def demo_score_noise_equivalence():
    """Demonstrate that noise prediction and score estimation are equivalent."""
    torch.manual_seed(42)
    device = torch.device("cpu")

    embed_dim = 64
    seq_len = 16
    batch_size = 4

    model = ContinuousScoreNet(embed_dim=embed_dim, d_model=128, n_heads=4, n_layers=2, d_ff=256)
    trainer = DenoisingScoreMatchingTrainer(model, num_timesteps=100)
    trainer.to(device)

    # Create fake clean embeddings
    x_0 = torch.randn(batch_size, seq_len, embed_dim, device=device)

    # Train for a few steps to get non-random predictions
    print("Training score network (10 steps)...")
    for step in range(10):
        loss = trainer.train_step(x_0)
        if step % 5 == 0:
            print(f"  Step {step}: loss = {loss:.4f}")

    # Demonstrate equivalence
    t = torch.tensor([50] * batch_size, device=device)
    x_t, epsilon_true = trainer.add_noise(x_0, t)

    model.eval()
    with torch.no_grad():
        epsilon_pred = model(x_t, t)

    # Score from noise prediction
    alpha_bar_t = trainer.alpha_bars[t[0]]
    sigma_t = (1 - alpha_bar_t).sqrt()
    score_from_noise = -epsilon_pred / sigma_t

    # Score directly from trainer
    score_direct = trainer.get_score(x_t, t)

    diff = (score_from_noise - score_direct).abs().max().item()
    print(f"\nMax difference between two score computation paths: {diff:.2e}")
    print("(Should be ~0, confirming equivalence)")

    # Show that the true score (using true noise) is -epsilon/sigma
    true_score = -epsilon_true / sigma_t
    pred_error = (score_direct - true_score).abs().mean().item()
    print(f"Mean score prediction error (untrained model): {pred_error:.4f}")
    print("(This would decrease with more training data and steps)")


if __name__ == "__main__":
    demo_score_noise_equivalence()
