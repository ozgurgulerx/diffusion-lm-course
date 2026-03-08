"""MLM as Diffusion: showing that masked language modeling IS a diffusion process.

Key insight: BERT-style MLM with a fixed masking rate is a *one-step* diffusion
model.  If we generalize to *variable* masking rates gamma(t) in [0, 1], then:

  - gamma(t) plays the role of the noise level,
  - masking = forward (corruption) process,
  - predicting original tokens = reverse (denoising) process.

Unlike D3PM, which uses full transition matrices Q_t of size (V x V), masked
diffusion uses a trivially simple forward process:

    q(x_t | x_0) = gamma(t) * one_hot(MASK) + (1 - gamma(t)) * one_hot(x_0)

Each token is independently masked with probability gamma(t).  The reverse
model predicts the original tokens at all masked positions.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Masking schedule: gamma(t) for t in [0, 1]
# ---------------------------------------------------------------------------

def cosine_masking_schedule(t: torch.Tensor, s: float = 0.008) -> torch.Tensor:
    """Cosine masking schedule.

    gamma(t) rises from ~0 at t=0 to ~1 at t=1, following a cosine curve.
    This mirrors the cosine noise schedule from Nichol & Dhariwal (2021).

    Args:
        t: Timestep values in [0, 1], any shape.
        s: Small offset to avoid gamma(0) = 0 exactly.

    Returns:
        gamma(t) values in [0, 1], same shape as t.
    """
    # f(t) = cos((t + s) / (1 + s) * pi/2)^2
    f_t = torch.cos(((t + s) / (1 + s)) * (math.pi / 2)) ** 2
    f_0 = math.cos((s / (1 + s)) * (math.pi / 2)) ** 2
    # gamma(t) = 1 - f(t)/f(0) -- starts at 0, ends near 1
    return (1 - f_t / f_0).clamp(0, 1)


def linear_masking_schedule(t: torch.Tensor) -> torch.Tensor:
    """Linear masking schedule: gamma(t) = t."""
    return t.clamp(0, 1)


# ---------------------------------------------------------------------------
# Simple Transformer backbone
# ---------------------------------------------------------------------------

class TransformerDenoiser(nn.Module):
    """Small Transformer that predicts original tokens from partially masked input.

    This is the neural network backbone shared by MLMDiffusion and later by MDM.
    It takes token IDs (some of which are [MASK]) and a scalar timestep, and
    outputs logits over the vocabulary for every position.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        max_seq_len: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        # Timestep embedding: project scalar t -> d_model via sinusoidal + linear
        self.time_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.out_proj = nn.Linear(d_model, vocab_size)

    # -- sinusoidal time embedding (borrowed from continuous diffusion) ----------
    @staticmethod
    def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
        """Create sinusoidal timestep embeddings.

        Args:
            t: (batch,) float tensor of timestep values in [0, 1].
            dim: Embedding dimension (must be even).

        Returns:
            (batch, dim) embeddings.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device).float() / half
        )
        args = t[:, None].float() * freqs[None, :]
        return torch.cat([args.sin(), args.cos()], dim=-1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len) token IDs (may contain MASK tokens).
            t: (batch,) timestep values in [0, 1].

        Returns:
            (batch, seq_len, vocab_size) logits.
        """
        B, L = x.shape
        tok = self.token_emb(x)  # (B, L, D)
        pos = self.pos_emb(torch.arange(L, device=x.device))  # (L, D)
        time_emb = self.time_mlp(
            self.sinusoidal_embedding(t, self.d_model)
        )  # (B, D)

        h = tok + pos + time_emb[:, None, :]  # (B, L, D)
        h = self.transformer(h)
        return self.out_proj(h)  # (B, L, V)


# ---------------------------------------------------------------------------
# MLMDiffusion: the full model
# ---------------------------------------------------------------------------

class MLMDiffusion(nn.Module):
    """Masked Language Modeling viewed as a discrete diffusion model.

    Forward process:
        q(x_t | x_0) — independently mask each token with probability gamma(t).

    Reverse process (denoising):
        The model predicts p(x_0 | x_t, t) at all positions.
        Loss = cross-entropy on masked positions only.

    Sampling:
        Start from fully masked sequence (t=1), iteratively unmask by
        decreasing t through T steps.

    Args:
        vocab_size: Number of tokens in the vocabulary.
        mask_token_id: The ID reserved for [MASK].
        d_model: Transformer hidden dimension.
        n_heads: Number of attention heads.
        n_layers: Number of Transformer layers.
        max_seq_len: Maximum sequence length.
        schedule: Masking schedule function name ("cosine" or "linear").
    """

    def __init__(
        self,
        vocab_size: int,
        mask_token_id: int,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        max_seq_len: int = 128,
        schedule: str = "cosine",
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.mask_token_id = mask_token_id

        self.denoiser = TransformerDenoiser(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            max_seq_len=max_seq_len,
        )

        if schedule == "cosine":
            self.gamma = cosine_masking_schedule
        elif schedule == "linear":
            self.gamma = linear_masking_schedule
        else:
            raise ValueError(f"Unknown schedule: {schedule}")

    # -- forward corruption ---------------------------------------------------
    def forward_corrupt(
        self, x_0: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply forward masking process.

        Args:
            x_0: (batch, seq_len) clean token IDs.
            t: (batch,) timestep values in [0, 1].

        Returns:
            x_t: (batch, seq_len) corrupted token IDs.
            mask: (batch, seq_len) bool — True where tokens were masked.
        """
        gamma_t = self.gamma(t)  # (batch,)
        # Each token is masked independently with probability gamma(t)
        rand = torch.rand_like(x_0, dtype=torch.float)
        mask = rand < gamma_t[:, None]  # (batch, seq_len)
        x_t = x_0.clone()
        x_t[mask] = self.mask_token_id
        return x_t, mask

    # -- training loss ---------------------------------------------------------
    def train_loss(self, x_0: torch.Tensor) -> torch.Tensor:
        """Compute the training loss (cross-entropy on masked positions).

        1. Sample t ~ Uniform(0, 1) for each example in the batch.
        2. Corrupt x_0 -> x_t using forward_corrupt.
        3. Predict p(x_0 | x_t, t) with the denoiser.
        4. Cross-entropy on masked positions only.

        Args:
            x_0: (batch, seq_len) clean token IDs.

        Returns:
            Scalar loss.
        """
        B, L = x_0.shape
        t = torch.rand(B, device=x_0.device)
        x_t, mask = self.forward_corrupt(x_0, t)

        logits = self.denoiser(x_t, t)  # (B, L, V)

        # Only compute loss on masked positions
        logits_masked = logits[mask]  # (N_masked, V)
        targets_masked = x_0[mask]  # (N_masked,)

        if logits_masked.numel() == 0:
            # Edge case: nothing was masked (t very close to 0)
            return torch.tensor(0.0, device=x_0.device, requires_grad=True)

        return F.cross_entropy(logits_masked, targets_masked)

    # -- sampling (iterative unmasking) ----------------------------------------
    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        seq_len: int,
        num_steps: int = 50,
        device: str = "cpu",
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Generate sequences by iterative unmasking.

        Starting from a fully masked sequence, we walk from t=1 down to t=0
        in `num_steps` steps.  At each step, we predict token probabilities
        and sample tokens for the currently masked positions.  We unmask a
        fraction of positions proportional to the change in gamma(t).

        Args:
            batch_size: Number of sequences to generate.
            seq_len: Length of each sequence.
            num_steps: Number of denoising steps.
            device: Device for tensors.
            temperature: Sampling temperature.

        Returns:
            (batch_size, seq_len) generated token IDs.
        """
        # Start fully masked
        x = torch.full(
            (batch_size, seq_len), self.mask_token_id, dtype=torch.long, device=device
        )

        timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device=device)

        for i in range(num_steps):
            t_now = timesteps[i]
            t_next = timesteps[i + 1]

            gamma_now = self.gamma(t_now.unsqueeze(0)).item()
            gamma_next = self.gamma(t_next.unsqueeze(0)).item()

            # Current mask: which positions are still [MASK]?
            is_masked = (x == self.mask_token_id)
            n_masked = is_masked.float().sum(dim=-1)  # (B,)

            # How many tokens to unmask this step?
            # We want the fraction of masked tokens to go from gamma_now to gamma_next
            if gamma_now > 0:
                frac_to_unmask = (gamma_now - gamma_next) / gamma_now
            else:
                frac_to_unmask = 1.0
            n_to_unmask = (n_masked * frac_to_unmask).clamp(min=1).long()  # (B,)

            # Get model predictions
            t_batch = t_now.expand(batch_size)
            logits = self.denoiser(x, t_batch)  # (B, L, V)
            probs = F.softmax(logits / temperature, dim=-1)  # (B, L, V)

            # Sample tokens from predicted distribution
            sampled = torch.multinomial(
                probs.view(-1, self.vocab_size), num_samples=1
            ).view(batch_size, seq_len)

            # Compute confidence (probability assigned to sampled token)
            confidence = torch.gather(probs, 2, sampled.unsqueeze(-1)).squeeze(-1)
            # Only consider currently masked positions
            confidence[~is_masked] = -1.0

            # Unmask the most confident positions
            for b in range(batch_size):
                n = n_to_unmask[b].item()
                masked_positions = is_masked[b].nonzero(as_tuple=True)[0]
                if len(masked_positions) == 0:
                    continue
                n = min(n, len(masked_positions))
                conf_at_masked = confidence[b, masked_positions]
                _, topk_idx = conf_at_masked.topk(n)
                positions_to_unmask = masked_positions[topk_idx]
                x[b, positions_to_unmask] = sampled[b, positions_to_unmask]

        # Final pass: unmask any remaining [MASK] tokens
        remaining = (x == self.mask_token_id)
        if remaining.any():
            t_zero = torch.zeros(batch_size, device=device)
            logits = self.denoiser(x, t_zero)
            probs = F.softmax(logits / temperature, dim=-1)
            sampled = torch.multinomial(
                probs.view(-1, self.vocab_size), num_samples=1
            ).view(batch_size, seq_len)
            x[remaining] = sampled[remaining]

        return x


# ---------------------------------------------------------------------------
# Demo / sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)
    vocab_size = 100
    mask_id = 99
    seq_len = 16
    batch_size = 4

    model = MLMDiffusion(
        vocab_size=vocab_size,
        mask_token_id=mask_id,
        d_model=64,
        n_heads=2,
        n_layers=2,
        max_seq_len=seq_len,
    )

    # Fake data
    x_0 = torch.randint(0, vocab_size - 1, (batch_size, seq_len))

    # Training step
    loss = model.train_loss(x_0)
    print(f"Training loss: {loss.item():.4f}")

    # Sampling
    samples = model.sample(batch_size=2, seq_len=seq_len, num_steps=10)
    print(f"Sample shape: {samples.shape}")
    print(f"Sample:\n{samples}")
