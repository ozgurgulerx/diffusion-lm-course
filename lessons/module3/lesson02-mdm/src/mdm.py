"""MDM: Masked Diffusion Model with continuous-time formulation.

Reference: Shi et al. "Simplified and Generalized Masked Diffusion for
Discrete Data" (2024). https://arxiv.org/abs/2406.04329

Key ideas:
  - The forward process is a continuous-time Markov chain (CTMC) where each
    token independently transitions to [MASK] at rate beta(t).
  - The masking probability at time t is gamma(t) = 1 - exp(-integral_0^t beta(s) ds).
  - The reverse process unmasks tokens.  The model predicts p(x_0 | x_t) and
    the reverse rate is derived from Bayes' rule.
  - Training loss simplifies to: E_t E_{x_0, x_t}[ -sum_{masked positions}
    log p_theta(x_0^i | x_t, t) * weight(t) ].
  - With the right weight, this is equivalent to an ELBO on log-likelihood.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Masking schedule for continuous-time CTMC
# ---------------------------------------------------------------------------

class MaskingSchedule:
    """Continuous-time masking schedule.

    Defines gamma(t) = 1 - exp(-integral_0^t beta(s) ds), the probability
    that a token is masked by time t.  We also need gamma'(t) for the
    continuous-time loss weighting.
    """

    def __init__(self, schedule_type: str = "cosine", s: float = 0.008):
        self.schedule_type = schedule_type
        self.s = s

    def gamma(self, t: torch.Tensor) -> torch.Tensor:
        """Masking probability at time t in [0, 1]."""
        if self.schedule_type == "cosine":
            f_t = torch.cos(((t + self.s) / (1 + self.s)) * (math.pi / 2)) ** 2
            f_0 = math.cos((self.s / (1 + self.s)) * (math.pi / 2)) ** 2
            return (1 - f_t / f_0).clamp(0, 1)
        elif self.schedule_type == "linear":
            return t.clamp(0, 1)
        else:
            raise ValueError(f"Unknown schedule: {self.schedule_type}")

    def gamma_prime(self, t: torch.Tensor) -> torch.Tensor:
        """d/dt gamma(t), computed via finite differences for robustness."""
        eps = 1e-4
        return (self.gamma(t + eps) - self.gamma(t - eps)) / (2 * eps)


# ---------------------------------------------------------------------------
# Transformer backbone (same architecture as lesson 01)
# ---------------------------------------------------------------------------

class TransformerDenoiser(nn.Module):
    """Transformer that maps (x_t, t) -> logits over vocabulary."""

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
        self.d_model = d_model
        self.vocab_size = vocab_size

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
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

    @staticmethod
    def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device).float() / half
        )
        args = t[:, None].float() * freqs[None, :]
        return torch.cat([args.sin(), args.cos()], dim=-1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        B, L = x.shape
        tok = self.token_emb(x)
        pos = self.pos_emb(torch.arange(L, device=x.device))
        time_emb = self.time_mlp(self.sinusoidal_embedding(t, self.d_model))
        h = tok + pos + time_emb[:, None, :]
        h = self.transformer(h)
        return self.out_proj(h)


# ---------------------------------------------------------------------------
# MDM: Masked Diffusion Model
# ---------------------------------------------------------------------------

class MDM(nn.Module):
    """Masked Diffusion Model (continuous-time formulation).

    Args:
        vocab_size: Vocabulary size (including [MASK] token).
        mask_token_id: ID of the [MASK] token.
        d_model: Transformer hidden dim.
        n_heads: Number of attention heads.
        n_layers: Number of Transformer layers.
        max_seq_len: Maximum sequence length.
        schedule: Masking schedule type ("cosine" or "linear").
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
        self.schedule = MaskingSchedule(schedule)

        self.denoiser = TransformerDenoiser(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            max_seq_len=max_seq_len,
        )

    def forward_corrupt(
        self, x_0: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Mask tokens according to the CTMC forward process at time t.

        Args:
            x_0: (B, L) clean token IDs.
            t: (B,) times in [0, 1].

        Returns:
            x_t: (B, L) corrupted sequences.
            mask: (B, L) boolean mask (True = masked).
        """
        gamma_t = self.schedule.gamma(t)  # (B,)
        rand = torch.rand_like(x_0, dtype=torch.float)
        mask = rand < gamma_t[:, None]
        x_t = x_0.clone()
        x_t[mask] = self.mask_token_id
        return x_t, mask

    def train_loss(self, x_0: torch.Tensor) -> torch.Tensor:
        """Continuous-time ELBO loss for MDM.

        The simplified loss from Shi et al. is:
            L = E_t[ (gamma'(t) / gamma(t)) * E_{x_t|x_0}[
                    sum_{masked i} -log p_theta(x_0^i | x_t, t)
                ] ]

        In practice, we sample t ~ U(0,1), corrupt, predict, and weight
        the per-position CE loss by gamma'(t) / gamma(t).

        Args:
            x_0: (B, L) clean token IDs.

        Returns:
            Scalar loss.
        """
        B, L = x_0.shape
        # Sample t, avoiding t=0 to prevent division by zero
        t = torch.rand(B, device=x_0.device).clamp(min=0.01, max=0.99)

        x_t, mask = self.forward_corrupt(x_0, t)
        logits = self.denoiser(x_t, t)  # (B, L, V)

        # Cross-entropy on masked positions
        logits_flat = logits[mask]  # (N_masked, V)
        targets_flat = x_0[mask]   # (N_masked,)

        if logits_flat.numel() == 0:
            return torch.tensor(0.0, device=x_0.device, requires_grad=True)

        per_token_loss = F.cross_entropy(logits_flat, targets_flat, reduction="none")

        # Weight by gamma'(t) / gamma(t) for each example
        gamma_t = self.schedule.gamma(t)          # (B,)
        gamma_prime_t = self.schedule.gamma_prime(t)  # (B,)
        weight = (gamma_prime_t / gamma_t.clamp(min=1e-6))  # (B,)

        # Expand weight to match each masked token's batch membership
        # mask is (B, L) — we need to know which batch element each masked token belongs to
        batch_indices = torch.arange(B, device=x_0.device)[:, None].expand_as(mask)
        batch_per_masked = batch_indices[mask]  # (N_masked,)
        weight_per_token = weight[batch_per_masked]  # (N_masked,)

        loss = (per_token_loss * weight_per_token).mean()
        return loss

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        seq_len: int,
        num_steps: int = 50,
        device: str = "cpu",
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Generate sequences via iterative unmasking.

        At each step, predict tokens for masked positions, then unmask the
        most confident ones according to the schedule.

        Args:
            batch_size: Number of sequences.
            seq_len: Sequence length.
            num_steps: Number of denoising steps.
            device: Device.
            temperature: Softmax temperature.

        Returns:
            (batch_size, seq_len) generated token IDs.
        """
        x = torch.full(
            (batch_size, seq_len), self.mask_token_id, dtype=torch.long, device=device
        )

        timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device=device)

        for i in range(num_steps):
            t_now = timesteps[i]
            t_next = timesteps[i + 1]

            gamma_now = self.schedule.gamma(t_now.unsqueeze(0)).item()
            gamma_next = self.schedule.gamma(t_next.unsqueeze(0)).item()

            is_masked = (x == self.mask_token_id)
            n_masked = is_masked.float().sum(dim=-1)  # (B,)

            # Fraction of currently masked tokens to reveal
            if gamma_now > 1e-6:
                frac = (gamma_now - gamma_next) / gamma_now
            else:
                frac = 1.0
            n_to_unmask = (n_masked * frac).clamp(min=1).long()

            t_batch = t_now.expand(batch_size)
            logits = self.denoiser(x, t_batch)
            probs = F.softmax(logits / temperature, dim=-1)

            sampled = torch.multinomial(
                probs.view(-1, self.vocab_size), num_samples=1
            ).view(batch_size, seq_len)

            confidence = torch.gather(probs, 2, sampled.unsqueeze(-1)).squeeze(-1)
            confidence[~is_masked] = -1.0

            for b in range(batch_size):
                n = n_to_unmask[b].item()
                masked_pos = is_masked[b].nonzero(as_tuple=True)[0]
                if len(masked_pos) == 0:
                    continue
                n = min(n, len(masked_pos))
                _, topk = confidence[b, masked_pos].topk(n)
                x[b, masked_pos[topk]] = sampled[b, masked_pos[topk]]

        # Unmask remaining
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
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)
    V, M = 100, 99
    model = MDM(vocab_size=V, mask_token_id=M, d_model=64, n_heads=2, n_layers=2)

    x = torch.randint(0, V - 1, (4, 16))
    loss = model.train_loss(x)
    print(f"MDM training loss: {loss.item():.4f}")

    samples = model.sample(batch_size=2, seq_len=16, num_steps=10)
    print(f"Samples shape: {samples.shape}")
    print(f"Samples:\n{samples}")
