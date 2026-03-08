"""SEDD: Score Entropy Discrete Diffusion.

Reference: Lou et al. "Discrete Diffusion Modeling by Estimating the Ratios
of the Data Distribution" (2024). https://arxiv.org/abs/2310.16834

Key ideas:
  - Instead of predicting masked tokens directly, SEDD estimates the
    "concrete score": for each position i and each candidate token y,
    the score s_theta(x_t, t)_{i,y} approximates p(x_0^i = y | x_t) / p(x_0^i = x_t^i | x_t).
  - This is the discrete analog of the continuous score (grad log p).
  - The loss is "score entropy": a cross-entropy-like objective that trains
    the model to output correct probability ratios.
  - Sampling uses a Tweedie-type correction: from the estimated scores,
    we can reconstruct the reverse transition rates.

This implementation uses an absorbing (mask) diffusion process, matching
the SEDD-absorbing variant from the paper.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------

class AbsorbingSchedule:
    """Absorbing-state forward process schedule.

    Each token transitions to [MASK] at rate beta(t).  The survival
    probability (not yet masked) at time t is alpha(t) = exp(-int_0^t beta(s) ds).
    """

    def __init__(self, schedule_type: str = "cosine", s: float = 0.008):
        self.schedule_type = schedule_type
        self.s = s

    def alpha(self, t: torch.Tensor) -> torch.Tensor:
        """Survival probability: probability a token is NOT masked at time t."""
        if self.schedule_type == "cosine":
            f_t = torch.cos(((t + self.s) / (1 + self.s)) * (math.pi / 2)) ** 2
            f_0 = math.cos((self.s / (1 + self.s)) * (math.pi / 2)) ** 2
            return (f_t / f_0).clamp(1e-6, 1)
        elif self.schedule_type == "linear":
            return (1 - t).clamp(1e-6, 1)
        else:
            raise ValueError(f"Unknown schedule: {self.schedule_type}")

    def gamma(self, t: torch.Tensor) -> torch.Tensor:
        """Masking probability = 1 - alpha(t)."""
        return 1 - self.alpha(t)


# ---------------------------------------------------------------------------
# Transformer backbone adapted for score output
# ---------------------------------------------------------------------------

class ScoreTransformer(nn.Module):
    """Transformer that outputs concrete scores s(x_t, t) of shape (B, L, V).

    For the absorbing process, the score at a masked position i gives the
    ratio p(x_0^i = y) / p(x_0^i = MASK) for each token y.  At unmasked
    positions, the score is not used during sampling (those tokens are fixed).

    In practice, the network outputs raw logits; we interpret them as
    log-scores: log s_{i,y} = log (p(y) / p(mask)).  The score entropy loss
    trains these to be correct.
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
        # Output: log-scores for each vocabulary token
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
        """
        Args:
            x: (B, L) token IDs.
            t: (B,) timestep in [0, 1].

        Returns:
            (B, L, V) log-scores.
        """
        B, L = x.shape
        tok = self.token_emb(x)
        pos = self.pos_emb(torch.arange(L, device=x.device))
        time_emb = self.time_mlp(self.sinusoidal_embedding(t, self.d_model))
        h = tok + pos + time_emb[:, None, :]
        h = self.transformer(h)
        return self.out_proj(h)


# ---------------------------------------------------------------------------
# SEDD model
# ---------------------------------------------------------------------------

class SEDD(nn.Module):
    """Score Entropy Discrete Diffusion (absorbing variant).

    Args:
        vocab_size: Total vocabulary size including [MASK].
        mask_token_id: ID of [MASK] token.
        d_model: Hidden dimension.
        n_heads: Attention heads.
        n_layers: Transformer layers.
        max_seq_len: Max sequence length.
        schedule: Schedule type.
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
        self.schedule = AbsorbingSchedule(schedule)

        self.score_net = ScoreTransformer(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            max_seq_len=max_seq_len,
        )

    def forward_corrupt(
        self, x_0: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply absorbing forward process."""
        gamma_t = self.schedule.gamma(t)
        rand = torch.rand_like(x_0, dtype=torch.float)
        mask = rand < gamma_t[:, None]
        x_t = x_0.clone()
        x_t[mask] = self.mask_token_id
        return x_t, mask

    def score_entropy_loss(self, x_0: torch.Tensor) -> torch.Tensor:
        """Score entropy loss for SEDD.

        For the absorbing process, at masked positions, the true concrete
        score is:
            s*(x_t, t)_{i,y} = p(x_0^i=y | x_t) / p(x_0^i=MASK | x_t)

        For a single masked position with true token x_0^i, the score
        entropy loss is:
            L_i = sum_{y != mask} [ s_theta_{i,y} - log(s_theta_{i,y}) * 1(y == x_0^i) ]
              + [ exp(log_s_theta_{i, x_0^i}) - log_s_theta_{i, x_0^i} - 1 ]

        Simplified: the loss encourages s_theta to match the one-hot score
        (which equals the Bayes-optimal ratio).

        We compute:
            L_i = sum_{y != mask} exp(log_s_{i,y}) - log_s_{i, x_0^i}

        This is the "score entropy" from Eq. 9 in Lou et al. (2024).

        Args:
            x_0: (B, L) clean tokens.

        Returns:
            Scalar loss.
        """
        B, L = x_0.shape
        t = torch.rand(B, device=x_0.device).clamp(min=0.01, max=0.99)

        x_t, mask = self.forward_corrupt(x_0, t)
        log_scores = self.score_net(x_t, t)  # (B, L, V)

        if not mask.any():
            return torch.tensor(0.0, device=x_0.device, requires_grad=True)

        # Extract scores at masked positions
        log_s_masked = log_scores[mask]  # (N_masked, V)
        targets = x_0[mask]              # (N_masked,)

        # Score entropy: sum_y exp(log_s_y) - log_s_{x_0}
        # Exclude the mask token from the sum (it's the reference state)
        # Create a mask for non-mask-token entries in vocab
        vocab_mask = torch.ones(self.vocab_size, device=x_0.device, dtype=torch.bool)
        vocab_mask[self.mask_token_id] = False

        # sum_{y != mask} exp(log_s_y) -- the "entropy" term
        exp_sum = log_s_masked[:, vocab_mask].exp().sum(dim=-1)  # (N_masked,)

        # -log_s_{x_0^i} -- the "cross-entropy" term
        log_s_target = log_s_masked.gather(1, targets.unsqueeze(1)).squeeze(1)

        # Full loss per token
        per_token_loss = exp_sum - log_s_target

        return per_token_loss.mean()

    def train_loss(self, x_0: torch.Tensor) -> torch.Tensor:
        """Alias for score_entropy_loss, for API consistency."""
        return self.score_entropy_loss(x_0)

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        seq_len: int,
        num_steps: int = 50,
        device: str = "cpu",
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Sample via Tweedie-type reverse process.

        From the estimated scores s_theta, we reconstruct the denoising
        distribution:
            p_theta(x_0^i = y | x_t) proportional to s_theta_{i,y}

        Then we iteratively unmask positions from t=1 to t=0.

        Args:
            batch_size: Number of sequences.
            seq_len: Sequence length.
            num_steps: Denoising steps.
            device: Device.
            temperature: Sampling temperature.

        Returns:
            (batch_size, seq_len) generated tokens.
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
            n_masked = is_masked.float().sum(dim=-1)

            if gamma_now > 1e-6:
                frac = (gamma_now - gamma_next) / gamma_now
            else:
                frac = 1.0
            n_to_unmask = (n_masked * frac).clamp(min=1).long()

            t_batch = t_now.expand(batch_size)
            log_scores = self.score_net(x, t_batch)  # (B, L, V)

            # Convert scores to probabilities via softmax (Tweedie correction)
            # At masked positions, scores approximate p(x_0=y) / p(x_0=MASK)
            # So softmax over non-mask tokens gives approximate p(x_0=y | masked)
            probs = F.softmax(log_scores / temperature, dim=-1)

            # Zero out probability of generating [MASK] token
            probs[:, :, self.mask_token_id] = 0
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)

            sampled = torch.multinomial(
                probs.view(-1, self.vocab_size), num_samples=1
            ).view(batch_size, seq_len)

            # Confidence for unmasking order
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

        # Final cleanup
        remaining = (x == self.mask_token_id)
        if remaining.any():
            t_zero = torch.zeros(batch_size, device=device)
            log_scores = self.score_net(x, t_zero)
            probs = F.softmax(log_scores / temperature, dim=-1)
            probs[:, :, self.mask_token_id] = 0
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
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
    model = SEDD(vocab_size=V, mask_token_id=M, d_model=64, n_heads=2, n_layers=2)

    x = torch.randint(0, V - 1, (4, 16))
    loss = model.train_loss(x)
    print(f"SEDD score entropy loss: {loss.item():.4f}")

    samples = model.sample(batch_size=2, seq_len=16, num_steps=10)
    print(f"Samples shape: {samples.shape}")
    print(f"Samples:\n{samples}")
