"""Masked Discrete Language Model (MDLM) implementation.

MDLM (Sahoo et al., 2024) simplifies discrete diffusion by:
1. Using only absorbing (mask) transitions — no uniform noise.
2. A continuous-time formulation with a simple masking schedule.
3. A simplified loss that directly trains on masked token prediction.

The key insight: if the only corruption is masking, then the model's job
is simply to predict the original token at each masked position — similar
to BERT, but with a diffusion-style schedule and generation procedure.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MDLMDenoiser(nn.Module):
    """Transformer denoiser for MDLM.

    Same architecture as D3PM denoiser but conceptually simpler:
    the input is a partially masked sequence and the output predicts
    the original tokens at all positions.

    Args:
        vocab_size: Number of tokens in vocabulary (including mask token).
        d_model: Hidden dimension.
        n_heads: Number of attention heads.
        n_layers: Number of transformer blocks.
        max_seq_len: Maximum sequence length.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        max_seq_len: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_enc = nn.Embedding(max_seq_len, d_model)
        self.time_emb = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])

        self.ln_final = nn.LayerNorm(d_model)
        self.output_head = nn.Linear(d_model, vocab_size)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _sinusoidal_embedding(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.d_model // 2
        emb = math.log(10000.0) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=torch.float) * -emb)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict original tokens from masked sequence.

        Args:
            x_t: Partially masked token IDs, shape (batch, seq_len).
            t: Continuous time values, shape (batch,), in [0, 1].

        Returns:
            Logits over vocabulary, shape (batch, seq_len, vocab_size).
        """
        batch_size, seq_len = x_t.shape

        h = self.token_emb(x_t)
        positions = torch.arange(seq_len, device=x_t.device)
        h = h + self.pos_enc(positions)

        t_emb = self.time_emb(self._sinusoidal_embedding(t))
        h = h + t_emb.unsqueeze(1)

        for block in self.blocks:
            h = block(h)

        h = self.ln_final(h)
        logits = self.output_head(h)
        return logits


class TransformerBlock(nn.Module):
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


class MDLM:
    """Masked Discrete Language Model.

    MDLM uses a continuous-time masking schedule where at time t in [0, 1]:
    - t=0 means fully clean (no masks)
    - t=1 means fully masked

    The masking rate at time t is given by a schedule function gamma(t).

    Args:
        denoiser: MDLMDenoiser network.
        vocab_size: Vocabulary size.
        mask_token_id: ID of the [MASK] token.
        num_timesteps: Number of discrete sampling steps (for generation).
        schedule_type: Masking schedule — "linear" or "cosine".
        device: Device to run on.
    """

    def __init__(
        self,
        denoiser: MDLMDenoiser,
        vocab_size: int,
        mask_token_id: int = 2,
        num_timesteps: int = 100,
        schedule_type: str = "cosine",
        device: torch.device = torch.device("cpu"),
    ):
        self.denoiser = denoiser
        self.vocab_size = vocab_size
        self.mask_token_id = mask_token_id
        self.num_timesteps = num_timesteps
        self.schedule_type = schedule_type
        self.device = device

    def masking_rate(self, t: torch.Tensor) -> torch.Tensor:
        """Compute the masking rate gamma(t) at continuous time t.

        gamma(t) is the probability that each token is masked at time t.
        - gamma(0) = 0 (no masking)
        - gamma(1) = 1 (fully masked)

        Args:
            t: Continuous time values in [0, 1], any shape.

        Returns:
            Masking rates, same shape as t.
        """
        if self.schedule_type == "linear":
            return t
        elif self.schedule_type == "cosine":
            # Cosine schedule: slower corruption at start, faster at end
            return 1.0 - torch.cos(t * math.pi / 2)
        else:
            raise ValueError(f"Unknown schedule: {self.schedule_type}")

    def mask_tokens(
        self, x_0: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Apply masking at continuous time t.

        Each token is independently masked with probability gamma(t).

        Args:
            x_0: Clean tokens, shape (batch, seq_len).
            t: Continuous time per sample, shape (batch,), in [0, 1].

        Returns:
            Masked tokens, shape (batch, seq_len).
        """
        batch_size, seq_len = x_0.shape
        gamma = self.masking_rate(t).unsqueeze(1)  # (batch, 1)

        # Each token is masked independently with probability gamma(t)
        mask = torch.rand(batch_size, seq_len, device=self.device) < gamma
        mask_tokens = torch.full_like(x_0, self.mask_token_id)
        x_t = torch.where(mask, mask_tokens, x_0)

        return x_t

    def train_loss(self, x_0: torch.Tensor) -> torch.Tensor:
        """Compute MDLM training loss.

        The loss is the cross-entropy of predicting the original tokens,
        weighted to focus on masked positions. This is the simplified ELBO
        from the MDLM paper.

        Args:
            x_0: Clean token IDs, shape (batch, seq_len).

        Returns:
            Scalar loss tensor.
        """
        batch_size, seq_len = x_0.shape

        # Sample random continuous time t ~ Uniform(0, 1)
        t = torch.rand(batch_size, device=self.device)
        # Clamp to avoid edge cases at t=0 and t=1
        t = t.clamp(1e-4, 1.0 - 1e-4)

        # Mask tokens according to the schedule
        x_t = self.mask_tokens(x_0, t)

        # Get model predictions
        logits = self.denoiser(x_t, t)  # (batch, seq_len, vocab_size)

        # Compute cross-entropy loss on masked positions only
        # This focuses the model on actually predicting masked content
        is_masked = (x_t == self.mask_token_id)

        if is_masked.any():
            masked_logits = logits[is_masked]  # (num_masked, vocab_size)
            masked_targets = x_0[is_masked]    # (num_masked,)
            loss = F.cross_entropy(masked_logits, masked_targets.long())
        else:
            # Edge case: no tokens were masked (very early time)
            # Fall back to full cross-entropy
            loss = F.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                x_0.reshape(-1).long(),
            )

        return loss

    @torch.no_grad()
    def sample(
        self,
        batch_size: int = 1,
        seq_len: int = 64,
        temperature: float = 1.0,
        verbose: bool = False,
    ) -> torch.Tensor:
        """Generate samples by iterative unmasking.

        Starts from fully masked and progressively unmasks tokens from
        t=1 to t=0 in discrete steps.

        Args:
            batch_size: Number of sequences to generate.
            seq_len: Length of sequences.
            temperature: Sampling temperature.
            verbose: Print intermediate steps.

        Returns:
            Generated token IDs, shape (batch_size, seq_len).
        """
        was_training = self.denoiser.training
        self.denoiser.train(False)

        # Start fully masked
        x_t = torch.full(
            (batch_size, seq_len), self.mask_token_id,
            dtype=torch.long, device=self.device
        )

        # Discrete time steps from t=1 to t=0
        time_steps = torch.linspace(1.0, 0.0, self.num_timesteps + 1, device=self.device)

        for i in range(self.num_timesteps):
            t_current = time_steps[i]
            t_next = time_steps[i + 1]

            gamma_current = self.masking_rate(t_current)
            gamma_next = self.masking_rate(t_next)

            # Fraction of currently-masked tokens to unmask in this step
            # We want to go from gamma_current to gamma_next
            if gamma_current > 0:
                unmask_frac = (gamma_current - gamma_next) / gamma_current
            else:
                unmask_frac = 0.0

            # Get model predictions
            t_batch = torch.full((batch_size,), t_current.item(), device=self.device)
            logits = self.denoiser(x_t, t_batch)

            if temperature != 1.0:
                logits = logits / temperature

            probs = F.softmax(logits, dim=-1)

            # For each masked position, decide whether to unmask
            is_masked = (x_t == self.mask_token_id)

            if is_masked.any() and unmask_frac > 0:
                # Sample which masked positions to unmask
                unmask_decisions = torch.rand_like(is_masked.float()) < unmask_frac
                to_unmask = is_masked & unmask_decisions

                # Sample tokens for positions being unmasked
                probs_flat = probs.reshape(-1, self.vocab_size)
                sampled = torch.multinomial(probs_flat, num_samples=1).squeeze(-1)
                sampled = sampled.reshape(batch_size, seq_len)

                x_t = x_t.clone()
                x_t[to_unmask] = sampled[to_unmask]

            if verbose and (i % max(1, self.num_timesteps // 8) == 0 or i == self.num_timesteps - 1):
                n_masked = (x_t == self.mask_token_id).sum().item()
                print(f"  Step {i+1}/{self.num_timesteps} (t={t_current:.3f}): {n_masked} masked")

        # Final pass: unmask any remaining masked tokens
        remaining_masked = (x_t == self.mask_token_id)
        if remaining_masked.any():
            t_batch = torch.full((batch_size,), 1e-4, device=self.device)
            logits = self.denoiser(x_t, t_batch)
            if temperature != 1.0:
                logits = logits / temperature
            final_tokens = logits.argmax(dim=-1)
            x_t[remaining_masked] = final_tokens[remaining_masked]

        self.denoiser.train(was_training)
        return x_t


def make_mdlm_loss_fn(mdlm: MDLM):
    """Create a loss function compatible with shared.utils.training.train_loop.

    Args:
        mdlm: MDLM instance.

    Returns:
        Callable(model, batch) -> loss tensor.
    """
    def loss_fn(model, batch):
        return mdlm.train_loss(batch)
    return loss_fn
