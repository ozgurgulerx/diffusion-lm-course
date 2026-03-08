"""Full D3PM implementation: denoiser model, loss, and sampling.

Implements Structured Denoising Diffusion Models in Discrete State-Spaces
(Austin et al., 2021), Section 4.

The D3PM consists of:
1. A transformer-based denoiser that predicts x_0 from (x_t, t)
2. A variational lower bound (VLB) loss
3. A sampling procedure that runs the reverse chain from t=T to t=0
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class D3PMDenoiser(nn.Module):
    """Transformer-based denoiser for D3PM.

    Takes corrupted tokens x_t and timestep t, predicts a distribution
    over the original clean tokens x_0 at each position.

    Architecture:
    - Token embedding + positional encoding + timestep embedding
    - Stack of transformer encoder blocks
    - Linear output head projecting to vocab_size logits

    Args:
        vocab_size: Number of tokens in vocabulary.
        d_model: Hidden dimension of the transformer.
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

        # Token embedding
        self.token_emb = nn.Embedding(vocab_size, d_model)

        # Positional encoding (sinusoidal)
        self.pos_enc = nn.Embedding(max_seq_len, d_model)

        # Timestep embedding: maps scalar t to d_model-dimensional vector
        self.time_emb = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])

        # Final layer norm and output projection
        self.ln_final = nn.LayerNorm(d_model)
        self.output_head = nn.Linear(d_model, vocab_size)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small values for stable training."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _sinusoidal_embedding(self, t: torch.Tensor) -> torch.Tensor:
        """Compute sinusoidal embedding for timestep t.

        Args:
            t: Timestep tensor, shape (batch,).

        Returns:
            Embedding of shape (batch, d_model).
        """
        half_dim = self.d_model // 2
        emb = math.log(10000.0) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=torch.float) * -emb)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb

    def forward(
        self, x_t: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Predict x_0 distribution from corrupted x_t and timestep t.

        Args:
            x_t: Corrupted token IDs, shape (batch, seq_len).
            t: Timestep for each sample, shape (batch,), values in [1, T].

        Returns:
            Logits over vocab for each position, shape (batch, seq_len, vocab_size).
        """
        batch_size, seq_len = x_t.shape

        # Token embeddings
        h = self.token_emb(x_t)  # (batch, seq, d_model)

        # Add positional encoding
        positions = torch.arange(seq_len, device=x_t.device)
        h = h + self.pos_enc(positions)  # broadcast over batch

        # Add timestep embedding (broadcast over sequence)
        t_emb = self.time_emb(self._sinusoidal_embedding(t))  # (batch, d_model)
        h = h + t_emb.unsqueeze(1)  # (batch, seq, d_model)

        # Transformer blocks
        for block in self.blocks:
            h = block(h)

        # Output
        h = self.ln_final(h)
        logits = self.output_head(h)  # (batch, seq, vocab_size)

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


class D3PM:
    """Full D3PM model combining forward process, denoiser, and sampling.

    Args:
        denoiser: D3PMDenoiser network.
        vocab_size: Vocabulary size.
        num_timesteps: Number of diffusion timesteps T.
        schedule: Corruption schedule ("uniform" or "absorbing").
        mask_token_id: ID of the mask/absorbing token.
        hybrid_loss_coeff: Weight for the auxiliary loss on masked positions.
        device: Device to run on.
    """

    def __init__(
        self,
        denoiser: D3PMDenoiser,
        vocab_size: int,
        num_timesteps: int = 100,
        schedule: str = "absorbing",
        mask_token_id: int = 2,
        hybrid_loss_coeff: float = 0.01,
        device: torch.device = torch.device("cpu"),
    ):
        self.denoiser = denoiser
        self.vocab_size = vocab_size
        self.num_timesteps = num_timesteps
        self.schedule = schedule
        self.mask_token_id = mask_token_id
        self.hybrid_loss_coeff = hybrid_loss_coeff
        self.device = device

        # Precompute noise schedule
        self.betas = torch.linspace(1e-4, 0.02, num_timesteps).to(device)
        alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(alphas, dim=0)

    def _sample_q_t(self, x_0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Sample x_t ~ q(x_t | x_0). Vectorized.

        Args:
            x_0: Clean tokens, shape (batch, seq_len).
            t: Timesteps, shape (batch,), values in [1, T].

        Returns:
            Corrupted tokens x_t, shape (batch, seq_len).
        """
        batch_size, seq_len = x_0.shape
        alpha_bar = self.alpha_bars[(t - 1).long()].unsqueeze(1)  # (batch, 1)
        corrupt_mask = torch.rand(batch_size, seq_len, device=self.device) > alpha_bar

        if self.schedule == "absorbing":
            mask_tokens = torch.full_like(x_0, self.mask_token_id)
            x_t = torch.where(corrupt_mask, mask_tokens, x_0)
        elif self.schedule == "uniform":
            random_tokens = torch.randint(
                0, self.vocab_size, (batch_size, seq_len), device=self.device
            )
            x_t = torch.where(corrupt_mask, random_tokens, x_0)
        else:
            raise ValueError(f"Unknown schedule: {self.schedule}")

        return x_t

    def train_loss(self, x_0: torch.Tensor) -> torch.Tensor:
        """Compute D3PM training loss on a batch of clean sequences.

        Uses x_0-prediction cross-entropy loss, which is a simple and effective
        bound on the VLB (see D3PM Section 4). Optionally adds a weighted
        auxiliary loss on corrupted (masked) positions.

        Args:
            x_0: Clean token IDs, shape (batch, seq_len).

        Returns:
            Scalar loss tensor.
        """
        batch_size, seq_len = x_0.shape

        # Sample random timesteps
        t = torch.randint(1, self.num_timesteps + 1, (batch_size,), device=self.device)

        # Corrupt
        x_t = self._sample_q_t(x_0, t)

        # Predict x_0
        logits = self.denoiser(x_t, t)  # (batch, seq, vocab)

        # Cross-entropy loss: predict x_0 from x_t
        ce_loss = F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            x_0.reshape(-1).long(),
            reduction="mean",
        )

        # Optional: extra loss weight on masked positions
        if self.schedule == "absorbing" and self.hybrid_loss_coeff > 0:
            is_masked = (x_t == self.mask_token_id)
            if is_masked.any():
                masked_logits = logits[is_masked]
                masked_targets = x_0[is_masked]
                vlb_loss = F.cross_entropy(masked_logits, masked_targets.long())
                total_loss = ce_loss + self.hybrid_loss_coeff * vlb_loss
            else:
                total_loss = ce_loss
        else:
            total_loss = ce_loss

        return total_loss

    @torch.no_grad()
    def sample(
        self,
        batch_size: int = 1,
        seq_len: int = 64,
        temperature: float = 1.0,
        verbose: bool = False,
    ) -> torch.Tensor:
        """Generate samples by running the reverse diffusion chain.

        Starts from fully corrupted x_T and iteratively denoises to x_0.

        Args:
            batch_size: Number of sequences to generate.
            seq_len: Length of sequences to generate.
            temperature: Sampling temperature for x_0 prediction.
            verbose: Print intermediate steps.

        Returns:
            Generated token IDs, shape (batch_size, seq_len).
        """
        was_training = self.denoiser.training
        self.denoiser.train(False)

        # Start from fully corrupted state
        if self.schedule == "absorbing":
            x_t = torch.full(
                (batch_size, seq_len), self.mask_token_id,
                dtype=torch.long, device=self.device
            )
        else:
            x_t = torch.randint(
                0, self.vocab_size, (batch_size, seq_len), device=self.device
            )

        # Reverse chain: t = T, T-1, ..., 1
        for t_val in range(self.num_timesteps, 0, -1):
            t = torch.full((batch_size,), t_val, device=self.device)

            # Get model prediction for x_0
            logits = self.denoiser(x_t, t)

            # Apply temperature
            if temperature != 1.0:
                logits = logits / temperature

            x_0_probs = F.softmax(logits, dim=-1)

            if t_val == 1:
                # Last step: just take argmax
                x_t = logits.argmax(dim=-1)
            else:
                # Compute reverse posterior and sample
                x_t = self._sample_reverse_step(x_t, x_0_probs, t_val)

            if verbose and (t_val % max(1, self.num_timesteps // 10) == 0 or t_val <= 3):
                print(f"  t={t_val}: {x_t[0, :20].tolist()}")

        self.denoiser.train(was_training)
        return x_t

    def _sample_reverse_step(
        self,
        x_t: torch.Tensor,
        x_0_probs: torch.Tensor,
        t_val: int,
    ) -> torch.Tensor:
        """Take one reverse diffusion step.

        Args:
            x_t: Current state, shape (batch, seq_len).
            x_0_probs: Predicted x_0 distribution, shape (batch, seq_len, vocab).
            t_val: Current timestep value.

        Returns:
            Sampled x_{t-1}, shape (batch, seq_len).
        """
        batch_size, seq_len = x_t.shape
        K = self.vocab_size

        beta_t = self.betas[t_val - 1]
        alpha_bar_t = self.alpha_bars[t_val - 1]
        alpha_bar_t_prev = (
            self.alpha_bars[t_val - 2]
            if t_val > 1
            else torch.tensor(1.0, device=self.device)
        )

        if self.schedule == "absorbing":
            # Simplified posterior for absorbing schedule
            is_masked = (x_t == self.mask_token_id)

            # Probability of unmasking at this step
            unmask_prob = (alpha_bar_t_prev - alpha_bar_t) / (1 - alpha_bar_t + 1e-10)
            unmask_prob = unmask_prob.clamp(0, 1)

            should_unmask = torch.rand(batch_size, seq_len, device=self.device) < unmask_prob

            # Sample x_0 candidates from predicted distribution
            x_0_flat = x_0_probs.reshape(-1, K)
            x_0_sample = torch.multinomial(x_0_flat, num_samples=1).squeeze(-1)
            x_0_sample = x_0_sample.reshape(batch_size, seq_len)

            x_t_minus_1 = x_t.clone()
            unmask_these = is_masked & should_unmask
            x_t_minus_1[unmask_these] = x_0_sample[unmask_these]

        elif self.schedule == "uniform":
            # Full posterior computation for uniform schedule
            Qt = (1 - beta_t) * torch.eye(K, device=self.device) + (
                beta_t / K
            ) * torch.ones(K, K, device=self.device)

            I_K = torch.eye(K, device=self.device)
            ones_K = torch.ones(K, K, device=self.device) / K
            Qt_bar_prev = alpha_bar_t_prev * I_K + (1 - alpha_bar_t_prev) * ones_K

            # Likelihood: p(x_t | x_{t-1}=j)
            x_t_onehot = F.one_hot(x_t.long(), K).float()
            likelihood = x_t_onehot @ Qt  # (batch, seq, K)

            # Prior: sum_i p(x_0=i) * Qt_bar_prev[i, j]
            prior = x_0_probs @ Qt_bar_prev  # (batch, seq, K)

            # Posterior
            posterior = likelihood * prior
            posterior = posterior / (posterior.sum(dim=-1, keepdim=True) + 1e-10)

            posterior_flat = posterior.reshape(-1, K)
            x_t_minus_1 = torch.multinomial(posterior_flat, num_samples=1).squeeze(-1)
            x_t_minus_1 = x_t_minus_1.reshape(batch_size, seq_len)

        else:
            raise ValueError(f"Unknown schedule: {self.schedule}")

        return x_t_minus_1


def make_d3pm_loss_fn(d3pm: D3PM):
    """Create a loss function compatible with shared.utils.training.train_loop.

    Args:
        d3pm: D3PM instance.

    Returns:
        Callable(model, batch) -> loss tensor.
    """
    def loss_fn(model, batch):
        return d3pm.train_loss(batch)
    return loss_fn
