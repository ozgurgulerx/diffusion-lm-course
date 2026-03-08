"""Side-by-side comparison: MDM vs D3PM.

Trains both models on the same synthetic/small text dataset and compares:
  - Training loss convergence speed
  - Sample quality (perplexity under a reference model)
  - Generation diversity (distinct n-grams)

D3PM uses a uniform transition matrix (each token can become any other),
while MDM uses the simpler absorbing (mask-only) corruption.
"""

import math
import time
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Shared Transformer backbone
# ---------------------------------------------------------------------------

class TransformerBackbone(nn.Module):
    """Shared Transformer backbone for both MDM and D3PM."""

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
# MDM (simplified version for comparison)
# ---------------------------------------------------------------------------

class MDMComparison(nn.Module):
    """Minimal MDM for fair comparison with D3PM."""

    def __init__(self, vocab_size: int, mask_token_id: int, **kwargs):
        super().__init__()
        self.vocab_size = vocab_size
        self.mask_token_id = mask_token_id
        self.net = TransformerBackbone(vocab_size=vocab_size, **kwargs)

    def train_loss(self, x_0: torch.Tensor) -> torch.Tensor:
        B, L = x_0.shape
        t = torch.rand(B, device=x_0.device).clamp(0.01, 0.99)

        # Cosine masking schedule
        s = 0.008
        f_t = torch.cos(((t + s) / (1 + s)) * (math.pi / 2)) ** 2
        f_0 = math.cos((s / (1 + s)) * (math.pi / 2)) ** 2
        gamma = (1 - f_t / f_0).clamp(0, 1)

        rand = torch.rand_like(x_0, dtype=torch.float)
        mask = rand < gamma[:, None]
        x_t = x_0.clone()
        x_t[mask] = self.mask_token_id

        logits = self.net(x_t, t)
        if not mask.any():
            return torch.tensor(0.0, device=x_0.device, requires_grad=True)

        return F.cross_entropy(logits[mask], x_0[mask])

    @torch.no_grad()
    def sample(self, batch_size: int, seq_len: int, num_steps: int = 50,
               device: str = "cpu") -> torch.Tensor:
        x = torch.full((batch_size, seq_len), self.mask_token_id,
                        dtype=torch.long, device=device)
        for i in range(num_steps):
            t_val = 1.0 - i / num_steps
            t = torch.full((batch_size,), t_val, device=device)
            logits = self.net(x, t)
            probs = F.softmax(logits, dim=-1)
            sampled = torch.multinomial(probs.view(-1, self.vocab_size), 1).view(batch_size, seq_len)

            is_masked = (x == self.mask_token_id)
            n_masked = is_masked.float().sum(dim=-1)
            n_to_reveal = (n_masked / (num_steps - i)).clamp(min=1).long()

            confidence = torch.gather(probs, 2, sampled.unsqueeze(-1)).squeeze(-1)
            confidence[~is_masked] = -1.0
            for b in range(batch_size):
                mp = is_masked[b].nonzero(as_tuple=True)[0]
                if len(mp) == 0:
                    continue
                n = min(n_to_reveal[b].item(), len(mp))
                _, topk = confidence[b, mp].topk(n)
                x[b, mp[topk]] = sampled[b, mp[topk]]

        remaining = (x == self.mask_token_id)
        if remaining.any():
            t = torch.zeros(batch_size, device=device)
            logits = self.net(x, t)
            s = torch.multinomial(F.softmax(logits, -1).view(-1, self.vocab_size), 1).view(batch_size, seq_len)
            x[remaining] = s[remaining]
        return x


# ---------------------------------------------------------------------------
# D3PM (uniform transition, simplified)
# ---------------------------------------------------------------------------

class D3PMComparison(nn.Module):
    """Simplified D3PM with uniform transition matrix for comparison.

    Forward process: q(x_t | x_{t-1}) = (1 - beta_t) * I + beta_t * (1/V) * 11^T
    Each token stays with probability (1 - beta_t) or becomes any random
    token with probability beta_t.
    """

    def __init__(self, vocab_size: int, num_timesteps: int = 100, **kwargs):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_timesteps = num_timesteps
        self.net = TransformerBackbone(vocab_size=vocab_size, **kwargs)

        # Linear beta schedule
        betas = torch.linspace(1e-4, 0.02, num_timesteps)
        # Cumulative product of (1 - beta) gives probability of staying original
        alphas = 1 - betas
        alpha_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("alpha_cumprod", alpha_cumprod)

    def forward_corrupt(self, x_0: torch.Tensor, t_idx: torch.Tensor
                        ) -> torch.Tensor:
        """Apply uniform corruption at discrete timestep t_idx.

        Each token stays as x_0 with probability alpha_bar_t,
        or becomes a random token with probability (1 - alpha_bar_t).
        """
        alpha_bar = self.alpha_cumprod[t_idx]  # (B,)
        rand = torch.rand_like(x_0, dtype=torch.float)
        corrupt_mask = rand > alpha_bar[:, None]
        random_tokens = torch.randint_like(x_0, 0, self.vocab_size)
        x_t = torch.where(corrupt_mask, random_tokens, x_0)
        return x_t

    def train_loss(self, x_0: torch.Tensor) -> torch.Tensor:
        B, L = x_0.shape
        # Sample discrete timestep
        t_idx = torch.randint(0, self.num_timesteps, (B,), device=x_0.device)
        t_continuous = t_idx.float() / self.num_timesteps  # for the network

        x_t = self.forward_corrupt(x_0, t_idx)
        logits = self.net(x_t, t_continuous)  # (B, L, V)

        # Cross-entropy: predict x_0 from x_t (x_0-parameterization)
        return F.cross_entropy(logits.view(-1, self.vocab_size), x_0.view(-1))

    @torch.no_grad()
    def sample(self, batch_size: int, seq_len: int, num_steps: int = 50,
               device: str = "cpu") -> torch.Tensor:
        """Ancestral sampling from t=T to t=0."""
        # Start from uniform random
        x = torch.randint(0, self.vocab_size, (batch_size, seq_len), device=device)

        step_size = max(1, self.num_timesteps // num_steps)
        timesteps = list(range(self.num_timesteps - 1, -1, -step_size))

        for t_idx_val in timesteps:
            t_idx = torch.full((batch_size,), t_idx_val, device=device, dtype=torch.long)
            t_cont = t_idx.float() / self.num_timesteps

            logits = self.net(x, t_cont)
            # Predict x_0
            x_0_pred = logits.argmax(dim=-1)

            # Re-corrupt to t-1 if not at step 0
            if t_idx_val > 0:
                t_prev = torch.full((batch_size,), t_idx_val - step_size,
                                     device=device, dtype=torch.long).clamp(min=0)
                alpha_bar_prev = self.alpha_cumprod[t_prev]
                rand = torch.rand_like(x, dtype=torch.float)
                corrupt_mask = rand > alpha_bar_prev[:, None]
                random_tokens = torch.randint_like(x, 0, self.vocab_size)
                x = torch.where(corrupt_mask, random_tokens, x_0_pred)
            else:
                x = x_0_pred

        return x


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def distinct_ngrams(sequences: torch.Tensor, n: int = 2) -> float:
    """Compute distinct n-gram ratio as a diversity metric.

    Args:
        sequences: (B, L) token IDs.
        n: n-gram size.

    Returns:
        Ratio of distinct n-grams to total n-grams.
    """
    all_ngrams = []
    total = 0
    for seq in sequences:
        tokens = seq.tolist()
        for i in range(len(tokens) - n + 1):
            all_ngrams.append(tuple(tokens[i:i + n]))
            total += 1
    if total == 0:
        return 0.0
    return len(set(all_ngrams)) / total


def sample_perplexity(
    model: nn.Module,
    samples: torch.Tensor,
    reference_data: torch.Tensor,
) -> float:
    """Rough perplexity estimate: how well does the model's own predictions
    match its generated samples (self-consistency).

    This is NOT true perplexity under an external LM, but gives a relative
    comparison signal between models.
    """
    # Use the model to score its own samples at t=0
    with torch.no_grad():
        t = torch.zeros(samples.shape[0], device=samples.device)
        if hasattr(model, "net"):
            logits = model.net(samples, t)
        elif hasattr(model, "denoiser"):
            logits = model.denoiser(samples, t)
        else:
            return float("inf")

        log_probs = F.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(2, samples.unsqueeze(-1)).squeeze(-1)
        avg_log_prob = token_log_probs.mean().item()
    return math.exp(-avg_log_prob)


# ---------------------------------------------------------------------------
# Training and comparison
# ---------------------------------------------------------------------------

@dataclass
class TrainResult:
    """Training result for one model."""
    model_name: str
    losses: list[float]
    wall_time: float
    samples: Optional[torch.Tensor] = None
    diversity: float = 0.0
    self_perplexity: float = 0.0


def train_and_compare(
    data: torch.Tensor,
    vocab_size: int,
    mask_token_id: int,
    d_model: int = 256,
    n_heads: int = 4,
    n_layers: int = 4,
    max_seq_len: int = 128,
    num_epochs: int = 5,
    batch_size: int = 32,
    lr: float = 3e-4,
    num_sample_steps: int = 50,
    device: str = "cpu",
) -> tuple[TrainResult, TrainResult]:
    """Train MDM and D3PM side-by-side and compare.

    Args:
        data: (N, L) training token IDs.
        vocab_size: Vocabulary size.
        mask_token_id: ID for [MASK] (used by MDM only).
        d_model: Model hidden dim.
        n_heads: Attention heads.
        n_layers: Transformer layers.
        max_seq_len: Max sequence length.
        num_epochs: Training epochs.
        batch_size: Batch size.
        lr: Learning rate.
        num_sample_steps: Sampling steps.
        device: Device.

    Returns:
        (mdm_result, d3pm_result): TrainResult for each model.
    """
    shared_kwargs = dict(
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        max_seq_len=max_seq_len,
    )

    # -- MDM --
    mdm = MDMComparison(
        vocab_size=vocab_size, mask_token_id=mask_token_id, **shared_kwargs
    ).to(device)
    mdm_opt = torch.optim.AdamW(mdm.parameters(), lr=lr)

    # -- D3PM --
    d3pm = D3PMComparison(
        vocab_size=vocab_size, num_timesteps=100, **shared_kwargs
    ).to(device)
    d3pm_opt = torch.optim.AdamW(d3pm.parameters(), lr=lr)

    N = data.shape[0]
    seq_len = data.shape[1]

    def train_model(model, optimizer, name):
        losses = []
        start = time.time()
        model.train()
        for epoch in range(num_epochs):
            perm = torch.randperm(N)
            epoch_loss = 0.0
            n_batches = 0
            for i in range(0, N, batch_size):
                batch = data[perm[i:i + batch_size]].to(device)
                if batch.shape[0] == 0:
                    continue
                optimizer.zero_grad()
                loss = model.train_loss(batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            avg_loss = epoch_loss / max(n_batches, 1)
            losses.append(avg_loss)
        wall_time = time.time() - start

        # Generate samples
        model.train(False)
        samples = model.sample(
            batch_size=16, seq_len=seq_len,
            num_steps=num_sample_steps, device=device
        )
        diversity = distinct_ngrams(samples, n=2)
        ppl = sample_perplexity(model, samples, data[:16].to(device))

        return TrainResult(
            model_name=name,
            losses=losses,
            wall_time=wall_time,
            samples=samples,
            diversity=diversity,
            self_perplexity=ppl,
        )

    mdm_result = train_model(mdm, mdm_opt, "MDM")
    d3pm_result = train_model(d3pm, d3pm_opt, "D3PM")

    return mdm_result, d3pm_result


def print_comparison(mdm_result: TrainResult, d3pm_result: TrainResult):
    """Print a formatted comparison of the two models."""
    print("=" * 60)
    print("MDM vs D3PM Comparison")
    print("=" * 60)

    print(f"\n{'Metric':<25} {'MDM':>15} {'D3PM':>15}")
    print("-" * 55)
    print(f"{'Final train loss':<25} {mdm_result.losses[-1]:>15.4f} {d3pm_result.losses[-1]:>15.4f}")
    print(f"{'Training time (s)':<25} {mdm_result.wall_time:>15.2f} {d3pm_result.wall_time:>15.2f}")
    print(f"{'Distinct-2':<25} {mdm_result.diversity:>15.4f} {d3pm_result.diversity:>15.4f}")
    print(f"{'Self-perplexity':<25} {mdm_result.self_perplexity:>15.2f} {d3pm_result.self_perplexity:>15.2f}")

    print(f"\nTraining loss trajectory:")
    for epoch, (m, d) in enumerate(zip(mdm_result.losses, d3pm_result.losses)):
        print(f"  Epoch {epoch+1}: MDM={m:.4f}  D3PM={d:.4f}")


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)

    # Synthetic data: repeating patterns
    vocab_size = 50
    mask_id = 49
    seq_len = 32
    n_train = 256

    data = torch.randint(0, vocab_size - 1, (n_train, seq_len))

    mdm_result, d3pm_result = train_and_compare(
        data=data,
        vocab_size=vocab_size,
        mask_token_id=mask_id,
        d_model=64,
        n_heads=2,
        n_layers=2,
        max_seq_len=seq_len,
        num_epochs=3,
        batch_size=32,
        device="cpu",
    )

    print_comparison(mdm_result, d3pm_result)
