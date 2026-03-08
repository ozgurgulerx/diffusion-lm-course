"""Solution for Lab 01: Implement MDM and compare with D3PM.

This solution implements:
1. A complete MDM (Masked Diffusion Model) from scratch
2. Training on WikiText-2
3. Comparison with a provided D3PM baseline
"""

import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------------
# 1. Transformer backbone
# ---------------------------------------------------------------------------

class TransformerDenoiser(nn.Module):
    """Transformer backbone for denoising masked sequences."""

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
# 2. Masking schedule
# ---------------------------------------------------------------------------

def cosine_masking_schedule(t: torch.Tensor, s: float = 0.008) -> torch.Tensor:
    """Cosine schedule: gamma(t) goes from 0 to 1 as t goes from 0 to 1."""
    f_t = torch.cos(((t + s) / (1 + s)) * (math.pi / 2)) ** 2
    f_0 = math.cos((s / (1 + s)) * (math.pi / 2)) ** 2
    return (1 - f_t / f_0).clamp(0, 1)


# ---------------------------------------------------------------------------
# 3. MDM implementation
# ---------------------------------------------------------------------------

class MDM(nn.Module):
    """Masked Diffusion Model.

    Forward process: mask each token independently with probability gamma(t).
    Reverse process: predict original tokens at masked positions.
    Loss: cross-entropy on masked positions, weighted by gamma'(t)/gamma(t).
    """

    def __init__(
        self,
        vocab_size: int,
        mask_token_id: int,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        max_seq_len: int = 128,
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

    def forward_corrupt(self, x_0, t):
        """Mask tokens with probability gamma(t)."""
        gamma_t = cosine_masking_schedule(t)  # (B,)
        rand = torch.rand_like(x_0, dtype=torch.float)
        mask = rand < gamma_t[:, None]
        x_t = x_0.clone()
        x_t[mask] = self.mask_token_id
        return x_t, mask

    def train_loss(self, x_0):
        """Compute MDM training loss."""
        B, L = x_0.shape
        t = torch.rand(B, device=x_0.device).clamp(0.01, 0.99)

        x_t, mask = self.forward_corrupt(x_0, t)
        logits = self.denoiser(x_t, t)  # (B, L, V)

        if not mask.any():
            return torch.tensor(0.0, device=x_0.device, requires_grad=True)

        # Cross-entropy on masked positions
        loss = F.cross_entropy(logits[mask], x_0[mask])
        return loss

    @torch.no_grad()
    def sample(self, batch_size, seq_len, num_steps=50, device="cpu",
               temperature=1.0):
        """Generate via iterative unmasking."""
        x = torch.full((batch_size, seq_len), self.mask_token_id,
                        dtype=torch.long, device=device)

        timesteps = torch.linspace(1.0, 0.0, num_steps + 1, device=device)

        for i in range(num_steps):
            t_now = timesteps[i]
            t_next = timesteps[i + 1]

            gamma_now = cosine_masking_schedule(t_now.unsqueeze(0)).item()
            gamma_next = cosine_masking_schedule(t_next.unsqueeze(0)).item()

            is_masked = (x == self.mask_token_id)
            n_masked = is_masked.float().sum(dim=-1)

            if gamma_now > 1e-6:
                frac = (gamma_now - gamma_next) / gamma_now
            else:
                frac = 1.0
            n_to_unmask = (n_masked * frac).clamp(min=1).long()

            t_batch = t_now.expand(batch_size)
            logits = self.denoiser(x, t_batch)
            probs = F.softmax(logits / temperature, dim=-1)

            sampled = torch.multinomial(
                probs.view(-1, self.vocab_size), 1
            ).view(batch_size, seq_len)

            confidence = torch.gather(probs, 2, sampled.unsqueeze(-1)).squeeze(-1)
            confidence[~is_masked] = -1.0

            for b in range(batch_size):
                n = n_to_unmask[b].item()
                mp = is_masked[b].nonzero(as_tuple=True)[0]
                if len(mp) == 0:
                    continue
                n = min(n, len(mp))
                _, topk = confidence[b, mp].topk(n)
                x[b, mp[topk]] = sampled[b, mp[topk]]

        # Final cleanup
        remaining = (x == self.mask_token_id)
        if remaining.any():
            t_zero = torch.zeros(batch_size, device=device)
            logits = self.denoiser(x, t_zero)
            probs = F.softmax(logits / temperature, dim=-1)
            s = torch.multinomial(probs.view(-1, self.vocab_size), 1).view(batch_size, seq_len)
            x[remaining] = s[remaining]

        return x


# ---------------------------------------------------------------------------
# 4. D3PM baseline (provided)
# ---------------------------------------------------------------------------

class D3PMBaseline(nn.Module):
    """D3PM with uniform transitions (provided as baseline)."""

    def __init__(self, vocab_size, num_timesteps=100, **kwargs):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_timesteps = num_timesteps
        self.net = TransformerDenoiser(vocab_size=vocab_size, **kwargs)

        betas = torch.linspace(1e-4, 0.02, num_timesteps)
        alphas = 1 - betas
        self.register_buffer("alpha_cumprod", torch.cumprod(alphas, dim=0))

    def train_loss(self, x_0):
        B, L = x_0.shape
        t_idx = torch.randint(0, self.num_timesteps, (B,), device=x_0.device)
        t_cont = t_idx.float() / self.num_timesteps

        alpha_bar = self.alpha_cumprod[t_idx]
        rand = torch.rand_like(x_0, dtype=torch.float)
        corrupt = rand > alpha_bar[:, None]
        random_tokens = torch.randint_like(x_0, 0, self.vocab_size)
        x_t = torch.where(corrupt, random_tokens, x_0)

        logits = self.net(x_t, t_cont)
        return F.cross_entropy(logits.view(-1, self.vocab_size), x_0.view(-1))

    @torch.no_grad()
    def sample(self, batch_size, seq_len, num_steps=50, device="cpu"):
        x = torch.randint(0, self.vocab_size, (batch_size, seq_len), device=device)
        step_size = max(1, self.num_timesteps // num_steps)

        for t_val in range(self.num_timesteps - 1, -1, -step_size):
            t_idx = torch.full((batch_size,), t_val, device=device, dtype=torch.long)
            logits = self.net(x, t_idx.float() / self.num_timesteps)
            x_0_pred = logits.argmax(dim=-1)

            if t_val > 0:
                t_prev = max(0, t_val - step_size)
                alpha_prev = self.alpha_cumprod[t_prev]
                rand = torch.rand_like(x, dtype=torch.float)
                corrupt = rand > alpha_prev
                x = torch.where(corrupt, torch.randint_like(x, 0, self.vocab_size), x_0_pred)
            else:
                x = x_0_pred
        return x


# ---------------------------------------------------------------------------
# 5. Data loading
# ---------------------------------------------------------------------------

def load_wikitext2(tokenizer, seq_len=128, max_samples=2000):
    """Load WikiText-2 and tokenize into fixed-length chunks.

    Args:
        tokenizer: A tokenizer with encode() method.
        seq_len: Chunk length.
        max_samples: Maximum number of chunks to return.

    Returns:
        (data, vocab_size, mask_token_id) where data is (N, seq_len) tensor.
    """
    from datasets import load_dataset

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = [t for t in dataset["text"] if len(t.strip()) > 20]

    # Tokenize all texts
    all_ids = []
    for text in texts:
        ids = tokenizer.encode(text)
        all_ids.extend(ids)
        if len(all_ids) > max_samples * seq_len:
            break

    # Chunk into fixed-length sequences
    n_chunks = min(len(all_ids) // seq_len, max_samples)
    all_ids = all_ids[:n_chunks * seq_len]
    data = torch.tensor(all_ids).view(n_chunks, seq_len)

    vocab_size = tokenizer.vocab_size
    # Use the last token ID as mask token (or tokenizer's mask token if available)
    if hasattr(tokenizer, "mask_token_id") and tokenizer.mask_token_id is not None:
        mask_token_id = tokenizer.mask_token_id
    else:
        mask_token_id = vocab_size  # append a new token
        vocab_size += 1

    return data, vocab_size, mask_token_id


# ---------------------------------------------------------------------------
# 6. Metrics
# ---------------------------------------------------------------------------

def distinct_ngrams(sequences: torch.Tensor, n: int = 2) -> float:
    """Distinct n-gram ratio (diversity metric)."""
    all_ngrams = set()
    total = 0
    for seq in sequences:
        tokens = seq.tolist()
        for i in range(len(tokens) - n + 1):
            all_ngrams.add(tuple(tokens[i:i + n]))
            total += 1
    return len(all_ngrams) / max(total, 1)


def self_perplexity(model, samples):
    """Self-consistency perplexity."""
    with torch.no_grad():
        t = torch.zeros(samples.shape[0], device=samples.device)
        if hasattr(model, "net"):
            logits = model.net(samples, t)
        else:
            logits = model.denoiser(samples, t)
        log_probs = F.log_softmax(logits, dim=-1)
        token_lp = log_probs.gather(2, samples.unsqueeze(-1)).squeeze(-1)
    return math.exp(-token_lp.mean().item())


# ---------------------------------------------------------------------------
# 7. Training loop
# ---------------------------------------------------------------------------

def train_model(model, data, num_epochs=5, batch_size=32, lr=3e-4, device="cpu"):
    """Train a model and return loss history and wall time."""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    dataset = TensorDataset(data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    losses = []
    start = time.time()

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = model.train_loss(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg = epoch_loss / max(n_batches, 1)
        losses.append(avg)
        print(f"  [{model.__class__.__name__}] Epoch {epoch+1}/{num_epochs}, Loss: {avg:.4f}")

    wall_time = time.time() - start
    return losses, wall_time


# ---------------------------------------------------------------------------
# 8. Main comparison
# ---------------------------------------------------------------------------

def run_comparison(
    data: torch.Tensor,
    vocab_size: int,
    mask_token_id: int,
    seq_len: int = 128,
    d_model: int = 256,
    n_heads: int = 4,
    n_layers: int = 4,
    num_epochs: int = 5,
    batch_size: int = 32,
    device: str = "cpu",
):
    """Run the full MDM vs D3PM comparison."""
    kwargs = dict(d_model=d_model, n_heads=n_heads, n_layers=n_layers,
                  max_seq_len=seq_len)

    # Build models
    mdm = MDM(vocab_size=vocab_size, mask_token_id=mask_token_id, **kwargs)
    d3pm = D3PMBaseline(vocab_size=vocab_size, **kwargs)

    print("Training MDM...")
    mdm_losses, mdm_time = train_model(mdm, data, num_epochs, batch_size, device=device)

    print("\nTraining D3PM...")
    d3pm_losses, d3pm_time = train_model(d3pm, data, num_epochs, batch_size, device=device)

    # Generate samples
    print("\nGenerating samples...")
    mdm.to(device)
    d3pm.to(device)

    mdm_samples = mdm.sample(16, seq_len, num_steps=50, device=device)
    d3pm_samples = d3pm.sample(16, seq_len, num_steps=50, device=device)

    # Metrics
    mdm_div = distinct_ngrams(mdm_samples)
    d3pm_div = distinct_ngrams(d3pm_samples)
    mdm_ppl = self_perplexity(mdm, mdm_samples)
    d3pm_ppl = self_perplexity(d3pm, d3pm_samples)

    # Report
    print("\n" + "=" * 60)
    print("RESULTS: MDM vs D3PM")
    print("=" * 60)
    print(f"{'Metric':<25} {'MDM':>15} {'D3PM':>15}")
    print("-" * 55)
    print(f"{'Final loss':<25} {mdm_losses[-1]:>15.4f} {d3pm_losses[-1]:>15.4f}")
    print(f"{'Training time (s)':<25} {mdm_time:>15.1f} {d3pm_time:>15.1f}")
    print(f"{'Distinct-2':<25} {mdm_div:>15.4f} {d3pm_div:>15.4f}")
    print(f"{'Self-perplexity':<25} {mdm_ppl:>15.1f} {d3pm_ppl:>15.1f}")

    return {
        "mdm": {"losses": mdm_losses, "time": mdm_time, "diversity": mdm_div,
                 "perplexity": mdm_ppl, "samples": mdm_samples},
        "d3pm": {"losses": d3pm_losses, "time": d3pm_time, "diversity": d3pm_div,
                  "perplexity": d3pm_ppl, "samples": d3pm_samples},
    }


if __name__ == "__main__":
    torch.manual_seed(42)

    # Quick demo with synthetic data
    vocab_size = 50
    mask_id = 49
    seq_len = 32
    data = torch.randint(0, vocab_size - 1, (256, seq_len))

    run_comparison(
        data=data,
        vocab_size=vocab_size,
        mask_token_id=mask_id,
        seq_len=seq_len,
        d_model=64,
        n_heads=2,
        n_layers=2,
        num_epochs=3,
        batch_size=32,
    )
