"""Reference solution for Lab 01: Flow Matching Text Generator.

This file contains complete implementations of:
    - SimpleTokenizer: word-level tokenizer with special tokens.
    - FlowMatchingTextGenerator: flow matching in embedding space.
    - SDETextGenerator: DDPM-style baseline for comparison.
"""

import math
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

class SimpleTokenizer:
    """Word-level tokenizer with special tokens."""

    PAD = "<pad>"
    UNK = "<unk>"
    BOS = "<bos>"
    EOS = "<eos>"

    def __init__(self, max_vocab_size: int = 5000):
        self.max_vocab_size = max_vocab_size
        self.word2idx: dict[str, int] = {}
        self.idx2word: list[str] = []

    def build_vocab(self, texts: list[str]):
        """Build vocabulary from a list of text strings."""
        # Add special tokens first
        special_tokens = [self.PAD, self.UNK, self.BOS, self.EOS]
        self.idx2word = list(special_tokens)
        self.word2idx = {tok: i for i, tok in enumerate(special_tokens)}

        # Count word frequencies
        counter: Counter[str] = Counter()
        for text in texts:
            words = text.lower().split()
            counter.update(words)

        # Add most common words up to max_vocab_size
        remaining = self.max_vocab_size - len(special_tokens)
        for word, _ in counter.most_common(remaining):
            if word not in self.word2idx:
                idx = len(self.idx2word)
                self.word2idx[word] = idx
                self.idx2word.append(word)

    @property
    def vocab_size(self) -> int:
        return len(self.idx2word)

    @property
    def pad_id(self) -> int:
        return self.word2idx[self.PAD]

    def encode(self, text: str, max_len: int = 64) -> list[int]:
        """Encode text to token IDs with BOS/EOS and padding."""
        words = text.lower().split()
        ids = [self.word2idx.get(w, self.word2idx[self.UNK]) for w in words]

        # Add BOS and EOS
        ids = [self.word2idx[self.BOS]] + ids + [self.word2idx[self.EOS]]

        # Truncate
        ids = ids[:max_len]

        # Pad
        ids = ids + [self.word2idx[self.PAD]] * (max_len - len(ids))

        return ids

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs back to text."""
        words = []
        for idx in ids:
            if idx >= len(self.idx2word):
                continue
            word = self.idx2word[idx]
            if word == self.EOS or word == self.PAD:
                break
            if word == self.BOS:
                continue
            words.append(word)
        return " ".join(words)


# ---------------------------------------------------------------------------
# Network components (same as notebook)
# ---------------------------------------------------------------------------

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.d_model // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=t.device, dtype=torch.float)
            / half
        )
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        x = x + self.ff(self.norm2(x))
        return x


class VelocityNet(nn.Module):
    def __init__(
        self,
        embed_dim=128,
        d_model=256,
        n_heads=4,
        n_layers=4,
        d_ff=512,
        dropout=0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(embed_dim, d_model)
        self.time_embed = SinusoidalTimeEmbedding(d_model)
        self.time_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(d_model, n_heads, d_ff, dropout)
                for _ in range(n_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, embed_dim)

    def forward(self, x_t, t):
        h = self.input_proj(x_t)
        t_emb = self.time_proj(self.time_embed(t))
        h = h + t_emb.unsqueeze(1)
        for block in self.blocks:
            h = block(h)
        h = self.output_norm(h)
        return self.output_proj(h)


# ---------------------------------------------------------------------------
# Flow Matching Text Generator
# ---------------------------------------------------------------------------

class FlowMatchingTextGenerator:
    """Flow matching text generator -- reference solution."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 512,
        seq_len: int = 32,
        lr: float = 1e-4,
    ):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.seq_len = seq_len

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        nn.init.normal_(self.embedding.weight, std=0.02)

        self.velocity_net = VelocityNet(
            embed_dim=embed_dim,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
        )

        self.optimizer = torch.optim.AdamW(
            list(self.embedding.parameters())
            + list(self.velocity_net.parameters()),
            lr=lr,
        )

    def to(self, device):
        self.embedding.to(device)
        self.velocity_net.to(device)
        return self

    @property
    def device(self):
        return self.embedding.weight.device

    def train_step(self, token_ids: torch.Tensor) -> float:
        """One flow matching training step."""
        self.velocity_net.train()
        batch_size = token_ids.shape[0]

        # 1. Embed tokens to get x_1 (target)
        x_1 = self.embedding(token_ids)

        # 2. Sample noise
        x_0 = torch.randn_like(x_1)

        # 3. Sample time
        t = torch.rand(batch_size, device=self.device)

        # 4. Interpolate
        t_expanded = t.unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1)
        x_t = (1 - t_expanded) * x_0 + t_expanded * x_1

        # 5. Target velocity
        v_target = x_1 - x_0

        # 6. Predict velocity
        v_pred = self.velocity_net(x_t, t)

        # 7. Compute loss
        loss = F.mse_loss(v_pred, v_target)

        # 8. Backprop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    @torch.no_grad()
    def sample(self, batch_size: int, n_steps: int = 100) -> torch.Tensor:
        """Generate token IDs via ODE sampling."""
        self.velocity_net.eval()

        # Start from noise
        x = torch.randn(
            batch_size, self.seq_len, self.embed_dim, device=self.device
        )
        dt = 1.0 / n_steps

        # Euler integration
        for step in range(n_steps):
            t = torch.full((batch_size,), step * dt, device=self.device)
            v = self.velocity_net(x, t)
            x = x + dt * v

        # Round to tokens via cosine similarity
        x_norm = F.normalize(x, dim=-1)
        table_norm = F.normalize(self.embedding.weight, dim=-1)
        sim = torch.matmul(x_norm, table_norm.t())
        token_ids = sim.argmax(dim=-1)

        return token_ids


# ---------------------------------------------------------------------------
# SDE Text Generator (DDPM baseline)
# ---------------------------------------------------------------------------

class SDETextGenerator:
    """Simplified DDPM-style text generator -- reference solution."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 512,
        seq_len: int = 32,
        num_timesteps: int = 1000,
        lr: float = 1e-4,
    ):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.num_timesteps = num_timesteps

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        nn.init.normal_(self.embedding.weight, std=0.02)

        self.noise_net = VelocityNet(
            embed_dim=embed_dim,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
        )

        # Noise schedule
        betas = torch.linspace(1e-4, 0.02, num_timesteps)
        alphas = 1.0 - betas
        self.betas = betas
        self.alphas = alphas
        self.alpha_bars = torch.cumprod(alphas, dim=0)

        self.optimizer = torch.optim.AdamW(
            list(self.embedding.parameters())
            + list(self.noise_net.parameters()),
            lr=lr,
        )

    def to(self, device):
        self.embedding.to(device)
        self.noise_net.to(device)
        self.alpha_bars = self.alpha_bars.to(device)
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        return self

    @property
    def device(self):
        return self.embedding.weight.device

    def train_step(self, token_ids: torch.Tensor) -> float:
        """One DDPM training step."""
        self.noise_net.train()
        batch_size = token_ids.shape[0]

        # 1. Embed tokens
        x_0 = self.embedding(token_ids)

        # 2. Sample timesteps
        t = torch.randint(
            0, self.num_timesteps, (batch_size,), device=self.device
        )

        # 3. Sample noise
        epsilon = torch.randn_like(x_0)

        # 4. Forward diffusion
        alpha_bar_t = self.alpha_bars[t].unsqueeze(-1).unsqueeze(-1)
        x_t = alpha_bar_t.sqrt() * x_0 + (1 - alpha_bar_t).sqrt() * epsilon

        # 5. Predict noise (normalize t to [0, 1])
        t_normalized = t.float() / self.num_timesteps
        epsilon_pred = self.noise_net(x_t, t_normalized)

        # 6. Loss
        loss = F.mse_loss(epsilon_pred, epsilon)

        # 7. Backprop
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    @torch.no_grad()
    def sample(self, batch_size: int, n_steps: int = None) -> torch.Tensor:
        """Generate token IDs via DDPM reverse process."""
        self.noise_net.eval()
        if n_steps is None:
            n_steps = self.num_timesteps

        # Start from noise
        x = torch.randn(
            batch_size, self.seq_len, self.embed_dim, device=self.device
        )

        # Create timestep schedule (evenly spaced, descending)
        timesteps = torch.linspace(
            self.num_timesteps - 1, 0, n_steps, device=self.device
        ).long()

        for i, t_val in enumerate(timesteps):
            t = torch.full(
                (batch_size,), t_val, device=self.device, dtype=torch.long
            )
            t_normalized = t.float() / self.num_timesteps

            # Predict noise
            epsilon_pred = self.noise_net(x, t_normalized)

            # Get schedule values
            alpha_bar_t = self.alpha_bars[t_val]
            alpha_t = self.alphas[t_val]
            beta_t = self.betas[t_val]

            # Predict x_0
            x_0_hat = (
                x - (1 - alpha_bar_t).sqrt() * epsilon_pred
            ) / alpha_bar_t.sqrt()
            x_0_hat = x_0_hat.clamp(-5, 5)

            if t_val > 0:
                alpha_bar_prev = self.alpha_bars[t_val - 1]

                # Posterior mean
                coeff_x0 = (
                    alpha_bar_prev.sqrt() * beta_t / (1 - alpha_bar_t)
                )
                coeff_xt = (
                    alpha_t.sqrt() * (1 - alpha_bar_prev) / (1 - alpha_bar_t)
                )
                mean = coeff_x0 * x_0_hat + coeff_xt * x

                # Posterior variance
                var = (
                    beta_t * (1 - alpha_bar_prev) / (1 - alpha_bar_t)
                )
                noise = torch.randn_like(x)
                x = mean + var.sqrt() * noise
            else:
                x = x_0_hat

        # Round to tokens
        x_norm = F.normalize(x, dim=-1)
        table_norm = F.normalize(self.embedding.weight, dim=-1)
        sim = torch.matmul(x_norm, table_norm.t())
        token_ids = sim.argmax(dim=-1)

        return token_ids
