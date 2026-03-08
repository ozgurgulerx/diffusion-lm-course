"""Flow matching adapted for token sequences.

This module applies flow matching in token embedding space:
    1. Embed discrete tokens into continuous vectors using a learned embedding.
    2. Train flow matching on these embeddings (noise -> embedding ODE).
    3. Sample via ODE integration (Euler method).
    4. Round continuous outputs back to discrete tokens via nearest-neighbor.

Key differences from SDE-based Diffusion-LM:
    - No noise schedule (beta_t) -- just linear interpolation.
    - Deterministic sampling (no stochastic noise during generation).
    - Typically needs fewer sampling steps (50-100 vs 1000).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Sinusoidal timestep embedding
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


# ---------------------------------------------------------------------------
# Transformer blocks
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        x = x + self.ff(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Velocity network for text embeddings
# ---------------------------------------------------------------------------

class TextVelocityNet(nn.Module):
    """Transformer that predicts velocity in embedding space.

    Args:
        embed_dim: Token embedding dimension.
        d_model: Transformer width.
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
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.output_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, embed_dim)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_t: Noisy/interpolated embeddings, (batch, seq_len, embed_dim).
            t: Time in [0, 1], (batch,).

        Returns:
            Predicted velocity, (batch, seq_len, embed_dim).
        """
        h = self.input_proj(x_t)
        t_emb = self.time_proj(self.time_embed(t))
        h = h + t_emb.unsqueeze(1)

        for block in self.blocks:
            h = block(h)

        h = self.output_norm(h)
        return self.output_proj(h)


# ---------------------------------------------------------------------------
# TextFlowMatcher
# ---------------------------------------------------------------------------

class TextFlowMatcher:
    """End-to-end flow matching for text generation.

    Pipeline:
        1. Token IDs -> continuous embeddings (via learned embedding table).
        2. Flow matching training in embedding space.
        3. ODE sampling from noise to embeddings.
        4. Rounding: nearest-neighbor lookup to recover token IDs.

    Args:
        vocab_size: Number of tokens in vocabulary.
        embed_dim: Embedding dimension.
        d_model: Transformer width.
        n_heads: Number of attention heads.
        n_layers: Number of Transformer blocks.
        d_ff: Feed-forward hidden dimension.
        seq_len: Maximum sequence length.
        lr: Learning rate.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 512,
        seq_len: int = 64,
        lr: float = 1e-4,
        dropout: float = 0.1,
    ):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.seq_len = seq_len

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        nn.init.normal_(self.embedding.weight, std=0.02)

        self.velocity_net = TextVelocityNet(
            embed_dim=embed_dim,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
        )

        self.optimizer = torch.optim.AdamW(
            list(self.embedding.parameters())
            + list(self.velocity_net.parameters()),
            lr=lr,
        )

    def to(self, device: torch.device) -> "TextFlowMatcher":
        self.embedding.to(device)
        self.velocity_net.to(device)
        return self

    @property
    def device(self) -> torch.device:
        return self.embedding.weight.device

    def embed_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Convert token IDs to continuous embeddings."""
        return self.embedding(token_ids)

    def round_to_tokens(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Round continuous embeddings to nearest token IDs via cosine similarity."""
        emb_norm = F.normalize(embeddings, dim=-1)
        table_norm = F.normalize(self.embedding.weight, dim=-1)
        sim = torch.matmul(emb_norm, table_norm.t())
        return sim.argmax(dim=-1)

    def train_step(self, token_ids: torch.Tensor) -> float:
        """One flow matching training step.

        Args:
            token_ids: (batch, seq_len) token IDs.

        Returns:
            Scalar loss value.
        """
        self.velocity_net.train()
        device = self.device
        batch_size = token_ids.shape[0]

        x_1 = self.embed_tokens(token_ids)
        x_0 = torch.randn_like(x_1)
        t = torch.rand(batch_size, device=device)

        t_expanded = t.unsqueeze(-1).unsqueeze(-1)
        x_t = (1 - t_expanded) * x_0 + t_expanded * x_1
        velocity_target = x_1 - x_0

        velocity_pred = self.velocity_net(x_t, t)
        loss = F.mse_loss(velocity_pred, velocity_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    @torch.no_grad()
    def sample_embeddings(
        self,
        batch_size: int,
        n_steps: int = 100,
    ) -> torch.Tensor:
        """Generate embeddings via ODE integration."""
        self.velocity_net.eval()
        device = self.device

        x = torch.randn(batch_size, self.seq_len, self.embed_dim, device=device)
        dt = 1.0 / n_steps

        for step in range(n_steps):
            t = torch.full((batch_size,), step * dt, device=device)
            v = self.velocity_net(x, t)
            x = x + dt * v

        return x

    @torch.no_grad()
    def generate(
        self,
        batch_size: int,
        n_steps: int = 100,
    ) -> torch.Tensor:
        """Generate token sequences: sample embeddings then round to tokens."""
        embeddings = self.sample_embeddings(batch_size, n_steps)
        return self.round_to_tokens(embeddings)

    @torch.no_grad()
    def generate_with_trajectory(
        self,
        batch_size: int,
        n_steps: int = 100,
        save_every: int = 10,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, float]]]:
        """Generate tokens and save intermediate ODE states."""
        self.velocity_net.eval()
        device = self.device

        x = torch.randn(batch_size, self.seq_len, self.embed_dim, device=device)
        dt = 1.0 / n_steps

        trajectory = [(self.round_to_tokens(x), 0.0)]

        for step in range(n_steps):
            t_val = step * dt
            t = torch.full((batch_size,), t_val, device=device)
            v = self.velocity_net(x, t)
            x = x + dt * v

            if (step + 1) % save_every == 0:
                trajectory.append((self.round_to_tokens(x), t_val + dt))

        final_tokens = self.round_to_tokens(x)
        return final_tokens, trajectory

    def count_parameters(self) -> dict[str, int]:
        """Count model parameters."""
        emb_params = sum(p.numel() for p in self.embedding.parameters())
        vel_params = sum(p.numel() for p in self.velocity_net.parameters())
        return {
            "embedding": emb_params,
            "velocity_net": vel_params,
            "total": emb_params + vel_params,
        }


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo_text_flow_matching():
    """Demonstrate TextFlowMatcher on synthetic token data."""
    torch.manual_seed(42)
    device = torch.device("cpu")

    vocab_size = 100
    seq_len = 16
    batch_size = 32

    tfm = TextFlowMatcher(
        vocab_size=vocab_size,
        embed_dim=64,
        d_model=128,
        n_heads=4,
        n_layers=2,
        d_ff=256,
        seq_len=seq_len,
        lr=3e-4,
    )
    tfm.to(device)

    params = tfm.count_parameters()
    print(f"Model parameters: {params}")

    train_data = torch.randint(0, vocab_size, (200, seq_len), device=device)

    print("\nTraining TextFlowMatcher...")
    for step in range(50):
        idx = torch.randint(0, 200, (batch_size,))
        batch = train_data[idx]
        loss = tfm.train_step(batch)
        if (step + 1) % 10 == 0:
            print(f"  Step {step+1}: loss = {loss:.4f}")

    print("\nGenerating token sequences...")
    tokens = tfm.generate(batch_size=4, n_steps=50)
    print(f"Generated token IDs shape: {tokens.shape}")
    print(f"Sample output: {tokens[0].tolist()}")

    final_tokens, trajectory = tfm.generate_with_trajectory(
        batch_size=1, n_steps=50, save_every=10
    )
    print(f"\nTrajectory (how tokens evolve during generation):")
    for tokens_at_t, t_val in trajectory:
        print(f"  t={t_val:.2f}: {tokens_at_t[0, :8].tolist()}...")
    print(f"  Final: {final_tokens[0, :8].tolist()}...")


if __name__ == "__main__":
    demo_text_flow_matching()
