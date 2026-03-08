"""Flow matching for generative modeling.

Flow matching defines a deterministic ODE that transports noise to data along
straight paths. Instead of learning a score function for an SDE, we learn a
velocity field v(x, t) for an ODE:

    dx/dt = v(x, t)

Training is remarkably simple:
    1. Sample a data point x_1 and noise x_0 ~ N(0, I).
    2. Interpolate: x_t = (1 - t) * x_0 + t * x_1.
    3. The target velocity is: v_target = x_1 - x_0.
    4. Train the model to predict v_target from (x_t, t).

Advantages over score-based diffusion:
    - Simpler loss (no noise schedule to tune).
    - Deterministic sampling via ODE (no stochastic noise during generation).
    - Fewer function evaluations needed (straighter paths).

Reference:
    Lipman et al., "Flow Matching for Generative Modeling" (2023)

This module implements:
    - FlowMatcher: training and sampling with flow matching.
    - VelocityNet: a small MLP-based velocity predictor (for 2D demos).
    - SequenceVelocityNet: Transformer velocity predictor for sequences.
    - Euler ODE solver for sampling.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Sinusoidal timestep embedding
# ---------------------------------------------------------------------------

class SinusoidalTimeEmbedding(nn.Module):
    """Maps scalar timestep t in [0, 1] to a d-dimensional embedding."""

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
# VelocityNet (simple MLP for 2D demonstrations)
# ---------------------------------------------------------------------------

class VelocityNet(nn.Module):
    """MLP that predicts velocity v(x_t, t) for low-dimensional data.

    Good for visualizing flow matching on 2D distributions.

    Args:
        data_dim: Dimensionality of x (e.g., 2 for 2D demos).
        hidden_dim: Width of hidden layers.
        n_layers: Number of hidden layers.
        time_dim: Dimensionality of time embedding.
    """

    def __init__(
        self,
        data_dim: int = 2,
        hidden_dim: int = 256,
        n_layers: int = 3,
        time_dim: int = 64,
    ):
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(time_dim)
        self.time_proj = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.GELU(),
        )

        layers = [nn.Linear(data_dim + hidden_dim, hidden_dim), nn.GELU()]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU()])
        layers.append(nn.Linear(hidden_dim, data_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_t: Current position, shape (batch, data_dim).
            t: Time in [0, 1], shape (batch,).

        Returns:
            Predicted velocity, shape (batch, data_dim).
        """
        t_emb = self.time_proj(self.time_embed(t))
        h = torch.cat([x_t, t_emb], dim=-1)
        return self.net(h)


# ---------------------------------------------------------------------------
# Transformer-based VelocityNet (for sequence data)
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """Pre-norm Transformer encoder block."""

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


class SequenceVelocityNet(nn.Module):
    """Transformer-based velocity predictor for sequence data.

    Args:
        input_dim: Dimension of each position (e.g., embed_dim).
        d_model: Transformer width.
        n_heads: Number of attention heads.
        n_layers: Number of Transformer blocks.
        d_ff: Feed-forward hidden dimension.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        input_dim: int = 128,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
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
        self.output_proj = nn.Linear(d_model, input_dim)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_t: Interpolated embeddings, shape (batch, seq_len, input_dim).
            t: Time in [0, 1], shape (batch,).

        Returns:
            Predicted velocity, shape (batch, seq_len, input_dim).
        """
        h = self.input_proj(x_t)
        t_emb = self.time_proj(self.time_embed(t))
        h = h + t_emb.unsqueeze(1)

        for block in self.blocks:
            h = block(h)

        h = self.output_norm(h)
        return self.output_proj(h)


# ---------------------------------------------------------------------------
# FlowMatcher
# ---------------------------------------------------------------------------

class FlowMatcher:
    """Flow matching training and sampling.

    Implements the conditional flow matching (CFM) framework:
        - Linear interpolation: x_t = (1 - t) * x_0 + t * x_1
        - Target velocity: v = x_1 - x_0
        - Loss: || model(x_t, t) - v ||^2

    Sampling uses the Euler method to integrate the learned ODE:
        x_{t+dt} = x_t + dt * v(x_t, t)

    Args:
        model: A velocity network (VelocityNet or SequenceVelocityNet).
        lr: Learning rate.
        sigma_min: Minimum noise added to the interpolation for stability.
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-4,
        sigma_min: float = 0.0,
    ):
        self.model = model
        self.sigma_min = sigma_min
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    def to(self, device: torch.device) -> "FlowMatcher":
        self.model.to(device)
        return self

    def interpolate(
        self, x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Linear interpolation from noise x_0 to data x_1.

        x_t = (1 - t) * x_0 + t * x_1
        velocity = x_1 - x_0

        Args:
            x_0: Source noise, shape (batch, ...).
            x_1: Target data, shape (batch, ...).
            t: Time in [0, 1], shape (batch,).

        Returns:
            (x_t, velocity) -- interpolated point and target velocity.
        """
        t_expanded = t
        while t_expanded.dim() < x_0.dim():
            t_expanded = t_expanded.unsqueeze(-1)

        x_t = (1 - t_expanded) * x_0 + t_expanded * x_1

        if self.sigma_min > 0:
            x_t = x_t + self.sigma_min * torch.randn_like(x_t)

        velocity = x_1 - x_0
        return x_t, velocity

    def train_step(self, x_1: torch.Tensor) -> float:
        """One training step of flow matching.

        Args:
            x_1: Data samples, shape (batch, ...).

        Returns:
            Scalar loss value.
        """
        self.model.train()
        batch_size = x_1.shape[0]
        device = x_1.device

        x_0 = torch.randn_like(x_1)
        t = torch.rand(batch_size, device=device)
        x_t, velocity_target = self.interpolate(x_0, x_1, t)
        velocity_pred = self.model(x_t, t)

        loss = F.mse_loss(velocity_pred, velocity_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    @torch.no_grad()
    def sample(
        self,
        shape: tuple,
        n_steps: int = 100,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        """Generate samples using Euler ODE integration.

        Integrates from t=0 (noise) to t=1 (data):
            x_{t+dt} = x_t + dt * v(x_t, t)

        Args:
            shape: Shape of the output.
            n_steps: Number of Euler steps.
            device: Device to generate on.

        Returns:
            Generated samples.
        """
        self.model.eval()
        x = torch.randn(shape, device=device)
        dt = 1.0 / n_steps

        for step in range(n_steps):
            t = torch.full((shape[0],), step * dt, device=device)
            v = self.model(x, t)
            x = x + dt * v

        return x

    @torch.no_grad()
    def sample_trajectory(
        self,
        shape: tuple,
        n_steps: int = 100,
        device: torch.device = torch.device("cpu"),
        save_every: int = 10,
    ) -> list[tuple[torch.Tensor, float]]:
        """Generate samples and save intermediate states for visualization.

        Returns:
            List of (x_t, t) at saved timesteps.
        """
        self.model.eval()
        x = torch.randn(shape, device=device)
        dt = 1.0 / n_steps

        trajectory = [(x.clone(), 0.0)]

        for step in range(n_steps):
            t_val = step * dt
            t = torch.full((shape[0],), t_val, device=device)
            v = self.model(x, t)
            x = x + dt * v

            if (step + 1) % save_every == 0:
                trajectory.append((x.clone(), t_val + dt))

        return trajectory


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def demo_2d_flow_matching():
    """Train flow matching on a 2D mixture of Gaussians."""
    torch.manual_seed(42)
    device = torch.device("cpu")

    def sample_target(n: int) -> torch.Tensor:
        centers = torch.tensor([
            [2.0, 2.0], [-2.0, 2.0], [2.0, -2.0], [-2.0, -2.0]
        ])
        idx = torch.randint(0, 4, (n,))
        return centers[idx] + torch.randn(n, 2) * 0.3

    model = VelocityNet(data_dim=2, hidden_dim=128, n_layers=3)
    fm = FlowMatcher(model, lr=1e-3)
    fm.to(device)

    print("Training flow matching on 2D Gaussian mixture...")
    for step in range(500):
        x_1 = sample_target(256).to(device)
        loss = fm.train_step(x_1)
        if (step + 1) % 100 == 0:
            print(f"  Step {step+1}: loss = {loss:.4f}")

    samples = fm.sample((500, 2), n_steps=50, device=device)
    print(f"\nGenerated {samples.shape[0]} samples")
    print(f"Sample mean: {samples.mean(0).tolist()}")
    print(f"Sample std:  {samples.std(0).tolist()}")


if __name__ == "__main__":
    demo_2d_flow_matching()
