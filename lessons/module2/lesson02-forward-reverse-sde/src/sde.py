"""Variance-Preserving SDE for continuous diffusion in embedding space.

Implements the forward and reverse SDEs from Song et al. (2021),
"Score-Based Generative Modeling through Stochastic Differential Equations."

Forward SDE:  dx = -0.5 * beta(t) * x dt + sqrt(beta(t)) dw
This gradually transforms data into Gaussian noise.

Reverse SDE:  dx = [-0.5 * beta(t) * x - beta(t) * score(x,t)] dt + sqrt(beta(t)) dw_bar
This denoises, using the learned score function.
"""

import math

import torch
import torch.nn as nn


class VPSDE:
    """Variance-Preserving Stochastic Differential Equation.

    The VP-SDE uses a linear noise schedule beta(t) = beta_min + t * (beta_max - beta_min)
    for t in [0, 1]. The forward process has closed-form marginals:

        x_t = sqrt(alpha_bar(t)) * x_0 + sqrt(1 - alpha_bar(t)) * eps

    where alpha_bar(t) = exp(-0.5 * integral_0^t beta(s) ds).

    Args:
        beta_min: Minimum noise rate.
        beta_max: Maximum noise rate.
    """

    def __init__(self, beta_min: float = 0.1, beta_max: float = 20.0):
        self.beta_min = beta_min
        self.beta_max = beta_max

    def beta(self, t: torch.Tensor) -> torch.Tensor:
        """Instantaneous noise rate at time t.

        Args:
            t: Time tensor with values in [0, 1].
        """
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def integral_beta(self, t: torch.Tensor) -> torch.Tensor:
        """Integral of beta from 0 to t: int_0^t beta(s) ds."""
        return self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * t ** 2

    def alpha_bar(self, t: torch.Tensor) -> torch.Tensor:
        """Cumulative signal retention: exp(-0.5 * integral_beta(t)).

        This is the fraction of original signal preserved at time t.
        """
        return torch.exp(-0.5 * self.integral_beta(t))

    def forward_marginal(
        self, x_0: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample from the forward marginal q(x_t | x_0).

        Args:
            x_0: Clean data of shape (batch, ...).
            t: Time values of shape (batch,) in [0, 1].

        Returns:
            Tuple of (x_t, noise, alpha_bar_t) where:
                x_t: Noisy samples, same shape as x_0.
                noise: The Gaussian noise added, same shape as x_0.
                alpha_bar_t: Signal retention coefficient per sample.
        """
        # Reshape t for broadcasting: (batch, 1, 1, ...) depending on x_0 dims
        ab = self.alpha_bar(t)
        while ab.dim() < x_0.dim():
            ab = ab.unsqueeze(-1)

        noise = torch.randn_like(x_0)
        mean = torch.sqrt(ab) * x_0
        std = torch.sqrt(1.0 - ab)
        x_t = mean + std * noise
        return x_t, noise, ab.squeeze()

    def sample_prior(self, shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
        """Sample from the prior distribution p(x_T) = N(0, I).

        Args:
            shape: Shape of the sample tensor.
            device: Device to create tensor on.
        """
        return torch.randn(shape, device=device)

    def reverse_step(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        score: torch.Tensor,
        dt: float,
        stochastic: bool = True,
    ) -> torch.Tensor:
        """One step of the reverse SDE (Euler-Maruyama discretization).

        Reverse SDE: dx = [-0.5*beta(t)*x - beta(t)*score(x,t)] dt + sqrt(beta(t)) dw

        Args:
            x_t: Current state, shape (batch, ...).
            t: Current time, shape (batch,).
            score: Score estimate at (x_t, t), same shape as x_t.
            dt: Time step size (negative, since we go from t=1 to t=0).
            stochastic: If True, add noise (SDE). If False, deterministic (ODE).

        Returns:
            x_{t+dt}: Next state.
        """
        beta_t = self.beta(t)
        while beta_t.dim() < x_t.dim():
            beta_t = beta_t.unsqueeze(-1)

        # Drift: -0.5 * beta(t) * x - beta(t) * score
        drift = -0.5 * beta_t * x_t - beta_t * score
        diffusion = torch.sqrt(beta_t)

        x_next = x_t + drift * dt
        if stochastic and dt < 0:
            # For reverse SDE, noise is added with sqrt(|dt|)
            noise = torch.randn_like(x_t)
            x_next = x_next + diffusion * math.sqrt(abs(dt)) * noise

        return x_next

    def score_from_noise(
        self, noise: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Convert a noise prediction to a score estimate.

        score(x, t) = -noise / sqrt(1 - alpha_bar(t))

        This is used when the model predicts the noise epsilon rather than
        the score directly.

        Args:
            noise: Predicted noise, shape (batch, ...).
            t: Time values, shape (batch,).

        Returns:
            Score estimate, same shape as noise.
        """
        ab = self.alpha_bar(t)
        while ab.dim() < noise.dim():
            ab = ab.unsqueeze(-1)
        return -noise / torch.sqrt(1.0 - ab + 1e-8)

    def noise_from_score(
        self, score: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Convert a score estimate to noise prediction.

        noise = -score * sqrt(1 - alpha_bar(t))

        Args:
            score: Score estimate, shape (batch, ...).
            t: Time values, shape (batch,).

        Returns:
            Noise prediction, same shape as score.
        """
        ab = self.alpha_bar(t)
        while ab.dim() < score.dim():
            ab = ab.unsqueeze(-1)
        return -score * torch.sqrt(1.0 - ab + 1e-8)


def generate_with_sde(
    sde: VPSDE,
    score_fn,
    shape: tuple[int, ...],
    device: torch.device,
    n_steps: int = 100,
    stochastic: bool = True,
) -> list[torch.Tensor]:
    """Generate samples by running the reverse SDE.

    Args:
        sde: A VPSDE instance.
        score_fn: Callable(x_t, t) -> score estimate.
        shape: Shape of samples to generate (batch, seq_len, embed_dim).
        device: Device to run on.
        n_steps: Number of reverse steps.
        stochastic: Use SDE (True) or probability flow ODE (False).

    Returns:
        List of intermediate states [x_T, ..., x_0] for visualization.
    """
    dt = -1.0 / n_steps
    x = sde.sample_prior(shape, device)
    trajectory = [x.clone()]

    for i in range(n_steps):
        t_val = 1.0 - i / n_steps
        t = torch.full((shape[0],), t_val, device=device)
        score = score_fn(x, t)
        x = sde.reverse_step(x, t, score, dt, stochastic=stochastic)
        trajectory.append(x.clone())

    return trajectory
