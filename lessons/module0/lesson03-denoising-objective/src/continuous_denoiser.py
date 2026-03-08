"""Continuous denoiser: predict clean embeddings (or noise) from noisy embeddings.

Training objective: given noisy embedding vectors x_t, predict either:
  - the clean embeddings x_0 ("x-prediction"), or
  - the noise epsilon that was added ("epsilon-prediction").

Both are trained with MSE loss. This is the objective used in continuous
diffusion models like Diffusion-LM (Li et al., 2022).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContinuousDenoiser(nn.Module):
    """Simple MLP that predicts clean embeddings from noisy embeddings.

    Takes noisy embeddings and a noise level as input, outputs predicted
    clean embeddings. Each position is processed independently.
    """

    def __init__(self, embed_dim: int = 32, hidden_dim: int = 64):
        super().__init__()
        # +1 for the noise level which is concatenated to each position
        self.net = nn.Sequential(
            nn.Linear(embed_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, noisy_emb: torch.Tensor, noise_level: torch.Tensor) -> torch.Tensor:
        """Predict clean embeddings from noisy embeddings.

        Args:
            noisy_emb: Noisy embeddings, shape (batch, seq_len, embed_dim).
            noise_level: Noise level per sample, shape (batch, 1, 1) or broadcastable.

        Returns:
            Predicted clean embeddings, shape (batch, seq_len, embed_dim).
        """
        # Expand noise_level to (batch, seq_len, 1) and concatenate
        batch, seq_len, _ = noisy_emb.shape
        t = noise_level.expand(batch, seq_len, 1)
        x = torch.cat([noisy_emb, t], dim=-1)  # (batch, seq_len, embed_dim + 1)
        return self.net(x)


def continuous_denoising_loss(
    model: ContinuousDenoiser,
    clean_emb: torch.Tensor,
    prediction_target: str = "x0",
) -> torch.Tensor:
    """Compute the continuous denoising loss for one batch.

    Steps:
    1. Sample a random noise level for each batch element.
    2. Add Gaussian noise to clean embeddings.
    3. Feed noisy embeddings + noise level to the model.
    4. Compute MSE loss between predictions and target (clean emb or noise).

    Args:
        model: A ContinuousDenoiser instance.
        clean_emb: Clean embeddings of shape (batch, seq_len, embed_dim).
        prediction_target: "x0" to predict clean embeddings, "eps" to predict noise.

    Returns:
        Scalar loss tensor.
    """
    batch_size = clean_emb.shape[0]

    # Sample random noise level per batch element
    noise_level = torch.rand(batch_size, 1, 1, device=clean_emb.device)

    # Forward diffusion: add noise
    alpha = 1.0 - noise_level
    noise = torch.randn_like(clean_emb)
    noisy_emb = (alpha ** 0.5) * clean_emb + ((1.0 - alpha) ** 0.5) * noise

    # Predict
    prediction = model(noisy_emb, noise_level)

    # Loss
    if prediction_target == "x0":
        target = clean_emb
    elif prediction_target == "eps":
        target = noise
    else:
        raise ValueError(f"Unknown prediction_target: {prediction_target}")

    return F.mse_loss(prediction, target)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(42)

    embed_dim = 16
    seq_len = 6
    batch_size = 4

    model = ContinuousDenoiser(embed_dim=embed_dim, hidden_dim=32)

    # Fake clean embeddings
    clean_emb = torch.randn(batch_size, seq_len, embed_dim)

    loss_x0 = continuous_denoising_loss(model, clean_emb, prediction_target="x0")
    loss_eps = continuous_denoising_loss(model, clean_emb, prediction_target="eps")

    print(f"Continuous denoising loss (x0-prediction): {loss_x0.item():.4f}")
    print(f"Continuous denoising loss (eps-prediction): {loss_eps.item():.4f}")
