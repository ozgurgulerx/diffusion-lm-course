"""Continuous noise for text: add Gaussian noise to token embeddings.

In continuous diffusion for text (e.g., Diffusion-LM by Li et al., 2022),
tokens are first embedded into a continuous vector space, and then
Gaussian noise is added to these embedding vectors. The forward process is:

    x_t = sqrt(alpha_t) * x_0 + sqrt(1 - alpha_t) * epsilon

where x_0 is the clean embedding and epsilon ~ N(0, I).
"""

import torch
import torch.nn as nn


def add_gaussian_noise(
    embeddings: torch.Tensor,
    noise_level: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Add Gaussian noise to embedding vectors.

    Uses the standard diffusion forward process parameterization:
        x_noisy = sqrt(alpha) * x_clean + sqrt(1 - alpha) * noise

    where alpha = 1 - noise_level (so noise_level=0 means clean,
    noise_level=1 means pure noise).

    Args:
        embeddings: Clean embeddings of shape (..., embed_dim).
        noise_level: Scalar in [0, 1]. Controls the signal-to-noise ratio.
            0.0 = clean signal, 1.0 = pure Gaussian noise.

    Returns:
        Tuple of (noisy_embeddings, noise) with same shape as input.
    """
    alpha = 1.0 - noise_level
    noise = torch.randn_like(embeddings)
    noisy = (alpha ** 0.5) * embeddings + ((1.0 - alpha) ** 0.5) * noise
    return noisy, noise


def embed_and_noise(
    token_ids: torch.Tensor,
    embedding_layer: nn.Embedding,
    noise_level: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Embed tokens and add Gaussian noise at a given level.

    This is the continuous-noise analog of discrete corruption:
    instead of replacing token IDs, we add noise to the embedding vectors.

    Args:
        token_ids: Token IDs of shape (seq_len,) or (batch, seq_len).
        embedding_layer: An nn.Embedding to map IDs to vectors.
        noise_level: Noise level in [0, 1].

    Returns:
        Tuple of (noisy_embeddings, clean_embeddings, noise).
    """
    clean = embedding_layer(token_ids)
    noisy, noise = add_gaussian_noise(clean, noise_level)
    return noisy, clean, noise


def noise_at_multiple_levels(
    token_ids: torch.Tensor,
    embedding_layer: nn.Embedding,
    levels: list[float] | None = None,
) -> dict[float, torch.Tensor]:
    """Generate noisy embeddings at multiple noise levels.

    Useful for visualization: see how embeddings degrade as noise increases.

    Args:
        token_ids: 1-D tensor of token IDs.
        embedding_layer: Embedding layer.
        levels: List of noise levels. Defaults to [0, 0.25, 0.5, 0.75, 1.0].

    Returns:
        Dict mapping noise_level -> noisy embeddings tensor.
    """
    if levels is None:
        levels = [0.0, 0.25, 0.5, 0.75, 1.0]

    results = {}
    for level in levels:
        noisy, _, _ = embed_and_noise(token_ids, embedding_layer, level)
        results[level] = noisy.detach()

    return results


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(42)

    vocab_size = 5
    embed_dim = 8
    emb = nn.Embedding(vocab_size, embed_dim)

    token_ids = torch.tensor([0, 1, 2, 3, 4])  # one of each token

    print("Embedding norms at different noise levels:")
    for level in [0.0, 0.25, 0.5, 0.75, 1.0]:
        noisy, clean, noise = embed_and_noise(token_ids, emb, level)
        # Show how far noisy is from clean
        dist = (noisy - clean).norm(dim=-1).mean().item()
        print(f"  noise={level:.2f}  mean L2 distance from clean: {dist:.3f}")
