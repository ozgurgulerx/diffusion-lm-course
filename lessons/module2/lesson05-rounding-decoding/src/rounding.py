"""Rounding strategies: converting continuous embeddings back to discrete tokens.

The core challenge in continuous text diffusion: after denoising in embedding
space, we must map back to discrete tokens. Simple nearest-neighbor works but
is fragile. Better strategies improve generation quality significantly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def nearest_neighbor_round(
    continuous: torch.Tensor,
    embedding_weight: torch.Tensor,
) -> torch.Tensor:
    """Simple nearest-neighbor rounding via argmax dot product.

    For each continuous vector, find the token embedding with the
    highest cosine similarity (or equivalently, argmin L2 distance
    when embeddings have similar norms).

    Args:
        continuous: Predicted embeddings (batch, seq_len, embed_dim).
        embedding_weight: Embedding matrix (vocab_size, embed_dim).

    Returns:
        Token IDs (batch, seq_len).
    """
    logits = torch.einsum("bsd,vd->bsv", continuous, embedding_weight)
    return logits.argmax(dim=-1)


def softmax_round(
    continuous: torch.Tensor,
    embedding_weight: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Soft rounding via temperature-scaled softmax sampling.

    Instead of hard argmax, sample from the softmax distribution over
    token similarities. Lower temperature = more deterministic.

    Args:
        continuous: Predicted embeddings (batch, seq_len, embed_dim).
        embedding_weight: Embedding matrix (vocab_size, embed_dim).
        temperature: Softmax temperature (lower = sharper).

    Returns:
        Token IDs (batch, seq_len).
    """
    logits = torch.einsum("bsd,vd->bsv", continuous, embedding_weight)
    probs = F.softmax(logits / temperature, dim=-1)
    # Sample from categorical distribution
    B, S, V = probs.shape
    flat_probs = probs.reshape(-1, V)
    sampled = torch.multinomial(flat_probs, num_samples=1).squeeze(-1)
    return sampled.reshape(B, S)


def clamped_round(
    continuous: torch.Tensor,
    embedding_weight: torch.Tensor,
    clamp_value: float = 3.0,
) -> torch.Tensor:
    """Clamped rounding: clip embedding values before nearest-neighbor.

    The clamping trick from Diffusion-LM: embeddings that drift too far
    from the embedding manifold produce poor roundings. Clamping the
    continuous values to a reasonable range before rounding helps.

    Args:
        continuous: Predicted embeddings (batch, seq_len, embed_dim).
        embedding_weight: Embedding matrix (vocab_size, embed_dim).
        clamp_value: Maximum absolute value for clamping.

    Returns:
        Token IDs (batch, seq_len).
    """
    # Compute statistics of embedding table for adaptive clamping
    e_std = embedding_weight.std().item()
    clamp_range = clamp_value * e_std

    clamped = continuous.clamp(-clamp_range, clamp_range)
    logits = torch.einsum("bsd,vd->bsv", clamped, embedding_weight)
    return logits.argmax(dim=-1)


def projection_round(
    continuous: torch.Tensor,
    embedding_weight: torch.Tensor,
    n_steps: int = 5,
    step_size: float = 0.1,
) -> torch.Tensor:
    """Iterative projection toward the embedding manifold.

    Repeatedly project the continuous vector toward the nearest embedding
    while retaining some of the original prediction. This produces
    smoother transitions than hard rounding.

    Args:
        continuous: Predicted embeddings (batch, seq_len, embed_dim).
        embedding_weight: Embedding matrix (vocab_size, embed_dim).
        n_steps: Number of projection iterations.
        step_size: Interpolation step size toward nearest embedding.

    Returns:
        Token IDs (batch, seq_len).
    """
    x = continuous.clone()
    for _ in range(n_steps):
        # Find nearest embedding for each position
        logits = torch.einsum("bsd,vd->bsv", x, embedding_weight)
        nearest_ids = logits.argmax(dim=-1)  # (B, S)
        nearest_embs = embedding_weight[nearest_ids]  # (B, S, D)
        # Step toward the nearest embedding
        x = x + step_size * (nearest_embs - x)

    # Final hard rounding
    logits = torch.einsum("bsd,vd->bsv", x, embedding_weight)
    return logits.argmax(dim=-1)


def self_conditioning_round(
    continuous: torch.Tensor,
    embedding_weight: torch.Tensor,
    denoise_fn,
    t_values: list[float] | None = None,
) -> torch.Tensor:
    """Self-conditioning rounding: use the denoiser iteratively at low noise.

    Run additional denoising steps at very low noise levels to refine
    the embedding prediction before final rounding. The denoiser
    "self-conditions" by seeing its own previous output.

    Args:
        continuous: Predicted embeddings (batch, seq_len, embed_dim).
        embedding_weight: Embedding matrix (vocab_size, embed_dim).
        denoise_fn: Callable(x_t, t) -> x_0_pred. The denoiser.
        t_values: List of small t values for refinement steps.
            Defaults to [0.05, 0.02, 0.01].

    Returns:
        Token IDs (batch, seq_len).
    """
    if t_values is None:
        t_values = [0.05, 0.02, 0.01]

    x = continuous
    B = x.shape[0]
    device = x.device

    for t_val in t_values:
        t = torch.full((B,), t_val, device=device)
        # Add a tiny amount of noise
        noise = torch.randn_like(x) * (t_val ** 0.5)
        x_noisy = x + noise
        # Denoise again
        x = denoise_fn(x_noisy, t)

    # Final nearest-neighbor
    logits = torch.einsum("bsd,vd->bsv", x, embedding_weight)
    return logits.argmax(dim=-1)


def compute_rounding_accuracy(
    continuous: torch.Tensor,
    embedding_weight: torch.Tensor,
    true_ids: torch.Tensor,
    round_fn,
    **kwargs,
) -> float:
    """Measure what fraction of positions round to the correct token.

    Args:
        continuous: Predicted embeddings (batch, seq_len, embed_dim).
        embedding_weight: Embedding matrix (vocab_size, embed_dim).
        true_ids: Ground truth token IDs (batch, seq_len).
        round_fn: A rounding function to evaluate.
        **kwargs: Additional arguments for round_fn.

    Returns:
        Accuracy as a float in [0, 1].
    """
    predicted_ids = round_fn(continuous, embedding_weight, **kwargs)
    return (predicted_ids == true_ids).float().mean().item()
