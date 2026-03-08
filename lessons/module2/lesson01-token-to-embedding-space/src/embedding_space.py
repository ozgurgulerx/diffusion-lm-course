"""Token embedding utilities for continuous diffusion over text.

Provides a learnable embedding layer that maps discrete tokens into a
continuous vector space where Gaussian diffusion can operate smoothly.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenEmbedder(nn.Module):
    """Learnable embedding layer that maps token IDs to continuous vectors.

    This is the entry point for continuous diffusion: we embed discrete tokens
    into R^d, apply Gaussian noise in that space, and later round back.

    Args:
        vocab_size: Number of tokens in the vocabulary.
        embed_dim: Dimensionality of the embedding space.
        padding_idx: Optional padding token index (embedding fixed to zeros).
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        padding_idx: Optional[int] = None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        # Scale initialization so embeddings start with unit variance
        nn.init.normal_(self.embedding.weight, mean=0.0, std=1.0 / math.sqrt(embed_dim))
        if padding_idx is not None:
            nn.init.zeros_(self.embedding.weight[padding_idx])

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Map token IDs to embedding vectors.

        Args:
            token_ids: Integer tensor of shape (batch, seq_len).

        Returns:
            Embeddings of shape (batch, seq_len, embed_dim).
        """
        return self.embedding(token_ids)

    def get_all_embeddings(self) -> torch.Tensor:
        """Return the full embedding matrix (vocab_size, embed_dim)."""
        return self.embedding.weight

    def round_to_nearest(self, continuous: torch.Tensor) -> torch.Tensor:
        """Round continuous vectors to nearest token embedding (argmin L2).

        Args:
            continuous: Tensor of shape (batch, seq_len, embed_dim).

        Returns:
            Token IDs of shape (batch, seq_len).
        """
        # (batch, seq_len, embed_dim) vs (vocab_size, embed_dim)
        weight = self.embedding.weight  # (V, D)
        # Compute squared L2 distances
        # ||x - e||^2 = ||x||^2 - 2 x.e + ||e||^2
        x_sq = (continuous ** 2).sum(dim=-1, keepdim=True)  # (B, S, 1)
        e_sq = (weight ** 2).sum(dim=-1).unsqueeze(0).unsqueeze(0)  # (1, 1, V)
        dot = torch.einsum("bsd,vd->bsv", continuous, weight)  # (B, S, V)
        dists = x_sq - 2 * dot + e_sq  # (B, S, V)
        return dists.argmin(dim=-1)  # (B, S)

    def logits_from_embeddings(self, continuous: torch.Tensor) -> torch.Tensor:
        """Compute logits (negative distances) from continuous embeddings.

        Useful for computing probabilities over the vocabulary.

        Args:
            continuous: Tensor of shape (batch, seq_len, embed_dim).

        Returns:
            Logits of shape (batch, seq_len, vocab_size).
        """
        # Use dot product as logits (equivalent to softmax nearest-neighbor)
        return torch.einsum("bsd,vd->bsv", continuous, self.embedding.weight)


def compute_pairwise_distances(embeddings: torch.Tensor) -> torch.Tensor:
    """Compute pairwise L2 distances between embedding vectors.

    Args:
        embeddings: Tensor of shape (N, D).

    Returns:
        Distance matrix of shape (N, N).
    """
    diff = embeddings.unsqueeze(0) - embeddings.unsqueeze(1)  # (N, N, D)
    return torch.sqrt((diff ** 2).sum(dim=-1) + 1e-8)


def reduce_to_2d(embeddings: torch.Tensor, method: str = "pca") -> torch.Tensor:
    """Reduce embedding vectors to 2D for visualization.

    Args:
        embeddings: Tensor of shape (N, D).
        method: "pca" for PCA projection.

    Returns:
        2D coordinates of shape (N, 2).
    """
    if method == "pca":
        # Center the data
        centered = embeddings - embeddings.mean(dim=0, keepdim=True)
        # SVD-based PCA
        _, _, Vt = torch.linalg.svd(centered, full_matrices=False)
        return centered @ Vt[:2].T
    else:
        raise ValueError(f"Unknown method: {method}. Use 'pca'.")


def visualize_embeddings_2d(
    embeddings_2d: torch.Tensor,
    labels: Optional[list[str]] = None,
    title: str = "Token Embeddings (2D Projection)",
    ax: Optional[object] = None,
) -> object:
    """Plot 2D token embeddings with optional labels.

    Args:
        embeddings_2d: Tensor of shape (N, 2).
        labels: Optional list of N label strings.
        title: Plot title.
        ax: Optional matplotlib axes.

    Returns:
        The matplotlib axes object.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(8, 6))

    pts = embeddings_2d.detach().cpu().numpy()
    ax.scatter(pts[:, 0], pts[:, 1], alpha=0.6, s=30)

    if labels is not None:
        for i, label in enumerate(labels):
            ax.annotate(label, (pts[i, 0], pts[i, 1]), fontsize=7, alpha=0.8)

    ax.set_title(title)
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    return ax


def show_noisy_embeddings(
    embedder: TokenEmbedder,
    token_ids: torch.Tensor,
    noise_levels: list[float],
    labels: Optional[list[str]] = None,
) -> None:
    """Visualize how embeddings look under increasing Gaussian noise.

    Args:
        embedder: A TokenEmbedder instance.
        token_ids: 1D tensor of token IDs to embed and visualize.
        noise_levels: List of noise standard deviations to apply.
        labels: Optional token labels for annotation.
    """
    import matplotlib.pyplot as plt

    clean = embedder(token_ids.unsqueeze(0)).squeeze(0)  # (S, D)
    fig, axes = plt.subplots(1, len(noise_levels) + 1, figsize=(5 * (len(noise_levels) + 1), 4))

    # Clean embeddings
    pts_2d = reduce_to_2d(clean)
    visualize_embeddings_2d(pts_2d, labels=labels, title="Clean (sigma=0)", ax=axes[0])

    for i, sigma in enumerate(noise_levels):
        noisy = clean + sigma * torch.randn_like(clean)
        pts_2d = reduce_to_2d(noisy)
        visualize_embeddings_2d(
            pts_2d, labels=labels, title=f"Noisy (sigma={sigma:.2f})", ax=axes[i + 1]
        )

    plt.tight_layout()
    plt.show()
