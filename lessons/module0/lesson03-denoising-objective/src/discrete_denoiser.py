"""Discrete denoiser: predict original tokens from corrupted token IDs.

Training objective: given a corrupted sequence of token IDs, predict the
original (clean) token at each position. This is a standard cross-entropy
classification task — the model sees corrupted IDs and outputs logits over
the vocabulary for each position.

This is the training objective used in masked/absorbing diffusion models
like D3PM (Austin et al., 2021) and MDLM (Sahoo et al., 2024).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiscreteDenoiser(nn.Module):
    """Simple MLP that predicts clean tokens from corrupted token IDs.

    Architecture: Embedding -> Hidden -> ReLU -> Output logits.
    Each position is processed independently (no cross-position attention).
    """

    def __init__(self, vocab_size: int, embed_dim: int = 32, hidden_dim: int = 64):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, vocab_size),
        )

    def forward(self, corrupted_ids: torch.Tensor) -> torch.Tensor:
        """Predict clean-token logits from corrupted token IDs.

        Args:
            corrupted_ids: Corrupted token IDs, shape (batch, seq_len).

        Returns:
            Logits of shape (batch, seq_len, vocab_size).
        """
        h = self.embedding(corrupted_ids)  # (batch, seq_len, embed_dim)
        return self.net(h)  # (batch, seq_len, vocab_size)


def discrete_denoising_loss(
    model: DiscreteDenoiser,
    clean_ids: torch.Tensor,
    corrupt_fn,
) -> torch.Tensor:
    """Compute the discrete denoising loss for one batch.

    Steps:
    1. Sample a random noise level for the batch.
    2. Corrupt the clean tokens using corrupt_fn.
    3. Feed corrupted tokens to the model.
    4. Compute cross-entropy loss between predictions and clean tokens.

    Args:
        model: A DiscreteDenoiser instance.
        clean_ids: Clean token IDs of shape (batch, seq_len).
        corrupt_fn: Callable(token_ids, noise_level) -> corrupted_ids.

    Returns:
        Scalar loss tensor.
    """
    batch_size = clean_ids.shape[0]

    # Sample a random noise level per batch element
    noise_level = torch.rand(batch_size, 1, device=clean_ids.device)
    # Expand to match token shape for element-wise corruption
    noise_mask = torch.rand_like(clean_ids, dtype=torch.float) < noise_level

    corrupted = corrupt_fn(clean_ids, noise_mask)

    # Predict clean tokens
    logits = model(corrupted)  # (batch, seq_len, vocab_size)

    # Cross-entropy loss: flatten batch and seq dims
    loss = F.cross_entropy(
        logits.view(-1, model.vocab_size),
        clean_ids.view(-1),
    )
    return loss


def simple_uniform_corrupt(token_ids: torch.Tensor, noise_mask: torch.Tensor, vocab_size: int = 5) -> torch.Tensor:
    """Corrupt tokens where noise_mask is True with random vocab tokens."""
    random_tokens = torch.randint_like(token_ids, low=0, high=vocab_size)
    return torch.where(noise_mask, random_tokens, token_ids)


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(42)

    vocab_size = 5
    seq_len = 6
    model = DiscreteDenoiser(vocab_size)

    # Fake batch of clean data
    clean = torch.randint(0, vocab_size, (4, seq_len))

    def corrupt_fn(ids, mask):
        return simple_uniform_corrupt(ids, mask, vocab_size)

    loss = discrete_denoising_loss(model, clean, corrupt_fn)
    print(f"Discrete denoising loss: {loss.item():.4f}")
    print(f"(Random baseline should be ~log({vocab_size}) = {torch.tensor(float(vocab_size)).log().item():.4f})")
