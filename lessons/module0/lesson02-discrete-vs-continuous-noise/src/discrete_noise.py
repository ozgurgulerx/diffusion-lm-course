"""Discrete noise for text: randomly replace tokens.

In discrete diffusion, noise is added by replacing tokens in the sequence
with random tokens from the vocabulary (uniform corruption) or with a
special [MASK] token (absorbing/masked diffusion).

This module provides functions for both corruption styles at varying noise levels.
"""

import torch


def uniform_corrupt(
    token_ids: torch.Tensor,
    noise_level: float,
    vocab_size: int,
) -> torch.Tensor:
    """Corrupt a token sequence by replacing tokens with random vocabulary tokens.

    Each token is independently replaced with probability `noise_level`.
    The replacement is drawn uniformly from the full vocabulary.

    Args:
        token_ids: Clean token IDs of shape (seq_len,) or (batch, seq_len).
        noise_level: Probability of replacing each token, in [0, 1].
            0.0 = no corruption, 1.0 = fully random.
        vocab_size: Size of the vocabulary (exclusive upper bound for random IDs).

    Returns:
        Corrupted token IDs, same shape as input.
    """
    mask = torch.rand_like(token_ids, dtype=torch.float) < noise_level
    random_tokens = torch.randint_like(token_ids, low=0, high=vocab_size)
    return torch.where(mask, random_tokens, token_ids)


def mask_corrupt(
    token_ids: torch.Tensor,
    noise_level: float,
    mask_token_id: int,
) -> torch.Tensor:
    """Corrupt a token sequence by replacing tokens with a [MASK] token.

    This is the "absorbing state" corruption used in models like D3PM and MDLM.
    Each token is independently replaced with the mask token with probability
    `noise_level`.

    Args:
        token_ids: Clean token IDs of shape (seq_len,) or (batch, seq_len).
        noise_level: Probability of masking each token, in [0, 1].
        mask_token_id: The ID of the [MASK] token in the vocabulary.

    Returns:
        Corrupted token IDs, same shape as input.
    """
    mask = torch.rand_like(token_ids, dtype=torch.float) < noise_level
    return torch.where(mask, torch.full_like(token_ids, mask_token_id), token_ids)


def show_corruption_at_levels(
    token_ids: torch.Tensor,
    vocab: list[str],
    levels: list[float] | None = None,
    mode: str = "uniform",
    mask_token: str = "[MASK]",
) -> dict[float, str]:
    """Demonstrate corruption at multiple noise levels.

    Args:
        token_ids: 1-D tensor of clean token IDs.
        vocab: List mapping token ID -> token string.
        levels: Noise levels to demonstrate. Defaults to [0, 0.25, 0.5, 0.75, 1.0].
        mode: "uniform" or "mask".
        mask_token: String representation of the mask token (for display).

    Returns:
        Dict mapping noise_level -> corrupted text string.
    """
    if levels is None:
        levels = [0.0, 0.25, 0.5, 0.75, 1.0]

    vocab_size = len(vocab)
    # Add mask token to vocab for display if not present
    display_vocab = list(vocab) + [mask_token] if mask_token not in vocab else list(vocab)
    mask_token_id = len(vocab)  # ID for mask token (appended at end)

    results = {}
    for level in levels:
        if mode == "uniform":
            corrupted = uniform_corrupt(token_ids, level, vocab_size)
        else:
            corrupted = mask_corrupt(token_ids, level, mask_token_id)

        text = " ".join(display_vocab[i] for i in corrupted.tolist())
        results[level] = text

    return results


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    vocab = ["the", "cat", "sat", "on", "the", "mat"]
    # Use unique vocab for corruption
    unique_vocab = ["the", "cat", "sat", "on", "mat"]
    token_ids = torch.tensor([0, 1, 2, 3, 0, 4])  # "the cat sat on the mat"

    torch.manual_seed(42)
    print("=== Uniform corruption ===")
    results = show_corruption_at_levels(token_ids, unique_vocab, mode="uniform")
    for level, text in results.items():
        print(f"  noise={level:.2f}: {text}")

    torch.manual_seed(42)
    print("\n=== Mask corruption ===")
    results = show_corruption_at_levels(token_ids, unique_vocab, mode="mask")
    for level, text in results.items():
        print(f"  noise={level:.2f}: {text}")
