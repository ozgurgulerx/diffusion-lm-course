"""Side-by-side comparison utilities for D3PM and MDLM.

Provides functions to train both models on the same data and compare
their samples, losses, and generation quality.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def compare_samples(
    d3pm_samples: list[str],
    mdlm_samples: list[str],
    reference_texts: list[str],
) -> dict:
    """Compare samples from D3PM and MDLM against reference texts.

    Computes simple quality metrics:
    - Average length
    - Character diversity (unique chars / total chars)
    - N-gram overlap with reference

    Args:
        d3pm_samples: List of generated strings from D3PM.
        mdlm_samples: List of generated strings from MDLM.
        reference_texts: List of reference training strings.

    Returns:
        Dictionary with comparison metrics.
    """
    def compute_stats(samples: list[str], name: str) -> dict:
        avg_len = sum(len(s) for s in samples) / max(len(samples), 1)
        all_chars = "".join(samples)
        char_diversity = len(set(all_chars)) / max(len(all_chars), 1)

        # Bigram overlap with reference
        ref_bigrams = set()
        for text in reference_texts:
            for i in range(len(text) - 1):
                ref_bigrams.add(text[i:i+2])

        sample_bigrams = set()
        for text in samples:
            for i in range(len(text) - 1):
                sample_bigrams.add(text[i:i+2])

        bigram_overlap = (
            len(sample_bigrams & ref_bigrams) / max(len(sample_bigrams), 1)
        )

        return {
            f"{name}_avg_length": avg_len,
            f"{name}_char_diversity": char_diversity,
            f"{name}_bigram_overlap": bigram_overlap,
        }

    results = {}
    results.update(compute_stats(d3pm_samples, "d3pm"))
    results.update(compute_stats(mdlm_samples, "mdlm"))

    return results


def print_comparison(
    d3pm_samples: list[str],
    mdlm_samples: list[str],
    reference_texts: list[str],
):
    """Print a formatted comparison of D3PM vs MDLM samples.

    Args:
        d3pm_samples: Generated strings from D3PM.
        mdlm_samples: Generated strings from MDLM.
        reference_texts: Reference training strings.
    """
    stats = compare_samples(d3pm_samples, mdlm_samples, reference_texts)

    print("=" * 60)
    print("D3PM vs MDLM Comparison")
    print("=" * 60)

    print("\nD3PM samples:")
    for i, s in enumerate(d3pm_samples[:5]):
        print(f"  [{i+1}] '{s}'")

    print("\nMDLM samples:")
    for i, s in enumerate(mdlm_samples[:5]):
        print(f"  [{i+1}] '{s}'")

    print("\nMetrics:")
    print(f"  {'Metric':<25} {'D3PM':>10} {'MDLM':>10}")
    print(f"  {'-'*25} {'-'*10} {'-'*10}")
    print(f"  {'Avg length':<25} {stats['d3pm_avg_length']:>10.1f} {stats['mdlm_avg_length']:>10.1f}")
    print(f"  {'Char diversity':<25} {stats['d3pm_char_diversity']:>10.3f} {stats['mdlm_char_diversity']:>10.3f}")
    print(f"  {'Bigram overlap w/ ref':<25} {stats['d3pm_bigram_overlap']:>10.3f} {stats['mdlm_bigram_overlap']:>10.3f}")
    print("=" * 60)


def evaluate_model_on_data(
    model,
    dataloader: DataLoader,
    mask_token_id: int,
    device: torch.device,
    model_type: str = "mdlm",
) -> float:
    """Evaluate a denoiser model on held-out data.

    Masks tokens and measures cross-entropy of predictions.

    Args:
        model: Denoiser model.
        dataloader: DataLoader for evaluation data.
        mask_token_id: Mask token ID.
        device: Device.
        model_type: "d3pm" or "mdlm" (affects time input format).

    Returns:
        Average cross-entropy loss.
    """
    was_training = model.training
    model.train(False)

    total_loss = 0.0
    n_batches = 0
    mask_rate = 0.5  # evaluate at 50% masking

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            batch_size, seq_len = batch.shape

            # Mask tokens
            mask = torch.rand(batch_size, seq_len, device=device) < mask_rate
            x_t = batch.clone()
            x_t[mask] = mask_token_id

            # Get predictions
            if model_type == "mdlm":
                t = torch.full((batch_size,), mask_rate, device=device)
            else:
                # D3PM uses integer timesteps
                t_int = int(mask_rate * 100)
                t = torch.full((batch_size,), t_int, device=device)

            logits = model(x_t, t)

            # CE on masked positions
            if mask.any():
                loss = F.cross_entropy(
                    logits[mask], batch[mask].long(), reduction="mean"
                )
                total_loss += loss.item()
                n_batches += 1

    model.train(was_training)
    return total_loss / max(n_batches, 1)
