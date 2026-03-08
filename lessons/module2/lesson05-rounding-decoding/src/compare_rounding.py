"""Comparison utility for rounding strategies.

Evaluates multiple rounding methods across different noise levels
and produces a summary table and plot.
"""

from typing import Optional

import torch
import torch.nn as nn

from .rounding import (
    clamped_round,
    compute_rounding_accuracy,
    nearest_neighbor_round,
    projection_round,
    softmax_round,
)


def compare_rounding_strategies(
    model: nn.Module,
    token_ids: torch.Tensor,
    noise_levels: Optional[list[float]] = None,
) -> dict[str, list[float]]:
    """Compare rounding strategies at various noise levels.

    Embeds tokens, adds Gaussian noise at each level, then measures
    rounding accuracy for each strategy.

    Args:
        model: A model with .token_embedding.weight and .embed() method.
        token_ids: Ground truth token IDs (batch, seq_len).
        noise_levels: List of noise standard deviations to test.

    Returns:
        Dictionary mapping strategy name to list of accuracies
        (one per noise level).
    """
    if noise_levels is None:
        noise_levels = [0.0, 0.1, 0.3, 0.5, 1.0, 2.0]

    weight = model.token_embedding.weight.detach()
    x_0 = model.embed(token_ids).detach()

    strategies = {
        "nearest_neighbor": lambda c, w: nearest_neighbor_round(c, w),
        "softmax_t=0.5": lambda c, w: softmax_round(c, w, temperature=0.5),
        "softmax_t=1.0": lambda c, w: softmax_round(c, w, temperature=1.0),
        "clamped_3.0": lambda c, w: clamped_round(c, w, clamp_value=3.0),
        "projection_5steps": lambda c, w: projection_round(c, w, n_steps=5),
    }

    results: dict[str, list[float]] = {name: [] for name in strategies}

    for sigma in noise_levels:
        if sigma == 0.0:
            noisy = x_0
        else:
            noisy = x_0 + sigma * torch.randn_like(x_0)

        for name, fn in strategies.items():
            acc = compute_rounding_accuracy(noisy, weight, token_ids, fn)
            results[name].append(acc)

    return results


def print_comparison_table(
    results: dict[str, list[float]],
    noise_levels: Optional[list[float]] = None,
) -> None:
    """Print a formatted comparison table.

    Args:
        results: Output from compare_rounding_strategies.
        noise_levels: Noise levels used.
    """
    if noise_levels is None:
        noise_levels = [0.0, 0.1, 0.3, 0.5, 1.0, 2.0]

    # Header
    header = f"{'Strategy':<25}" + "".join(f"{'s=' + str(s):<10}" for s in noise_levels)
    print(header)
    print("-" * len(header))

    for name, accs in results.items():
        row = f"{name:<25}" + "".join(f"{a:.3f}     " for a in accs)
        print(row)


def plot_comparison(
    results: dict[str, list[float]],
    noise_levels: Optional[list[float]] = None,
    ax: Optional[object] = None,
) -> object:
    """Plot rounding accuracy vs noise level for each strategy.

    Args:
        results: Output from compare_rounding_strategies.
        noise_levels: Noise levels used.
        ax: Optional matplotlib axes.

    Returns:
        Matplotlib axes.
    """
    import matplotlib.pyplot as plt

    if noise_levels is None:
        noise_levels = [0.0, 0.1, 0.3, 0.5, 1.0, 2.0]

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(8, 5))

    for name, accs in results.items():
        ax.plot(noise_levels, accs, marker="o", label=name)

    ax.set_xlabel("Noise Level (sigma)")
    ax.set_ylabel("Rounding Accuracy")
    ax.set_title("Rounding Strategy Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax
