"""Practical training utilities for discrete diffusion models.

Includes loss weighting strategies, learning rate schedules, and
sampling with temperature control.
"""

import math

import torch
import torch.nn.functional as F


def importance_weight_timesteps(
    t: torch.Tensor,
    num_timesteps: int,
    strategy: str = "uniform",
) -> torch.Tensor:
    """Compute importance weights for different timesteps.

    Different timesteps contribute differently to generation quality.
    Weighting them appropriately can improve training.

    Args:
        t: Sampled timesteps, shape (batch,).
        num_timesteps: Total number of timesteps T.
        strategy: Weighting strategy:
            - "uniform": Equal weight for all timesteps.
            - "snr": Weight proportional to signal-to-noise ratio change.
            - "truncated": Zero weight for very early timesteps.

    Returns:
        Weights tensor, shape (batch,).
    """
    t_normalized = t.float() / num_timesteps  # in [0, 1]

    if strategy == "uniform":
        return torch.ones_like(t_normalized)

    elif strategy == "snr":
        # Higher weight for intermediate timesteps where
        # the model has to do the most work
        weight = 4.0 * t_normalized * (1.0 - t_normalized)
        return weight.clamp(min=0.01)

    elif strategy == "truncated":
        # Zero weight for very early timesteps (t < 0.05 * T)
        weight = (t_normalized > 0.05).float()
        return weight

    else:
        raise ValueError(f"Unknown weighting strategy: {strategy}")


def sample_timesteps_with_importance(
    batch_size: int,
    num_timesteps: int,
    strategy: str = "uniform",
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Sample timesteps with importance sampling.

    Args:
        batch_size: Number of timesteps to sample.
        num_timesteps: Total number of timesteps T.
        strategy: Sampling strategy:
            - "uniform": Uniform sampling over [1, T].
            - "low_discrepancy": Stratified sampling for better coverage.

    Returns:
        Sampled timesteps, shape (batch_size,).
    """
    if strategy == "uniform":
        return torch.randint(1, num_timesteps + 1, (batch_size,), device=device)

    elif strategy == "low_discrepancy":
        # Stratified sampling: divide [1, T] into batch_size strata
        # and sample one timestep per stratum
        strata_size = num_timesteps / batch_size
        offsets = torch.arange(batch_size, device=device, dtype=torch.float)
        random_offsets = torch.rand(batch_size, device=device)
        t = ((offsets + random_offsets) * strata_size).long() + 1
        t = t.clamp(1, num_timesteps)
        # Shuffle to avoid ordering effects
        perm = torch.randperm(batch_size, device=device)
        return t[perm]

    else:
        raise ValueError(f"Unknown sampling strategy: {strategy}")


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.1,
):
    """Create a cosine annealing LR scheduler with linear warmup.

    Args:
        optimizer: The optimizer.
        num_warmup_steps: Number of warmup steps.
        num_training_steps: Total number of training steps.
        min_lr_ratio: Minimum LR as a fraction of the peak LR.

    Returns:
        torch.optim.lr_scheduler.LambdaLR scheduler.
    """
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            # Linear warmup
            return current_step / max(1, num_warmup_steps)
        # Cosine decay
        progress = (current_step - num_warmup_steps) / max(
            1, num_training_steps - num_warmup_steps
        )
        return max(min_lr_ratio, 0.5 * (1 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def sample_with_temperature(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> torch.Tensor:
    """Sample from logits with temperature, top-k, and top-p (nucleus) filtering.

    Args:
        logits: Raw logits, shape (..., vocab_size).
        temperature: Sampling temperature. Lower = more deterministic.
        top_k: If > 0, keep only the top k tokens.
        top_p: If < 1.0, keep smallest set of tokens with cumulative prob >= top_p.

    Returns:
        Sampled token IDs, shape (...).
    """
    original_shape = logits.shape[:-1]
    vocab_size = logits.shape[-1]
    logits = logits.reshape(-1, vocab_size)

    # Apply temperature
    if temperature != 1.0:
        logits = logits / temperature

    # Top-k filtering
    if top_k > 0:
        top_k = min(top_k, vocab_size)
        top_k_values, _ = torch.topk(logits, top_k, dim=-1)
        threshold = top_k_values[:, -1].unsqueeze(-1)
        logits[logits < threshold] = float("-inf")

    # Top-p (nucleus) filtering
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
        sorted_logits[sorted_mask] = float("-inf")

        # Scatter back to original order
        logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

    # Sample
    probs = F.softmax(logits, dim=-1)
    sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)

    return sampled.reshape(original_shape)


def compute_perplexity_proxy(
    model,
    x_0: torch.Tensor,
    mask_token_id: int,
    num_eval_timesteps: int = 10,
    device: torch.device = torch.device("cpu"),
) -> float:
    """Compute a perplexity-like metric for a discrete diffusion model.

    We evaluate the model's ability to predict masked tokens at various
    masking rates. This serves as a proxy for generation quality.

    Args:
        model: The denoiser model (takes x_t and t, returns logits).
        x_0: Clean token IDs, shape (batch, seq_len).
        mask_token_id: ID of the mask token.
        num_eval_timesteps: Number of masking rates to evaluate.
        device: Device.

    Returns:
        Average cross-entropy loss (lower is better).
    """
    model_was_training = model.training
    model.train(False)

    total_loss = 0.0
    n_evaluated = 0

    with torch.no_grad():
        for i in range(1, num_eval_timesteps + 1):
            mask_rate = i / (num_eval_timesteps + 1)
            batch_size, seq_len = x_0.shape

            # Mask tokens
            mask = torch.rand(batch_size, seq_len, device=device) < mask_rate
            x_t = x_0.clone()
            x_t[mask] = mask_token_id

            # Predict
            t = torch.full((batch_size,), mask_rate, device=device)
            logits = model(x_t, t)

            # CE on masked positions
            if mask.any():
                loss = F.cross_entropy(
                    logits[mask], x_0[mask].long(), reduction="mean"
                )
                total_loss += loss.item()
                n_evaluated += 1

    model.train(model_was_training)
    avg_loss = total_loss / max(n_evaluated, 1)
    return avg_loss
