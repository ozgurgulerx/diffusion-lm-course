"""Shared training loop and evaluation helpers."""

from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_loop(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: Any,
    device: torch.device,
    epochs: int = 10,
    log_every: int = 50,
) -> list[float]:
    """Generic training loop. Returns list of average losses per epoch.

    Args:
        model: The model to train.
        dataloader: Training data loader.
        optimizer: Optimizer instance.
        loss_fn: Callable(model, batch) -> loss tensor.
        device: Device to train on.
        epochs: Number of training epochs.
        log_every: Print loss every N steps.
    """
    model.train()
    epoch_losses = []

    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")
        for step, batch in enumerate(pbar):
            if isinstance(batch, (list, tuple)):
                batch = [b.to(device) if isinstance(b, torch.Tensor) else b for b in batch]
            elif isinstance(batch, torch.Tensor):
                batch = batch.to(device)

            loss = loss_fn(model, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            if (step + 1) % log_every == 0:
                pbar.set_postfix(loss=f"{total_loss / n_batches:.4f}")

        avg_loss = total_loss / max(n_batches, 1)
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch + 1}/{epochs} — avg loss: {avg_loss:.4f}")

    return epoch_losses


@torch.no_grad()
def compute_eval_loss(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: Any,
    device: torch.device,
) -> float:
    """Compute average loss on a dataset. Returns average loss."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        if isinstance(batch, (list, tuple)):
            batch = [b.to(device) if isinstance(b, torch.Tensor) else b for b in batch]
        elif isinstance(batch, torch.Tensor):
            batch = batch.to(device)

        loss = loss_fn(model, batch)
        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)
