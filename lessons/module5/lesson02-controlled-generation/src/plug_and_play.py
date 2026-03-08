"""Plug-and-play guidance for diffusion language models.

Uses any differentiable constraint function to steer generation, without
requiring any training of the constraint. The constraint is applied at
inference time by computing gradients through soft token representations.

Examples of plug-and-play constraints:
- Sentiment: use a pre-trained sentiment classifier
- Toxicity avoidance: use a toxicity detector
- Semantic similarity: use sentence embeddings
- Keyword inclusion: use soft matching loss

References:
    - Li et al. (2022): "Diffusion-LM Improves Controllable Text Generation"
    - PPLM (Dathathri et al., 2020): "Plug and Play Language Models"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Optional


class PlugAndPlaySampler:
    """Sampler that uses arbitrary differentiable constraints for guidance.

    At each denoising step:
    1. Get diffusion model's predicted logits
    2. Convert to soft token representations (Gumbel-softmax)
    3. Compute constraint loss on soft tokens
    4. Use gradient of constraint to shift the distribution

    The constraint function should accept soft token representations
    (B, L, V) and return a scalar loss to MINIMIZE (lower = better match).

    Args:
        diffusion_model: The denoising model.
        mask_token_id: Token ID for [MASK].
        guidance_scale: Strength of the constraint guidance.
        num_guidance_steps: Number of gradient steps per denoising step.
        step_size: Learning rate for gradient-based updates.
        temperature: Sampling temperature.
    """

    def __init__(
        self,
        diffusion_model: nn.Module,
        mask_token_id: int,
        guidance_scale: float = 5.0,
        num_guidance_steps: int = 3,
        step_size: float = 0.1,
        temperature: float = 1.0,
    ):
        self.diffusion_model = diffusion_model
        self.mask_token_id = mask_token_id
        self.guidance_scale = guidance_scale
        self.num_guidance_steps = num_guidance_steps
        self.step_size = step_size
        self.temperature = temperature

    def _apply_constraint(
        self,
        logits: torch.Tensor,
        constraint_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """Iteratively refine logits using the constraint function.

        Performs gradient descent on the logits to minimize the constraint loss.

        Args:
            logits: Current predicted logits, shape (B, L, V).
            constraint_fn: Function mapping soft tokens (B, L, V) -> scalar loss.

        Returns:
            Refined logits, shape (B, L, V).
        """
        # Work with a copy that requires gradients
        refined_logits = logits.detach().clone()
        refined_logits.requires_grad_(True)

        for _ in range(self.num_guidance_steps):
            # Convert to soft tokens
            soft_tokens = F.softmax(refined_logits / self.temperature, dim=-1)

            # Compute constraint loss
            loss = constraint_fn(soft_tokens)

            # Gradient step
            loss.backward()

            with torch.no_grad():
                # Update logits in the direction that reduces constraint loss
                refined_logits -= self.step_size * self.guidance_scale * refined_logits.grad
                refined_logits.grad.zero_()

        return refined_logits.detach()

    @torch.no_grad()
    def sample(
        self,
        constraint_fn: Callable[[torch.Tensor], torch.Tensor],
        seq_len: int = 64,
        batch_size: int = 4,
        num_steps: int = 50,
        num_timesteps: int = 1000,
    ) -> torch.Tensor:
        """Generate text satisfying a differentiable constraint.

        Args:
            constraint_fn: Differentiable function on soft tokens -> scalar loss.
            seq_len: Sequence length.
            batch_size: Number of sequences.
            num_steps: Denoising steps.
            num_timesteps: Total timesteps.

        Returns:
            Generated token IDs, shape (batch_size, seq_len).
        """
        device = next(self.diffusion_model.parameters()).device
        mask_rates = torch.linspace(0.0, 1.0, num_timesteps + 1)[1:]

        x = torch.full(
            (batch_size, seq_len), self.mask_token_id,
            dtype=torch.long, device=device,
        )
        attention_mask = torch.ones(batch_size, seq_len, device=device)
        timesteps = torch.linspace(num_timesteps - 1, 0, num_steps, dtype=torch.long)

        for step_t in timesteps:
            t = torch.full((batch_size,), step_t.item(), device=device)

            # Get diffusion model predictions
            logits = self.diffusion_model(x, t.float(), attention_mask)

            # Apply constraint guidance (with gradients enabled)
            with torch.enable_grad():
                guided_logits = self._apply_constraint(logits, constraint_fn)

            # Sample tokens
            is_masked = x == self.mask_token_id
            probs = F.softmax(guided_logits / self.temperature, dim=-1)
            sampled = torch.multinomial(
                probs.view(-1, probs.shape[-1]), num_samples=1
            ).view(batch_size, seq_len)

            # Unmasking schedule
            if step_t > 0:
                curr = mask_rates[int(step_t.item())]
                nxt = mask_rates[max(0, int(step_t.item()) - 1)]
                unmask_prob = (curr - nxt) / max(curr, 1e-8)
            else:
                unmask_prob = 1.0

            unmask = (torch.rand_like(x.float()) < unmask_prob) & is_masked
            x = torch.where(unmask, sampled, x)

        # Final unmasking
        if (x == self.mask_token_id).any():
            t = torch.zeros(batch_size, device=device)
            logits = self.diffusion_model(x, t, attention_mask)
            preds = logits.argmax(dim=-1)
            x = torch.where(x == self.mask_token_id, preds, x)

        return x


# ---------------------------------------------------------------------------
# Example constraint functions
# ---------------------------------------------------------------------------

def make_sentiment_constraint(
    sentiment_model: nn.Module,
    target_class: int = 1,
    embedding_matrix: Optional[torch.Tensor] = None,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Create a sentiment constraint function.

    Uses a pre-trained sentiment classifier to guide generation toward
    a target sentiment (e.g., positive).

    Args:
        sentiment_model: A classifier that takes token embeddings and returns logits.
        target_class: Desired sentiment class index.
        embedding_matrix: Token embedding matrix, shape (V, D).

    Returns:
        Constraint function: soft_tokens (B, L, V) -> scalar loss.
    """

    def constraint_fn(soft_tokens: torch.Tensor) -> torch.Tensor:
        # Convert soft tokens to embeddings
        if embedding_matrix is not None:
            embeddings = soft_tokens @ embedding_matrix  # (B, L, D)
        else:
            embeddings = soft_tokens

        # Get classifier prediction
        logits = sentiment_model(embeddings)
        log_probs = F.log_softmax(logits, dim=-1)

        # Minimize negative log probability of target class
        return -log_probs[:, target_class].mean()

    return constraint_fn


def make_keyword_constraint(
    target_token_ids: list[int],
    vocab_size: int,
    strength: float = 10.0,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Create a keyword inclusion constraint.

    Encourages the generated text to include specific tokens.

    Args:
        target_token_ids: List of token IDs that should appear.
        vocab_size: Total vocabulary size.
        strength: Weight for the constraint.

    Returns:
        Constraint function: soft_tokens (B, L, V) -> scalar loss.
    """

    def constraint_fn(soft_tokens: torch.Tensor) -> torch.Tensor:
        total_loss = torch.tensor(0.0, device=soft_tokens.device)

        for token_id in target_token_ids:
            # Max probability of this token across all positions
            token_probs = soft_tokens[:, :, token_id]  # (B, L)
            max_prob = token_probs.max(dim=1).values  # (B,)

            # Loss: penalize if max probability is low
            total_loss = total_loss - max_prob.mean()

        return strength * total_loss

    return constraint_fn


def make_length_constraint(
    target_length: int,
    pad_token_id: int,
    vocab_size: int,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Create a length constraint.

    Encourages the generated text to be close to a target length.

    Args:
        target_length: Desired number of non-padding tokens.
        pad_token_id: Token ID for padding.
        vocab_size: Total vocabulary size.

    Returns:
        Constraint function: soft_tokens (B, L, V) -> scalar loss.
    """

    def constraint_fn(soft_tokens: torch.Tensor) -> torch.Tensor:
        # Probability of NOT being a pad token at each position
        non_pad_prob = 1.0 - soft_tokens[:, :, pad_token_id]  # (B, L)
        expected_length = non_pad_prob.sum(dim=1)  # (B,)

        # MSE loss toward target length
        return ((expected_length - target_length) ** 2).mean()

    return constraint_fn
