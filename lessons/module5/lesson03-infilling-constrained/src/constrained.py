"""Constrained sampling for diffusion language models.

Enforces hard constraints during the denoising process, such as:
- Certain positions must contain specific tokens
- Generated text must match a template pattern
- Specific words must appear in the output

Unlike soft guidance (classifier or plug-and-play), constrained sampling
directly forces tokens at designated positions, guaranteeing the constraints
are satisfied.

References:
    - Sahoo et al. (2024): MDLM - https://arxiv.org/abs/2406.07524
    - Li et al. (2022): "Diffusion-LM Improves Controllable Text Generation"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class TokenConstraint:
    """Defines a constraint that specific positions must contain specific tokens.

    Args:
        positions: List of sequence positions to constrain.
        token_ids: List of required token IDs at those positions.
            Must have the same length as positions.
    """

    def __init__(self, positions: list[int], token_ids: list[int]):
        assert len(positions) == len(token_ids), (
            "positions and token_ids must have the same length"
        )
        self.positions = positions
        self.token_ids = token_ids

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the constraint to a batch of sequences.

        Args:
            x: Token IDs, shape (B, L).

        Returns:
            Constrained token IDs with forced values at specified positions.
        """
        x_out = x.clone()
        for pos, tid in zip(self.positions, self.token_ids):
            if pos < x_out.shape[1]:
                x_out[:, pos] = tid
        return x_out

    def get_constrained_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Get a boolean mask of constrained positions.

        Args:
            seq_len: Sequence length.
            device: Torch device.

        Returns:
            Boolean tensor of shape (seq_len,), True at constrained positions.
        """
        mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
        for pos in self.positions:
            if pos < seq_len:
                mask[pos] = True
        return mask


class TemplateConstraint:
    """Defines a template constraint with fixed and free positions.

    A template looks like: ["The", None, None, "is", None, "."]
    where None indicates positions that should be generated.

    Args:
        template: List of token IDs or None for free positions.
    """

    def __init__(self, template: list[Optional[int]]):
        self.template = template
        self.fixed_positions = []
        self.fixed_tokens = []
        self.free_positions = []

        for i, token in enumerate(template):
            if token is not None:
                self.fixed_positions.append(i)
                self.fixed_tokens.append(token)
            else:
                self.free_positions.append(i)

    @property
    def seq_len(self) -> int:
        return len(self.template)

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        """Apply template constraints.

        Args:
            x: Token IDs, shape (B, L). L must equal len(template).

        Returns:
            Constrained sequences.
        """
        x_out = x.clone()
        for pos, tid in zip(self.fixed_positions, self.fixed_tokens):
            x_out[:, pos] = tid
        return x_out

    def get_free_mask(self, device: torch.device) -> torch.Tensor:
        """Get mask of positions that are free to generate.

        Returns:
            Boolean tensor where True = free to generate.
        """
        mask = torch.zeros(self.seq_len, dtype=torch.bool, device=device)
        for pos in self.free_positions:
            mask[pos] = True
        return mask


class KeywordConstraint:
    """Constraint that requires specific keywords to appear in the output.

    Unlike TokenConstraint, this doesn't fix positions -- it only requires
    that the keyword token IDs appear somewhere in the sequence. Uses a
    greedy projection step to enforce this.

    Args:
        keyword_token_ids: List of lists, each inner list is a keyword
            as a sequence of token IDs.
    """

    def __init__(self, keyword_token_ids: list[list[int]]):
        self.keyword_token_ids = keyword_token_ids

    def project(
        self,
        x: torch.Tensor,
        logits: torch.Tensor,
        mask_token_id: int,
    ) -> torch.Tensor:
        """Project the current generation to include required keywords.

        Finds the best positions to insert each keyword by looking at
        model confidence (logits) and current [MASK] positions.

        Args:
            x: Current token IDs, shape (B, L).
            logits: Model logits, shape (B, L, V).
            mask_token_id: Token ID for [MASK].

        Returns:
            Modified token IDs with keywords inserted.
        """
        x_out = x.clone()
        B, L = x.shape

        for batch_idx in range(B):
            for keyword in self.keyword_token_ids:
                kw_len = len(keyword)

                # Check if keyword already present
                seq = x_out[batch_idx].tolist()
                already_present = False
                for start in range(L - kw_len + 1):
                    if seq[start : start + kw_len] == keyword:
                        already_present = True
                        break

                if already_present:
                    continue

                # Find the best position to insert the keyword
                # Prefer positions that are currently masked
                best_pos = -1
                best_score = -float("inf")

                for start in range(L - kw_len + 1):
                    # Score: prefer masked positions, prefer positions where
                    # the model already assigns high probability to keyword tokens
                    pos_range = range(start, start + kw_len)
                    score = 0.0
                    for offset, pos in enumerate(pos_range):
                        if x_out[batch_idx, pos] == mask_token_id:
                            score += 1.0  # Bonus for masked positions
                        # Add logit score for the keyword token
                        score += logits[batch_idx, pos, keyword[offset]].item() * 0.1

                    if score > best_score:
                        best_score = score
                        best_pos = start

                # Insert keyword at best position
                if best_pos >= 0:
                    for offset, token_id in enumerate(keyword):
                        x_out[batch_idx, best_pos + offset] = token_id

        return x_out


class ConstrainedSampler:
    """Sampler that enforces hard constraints during denoising.

    At each denoising step:
    1. Run the diffusion model to get predictions
    2. Sample/unmask tokens normally
    3. Apply constraints to force certain tokens

    Supports TokenConstraint, TemplateConstraint, and KeywordConstraint.

    Args:
        model: Trained diffusion language model.
        mask_token_id: Token ID for [MASK].
        temperature: Sampling temperature.
    """

    def __init__(
        self,
        model: nn.Module,
        mask_token_id: int,
        temperature: float = 1.0,
    ):
        self.model = model
        self.mask_token_id = mask_token_id
        self.temperature = temperature

    @torch.no_grad()
    def sample_with_token_constraint(
        self,
        constraint: TokenConstraint,
        seq_len: int = 64,
        batch_size: int = 4,
        num_steps: int = 50,
        num_timesteps: int = 1000,
    ) -> torch.Tensor:
        """Generate text with fixed token constraints.

        Args:
            constraint: TokenConstraint specifying fixed positions/tokens.
            seq_len: Sequence length.
            batch_size: Number of sequences.
            num_steps: Denoising steps.
            num_timesteps: Total timesteps.

        Returns:
            Generated token IDs, shape (batch_size, seq_len).
        """
        device = next(self.model.parameters()).device
        mask_rates = torch.linspace(0.0, 1.0, num_timesteps + 1)[1:]

        # Start fully masked, then apply constraints
        x = torch.full(
            (batch_size, seq_len), self.mask_token_id,
            dtype=torch.long, device=device,
        )
        x = constraint.apply(x)

        constrained_mask = constraint.get_constrained_mask(seq_len, device)
        attention_mask = torch.ones(batch_size, seq_len, device=device)

        timesteps = torch.linspace(num_timesteps - 1, 0, num_steps, dtype=torch.long)

        for step_t in timesteps:
            t = torch.full((batch_size,), step_t.item(), device=device)
            logits = self.model(x, t.float(), attention_mask)

            # Sample tokens
            is_masked = x == self.mask_token_id
            probs = F.softmax(logits / self.temperature, dim=-1)
            sampled = torch.multinomial(
                probs.view(-1, probs.shape[-1]), num_samples=1
            ).view(batch_size, seq_len)

            # Unmasking rate
            if step_t > 0:
                curr = mask_rates[int(step_t.item())]
                nxt = mask_rates[max(0, int(step_t.item()) - 1)]
                unmask_prob = (curr - nxt) / max(curr, 1e-8)
            else:
                unmask_prob = 1.0

            unmask = (torch.rand_like(x.float()) < unmask_prob) & is_masked
            x = torch.where(unmask, sampled, x)

            # Re-apply constraints (ensure they are never overwritten)
            x = constraint.apply(x)

        # Final unmasking
        if (x == self.mask_token_id).any():
            t = torch.zeros(batch_size, device=device)
            logits = self.model(x, t, attention_mask)
            preds = logits.argmax(dim=-1)
            still_masked = x == self.mask_token_id
            x = torch.where(still_masked, preds, x)
            x = constraint.apply(x)

        return x

    @torch.no_grad()
    def sample_with_template(
        self,
        template: TemplateConstraint,
        batch_size: int = 4,
        num_steps: int = 50,
        num_timesteps: int = 1000,
    ) -> torch.Tensor:
        """Generate text matching a template.

        Args:
            template: TemplateConstraint defining fixed and free positions.
            batch_size: Number of sequences.
            num_steps: Denoising steps.
            num_timesteps: Total timesteps.

        Returns:
            Generated token IDs, shape (batch_size, seq_len).
        """
        device = next(self.model.parameters()).device
        seq_len = template.seq_len
        mask_rates = torch.linspace(0.0, 1.0, num_timesteps + 1)[1:]

        # Initialize with template
        x = torch.full(
            (batch_size, seq_len), self.mask_token_id,
            dtype=torch.long, device=device,
        )
        x = template.apply(x)

        free_mask = template.get_free_mask(device).unsqueeze(0).expand(batch_size, -1)
        attention_mask = torch.ones(batch_size, seq_len, device=device)

        timesteps = torch.linspace(num_timesteps - 1, 0, num_steps, dtype=torch.long)

        for step_t in timesteps:
            t = torch.full((batch_size,), step_t.item(), device=device)
            logits = self.model(x, t.float(), attention_mask)

            is_free_masked = free_mask & (x == self.mask_token_id)
            probs = F.softmax(logits / self.temperature, dim=-1)
            sampled = torch.multinomial(
                probs.view(-1, probs.shape[-1]), num_samples=1
            ).view(batch_size, seq_len)

            if step_t > 0:
                curr = mask_rates[int(step_t.item())]
                nxt = mask_rates[max(0, int(step_t.item()) - 1)]
                unmask_prob = (curr - nxt) / max(curr, 1e-8)
            else:
                unmask_prob = 1.0

            unmask = (torch.rand_like(x.float()) < unmask_prob) & is_free_masked
            x = torch.where(unmask, sampled, x)

            # Enforce template
            x = template.apply(x)

        # Final unmasking for free positions
        if (x == self.mask_token_id).any():
            t = torch.zeros(batch_size, device=device)
            logits = self.model(x, t, attention_mask)
            preds = logits.argmax(dim=-1)
            still_masked = x == self.mask_token_id
            x = torch.where(still_masked, preds, x)
            x = template.apply(x)

        return x

    @torch.no_grad()
    def sample_with_keywords(
        self,
        keyword_constraint: KeywordConstraint,
        seq_len: int = 64,
        batch_size: int = 4,
        num_steps: int = 50,
        num_timesteps: int = 1000,
        project_every: int = 5,
    ) -> torch.Tensor:
        """Generate text that must contain specific keywords.

        Periodically projects the current generation to include the required
        keywords. Between projections, normal denoising continues.

        Args:
            keyword_constraint: KeywordConstraint with required keywords.
            seq_len: Sequence length.
            batch_size: Number of sequences.
            num_steps: Denoising steps.
            num_timesteps: Total timesteps.
            project_every: Apply keyword projection every N steps.

        Returns:
            Generated token IDs, shape (batch_size, seq_len).
        """
        device = next(self.model.parameters()).device
        mask_rates = torch.linspace(0.0, 1.0, num_timesteps + 1)[1:]

        x = torch.full(
            (batch_size, seq_len), self.mask_token_id,
            dtype=torch.long, device=device,
        )
        attention_mask = torch.ones(batch_size, seq_len, device=device)

        timesteps = torch.linspace(num_timesteps - 1, 0, num_steps, dtype=torch.long)

        for step_idx, step_t in enumerate(timesteps):
            t = torch.full((batch_size,), step_t.item(), device=device)
            logits = self.model(x, t.float(), attention_mask)

            is_masked = x == self.mask_token_id
            probs = F.softmax(logits / self.temperature, dim=-1)
            sampled = torch.multinomial(
                probs.view(-1, probs.shape[-1]), num_samples=1
            ).view(batch_size, seq_len)

            if step_t > 0:
                curr = mask_rates[int(step_t.item())]
                nxt = mask_rates[max(0, int(step_t.item()) - 1)]
                unmask_prob = (curr - nxt) / max(curr, 1e-8)
            else:
                unmask_prob = 1.0

            unmask = (torch.rand_like(x.float()) < unmask_prob) & is_masked
            x = torch.where(unmask, sampled, x)

            # Periodically project to include keywords
            if (step_idx + 1) % project_every == 0:
                x = keyword_constraint.project(x, logits, self.mask_token_id)

        # Final unmasking and keyword projection
        if (x == self.mask_token_id).any():
            t = torch.zeros(batch_size, device=device)
            logits = self.model(x, t, attention_mask)
            preds = logits.argmax(dim=-1)
            still_masked = x == self.mask_token_id
            x = torch.where(still_masked, preds, x)

        x = keyword_constraint.project(x, logits, self.mask_token_id)
        return x
