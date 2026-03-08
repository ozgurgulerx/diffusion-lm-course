"""Infilling sampler for diffusion language models using Repaint-style approach.

Infilling generates text for missing spans given surrounding context.
This is a natural strength of diffusion LMs: unlike autoregressive models
that can only generate left-to-right, diffusion models can generate
any subset of positions conditioned on the rest.

The Repaint approach (Lugmayr et al., 2022) works by:
1. At each denoising step, fix the known positions to their correct values
2. Denoise the unknown positions normally
3. Re-noise everything and repeat (the "repaint" trick for coherence)

References:
    - Lugmayr et al. (2022): "RePaint: Inpainting using Denoising Diffusion"
    - Sahoo et al. (2024): MDLM - https://arxiv.org/abs/2406.07524
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class RepaintScheduler:
    """Scheduler for the Repaint infilling algorithm.

    Controls the re-noising and denoising schedule. The key insight is that
    at each step, we re-noise and denoise multiple times (controlled by
    `resample_steps`) to improve coherence between known and generated regions.

    Args:
        num_timesteps: Total diffusion timesteps.
        resample_steps: Number of times to re-noise at each denoising step.
            More resampling = better coherence but slower generation.
        jump_length: How many steps to jump back during resampling.
    """

    def __init__(
        self,
        num_timesteps: int = 1000,
        resample_steps: int = 10,
        jump_length: int = 10,
    ):
        self.num_timesteps = num_timesteps
        self.resample_steps = resample_steps
        self.jump_length = jump_length

        # Linear mask rate schedule
        self.mask_rates = torch.linspace(0.0, 1.0, num_timesteps + 1)[1:]

    def get_schedule(self, num_inference_steps: int = 50) -> list[int]:
        """Generate the Repaint sampling schedule with jumps.

        Returns a sequence of timesteps that includes "jumps back" for
        resampling. For example, instead of [100, 80, 60, 40, 20, 0],
        we might get [100, 80, 90, 70, 80, 60, ...].

        Args:
            num_inference_steps: Base number of denoising steps.

        Returns:
            List of timesteps in the order they should be processed.
        """
        # Basic linear schedule
        base_steps = torch.linspace(
            self.num_timesteps - 1, 0, num_inference_steps, dtype=torch.long
        ).tolist()

        # Insert resampling jumps
        schedule = []
        i = 0
        while i < len(base_steps):
            current_t = base_steps[i]
            schedule.append(int(current_t))

            # Every few steps, jump back for resampling
            if (
                i > 0
                and i % (num_inference_steps // max(1, self.resample_steps)) == 0
                and current_t + self.jump_length < self.num_timesteps
            ):
                # Jump back
                jump_t = min(int(current_t) + self.jump_length, self.num_timesteps - 1)
                schedule.append(jump_t)

            i += 1

        return schedule

    def get_mask_rate(self, t: int) -> float:
        """Get the masking rate at timestep t.

        Args:
            t: Timestep index.

        Returns:
            Probability of each token being masked.
        """
        return self.mask_rates[min(t, len(self.mask_rates) - 1)].item()


class InfillingSampler:
    """Sampler for text infilling using the Repaint approach.

    Given a sequence with some positions marked as "to generate" and others
    as "known", generates text for the unknown positions while maintaining
    coherence with the known context.

    Args:
        model: Trained diffusion language model.
        mask_token_id: Token ID for [MASK].
        scheduler: RepaintScheduler instance.
        temperature: Sampling temperature.
    """

    def __init__(
        self,
        model: nn.Module,
        mask_token_id: int,
        scheduler: Optional[RepaintScheduler] = None,
        temperature: float = 1.0,
    ):
        self.model = model
        self.mask_token_id = mask_token_id
        self.scheduler = scheduler or RepaintScheduler()
        self.temperature = temperature

    def _mask_at_rate(
        self,
        x: torch.Tensor,
        rate: float,
        positions_to_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Apply random masking at a given rate to specified positions.

        Args:
            x: Current token IDs, shape (B, L).
            rate: Probability of masking each position.
            positions_to_mask: Boolean mask of positions eligible for masking.

        Returns:
            Masked token IDs.
        """
        x_out = x.clone()
        noise = torch.rand_like(x.float())
        should_mask = (noise < rate) & positions_to_mask
        x_out[should_mask] = self.mask_token_id
        return x_out

    @torch.no_grad()
    def infill(
        self,
        prefix: torch.Tensor,
        suffix: torch.Tensor,
        infill_length: int,
        batch_size: int = 1,
        num_steps: int = 50,
    ) -> torch.Tensor:
        """Generate text to fill between a prefix and suffix.

        Args:
            prefix: Token IDs for the beginning, shape (prefix_len,).
            suffix: Token IDs for the ending, shape (suffix_len,).
            infill_length: Number of tokens to generate in the middle.
            batch_size: Number of infilling variants to generate.
            num_steps: Number of denoising steps.

        Returns:
            Complete sequences with infilled middle, shape (batch_size, total_len).
        """
        device = next(self.model.parameters()).device
        prefix = prefix.to(device)
        suffix = suffix.to(device)

        prefix_len = prefix.shape[0]
        suffix_len = suffix.shape[0]
        total_len = prefix_len + infill_length + suffix_len

        # Build the known/unknown mask
        # known_mask: True for positions that are fixed (prefix and suffix)
        known_mask = torch.zeros(batch_size, total_len, dtype=torch.bool, device=device)
        known_mask[:, :prefix_len] = True
        known_mask[:, prefix_len + infill_length :] = True

        # Initialize: known positions get real tokens, unknown get [MASK]
        x = torch.full(
            (batch_size, total_len), self.mask_token_id,
            dtype=torch.long, device=device,
        )
        x[:, :prefix_len] = prefix.unsqueeze(0).expand(batch_size, -1)
        x[:, prefix_len + infill_length :] = suffix.unsqueeze(0).expand(batch_size, -1)

        attention_mask = torch.ones(batch_size, total_len, device=device)

        # Get the repaint schedule
        schedule = self.scheduler.get_schedule(num_steps)

        prev_t = schedule[0] if schedule else 0
        for step_idx, current_t in enumerate(schedule):
            t = torch.full((batch_size,), current_t, device=device, dtype=torch.float)

            # If we jumped backward (re-noising step), re-mask the unknown region
            if current_t > prev_t:
                rate = self.scheduler.get_mask_rate(current_t)
                x = self._mask_at_rate(x, rate, ~known_mask)
            else:
                # Forward denoising: predict and unmask
                logits = self.model(x, t, attention_mask)
                probs = F.softmax(logits / self.temperature, dim=-1)

                # Sample new tokens
                sampled = torch.multinomial(
                    probs.view(-1, probs.shape[-1]), num_samples=1
                ).view(batch_size, total_len)

                # Only update unknown masked positions
                is_unknown_masked = (~known_mask) & (x == self.mask_token_id)

                if current_t > 0:
                    next_rate = self.scheduler.get_mask_rate(current_t - 1) if current_t > 0 else 0
                    curr_rate = self.scheduler.get_mask_rate(current_t)
                    if curr_rate > 0:
                        unmask_frac = (curr_rate - next_rate) / curr_rate
                    else:
                        unmask_frac = 1.0
                    unmask = (torch.rand_like(x.float()) < unmask_frac) & is_unknown_masked
                else:
                    unmask = is_unknown_masked

                x = torch.where(unmask, sampled, x)

            # Always restore known positions
            x[:, :prefix_len] = prefix.unsqueeze(0).expand(batch_size, -1)
            x[:, prefix_len + infill_length :] = suffix.unsqueeze(0).expand(batch_size, -1)

            prev_t = current_t

        # Final: unmask any remaining [MASK] tokens in the infill region
        still_masked = (~known_mask) & (x == self.mask_token_id)
        if still_masked.any():
            t = torch.zeros(batch_size, device=device)
            logits = self.model(x, t, attention_mask)
            preds = logits.argmax(dim=-1)
            x = torch.where(still_masked, preds, x)

        return x

    @torch.no_grad()
    def infill_with_mask(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        batch_size: int = 1,
        num_steps: int = 50,
    ) -> torch.Tensor:
        """Infill arbitrary positions specified by a mask.

        More flexible than prefix/suffix infilling: can infill any
        combination of positions.

        Args:
            tokens: Full sequence with placeholder values at infill positions.
                Shape (seq_len,).
            mask: Boolean mask where True = position to generate, False = keep.
                Shape (seq_len,).
            batch_size: Number of variants to generate.
            num_steps: Denoising steps.

        Returns:
            Completed sequences, shape (batch_size, seq_len).
        """
        device = next(self.model.parameters()).device
        tokens = tokens.to(device)
        mask = mask.to(device)
        seq_len = tokens.shape[0]

        # known_mask: True for positions to keep
        known_mask = (~mask).unsqueeze(0).expand(batch_size, -1)

        # Initialize
        x = tokens.unsqueeze(0).expand(batch_size, -1).clone()
        x[:, mask] = self.mask_token_id

        attention_mask = torch.ones(batch_size, seq_len, device=device)
        schedule = self.scheduler.get_schedule(num_steps)

        prev_t = schedule[0] if schedule else 0
        for current_t in schedule:
            t = torch.full((batch_size,), current_t, device=device, dtype=torch.float)

            if current_t > prev_t:
                # Re-noise step
                rate = self.scheduler.get_mask_rate(current_t)
                x = self._mask_at_rate(x, rate, ~known_mask)
            else:
                # Denoise step
                logits = self.model(x, t, attention_mask)
                probs = F.softmax(logits / self.temperature, dim=-1)
                sampled = torch.multinomial(
                    probs.view(-1, probs.shape[-1]), num_samples=1
                ).view(batch_size, seq_len)

                is_unknown_masked = (~known_mask) & (x == self.mask_token_id)

                if current_t > 0:
                    next_rate = self.scheduler.get_mask_rate(current_t - 1)
                    curr_rate = self.scheduler.get_mask_rate(current_t)
                    unmask_frac = (curr_rate - next_rate) / max(curr_rate, 1e-8)
                    unmask = (torch.rand_like(x.float()) < unmask_frac) & is_unknown_masked
                else:
                    unmask = is_unknown_masked

                x = torch.where(unmask, sampled, x)

            # Restore known positions
            known_tokens = tokens.unsqueeze(0).expand(batch_size, -1)
            x = torch.where(known_mask, known_tokens, x)

            prev_t = current_t

        # Final unmasking
        still_masked = (~known_mask) & (x == self.mask_token_id)
        if still_masked.any():
            t = torch.zeros(batch_size, device=device)
            logits = self.model(x, t, attention_mask)
            preds = logits.argmax(dim=-1)
            x = torch.where(still_masked, preds, x)

        return x
