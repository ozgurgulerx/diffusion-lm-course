"""Classifier-free guidance for diffusion language models.

Trains a single model that can generate both conditionally and unconditionally.
During training, the condition is randomly dropped with some probability.
At inference, we interpolate between conditional and unconditional predictions.

This avoids needing a separate classifier and often produces higher-quality results.

References:
    - Ho & Salimans (2022): "Classifier-Free Diffusion Guidance"
    - Zheng et al. (2023): "A Reparameterized Discrete Diffusion Model"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ClassifierFreeDiffusion(nn.Module):
    """Diffusion language model with classifier-free guidance support.

    The model takes an optional condition embedding. During training,
    the condition is dropped (replaced with a null embedding) with
    probability `cond_drop_prob`. At inference, we compute:

        guided_logits = (1 + w) * model(x_t, t, c) - w * model(x_t, t, null)

    where w is the guidance scale.

    Args:
        vocab_size: Vocabulary size.
        d_model: Hidden dimension.
        nhead: Number of attention heads.
        num_layers: Number of Transformer layers.
        num_classes: Number of condition classes (e.g., sentiment labels).
        max_seq_len: Maximum sequence length.
        cond_drop_prob: Probability of dropping condition during training.
        mask_token_id: Absorbing state token ID.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        num_classes: int = 2,
        max_seq_len: int = 128,
        cond_drop_prob: float = 0.1,
        mask_token_id: int = 103,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_classes = num_classes
        self.cond_drop_prob = cond_drop_prob
        self.mask_token_id = mask_token_id

        # Token and position embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

        # Timestep embedding
        self.time_embedding = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        # Condition embedding: num_classes + 1 for the null/unconditional token
        self.cond_embedding = nn.Embedding(num_classes + 1, d_model)
        self.null_cond_id = num_classes  # The last ID is the "no condition" token

        # Transformer backbone
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.final_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        force_uncond: bool = False,
    ) -> torch.Tensor:
        """Forward pass with optional conditioning.

        Args:
            x_t: Noisy token IDs, shape (B, L).
            t: Timesteps, shape (B,).
            condition: Class labels, shape (B,). None = unconditional.
            attention_mask: 1 for real tokens, shape (B, L).
            force_uncond: If True, always use null condition (for CFG inference).

        Returns:
            Token logits, shape (B, L, V).
        """
        B, L = x_t.shape
        device = x_t.device

        # Token + position embeddings
        positions = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
        h = self.token_embedding(x_t) + self.position_embedding(positions)

        # Timestep embedding
        t_emb = self.time_embedding(t.float().unsqueeze(-1))  # (B, D)

        # Condition embedding with random dropout during training
        if condition is None or force_uncond:
            cond_ids = torch.full((B,), self.null_cond_id, device=device, dtype=torch.long)
        else:
            cond_ids = condition.clone()
            if self.training:
                # Randomly drop condition
                drop_mask = torch.rand(B, device=device) < self.cond_drop_prob
                cond_ids[drop_mask] = self.null_cond_id

        cond_emb = self.cond_embedding(cond_ids)  # (B, D)

        # Add timestep and condition as bias
        h = h + (t_emb + cond_emb).unsqueeze(1)

        # Key padding mask
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = attention_mask == 0

        h = self.encoder(h, src_key_padding_mask=key_padding_mask)
        h = self.final_norm(h)
        logits = self.output_proj(h)
        return logits


class ClassifierFreeTrainer:
    """Trainer for classifier-free diffusion language models.

    Handles the training loop with condition dropout and MDLM-style masking.

    Args:
        model: ClassifierFreeDiffusion instance.
        num_timesteps: Number of diffusion timesteps.
        lr: Learning rate.
        device: Training device.
    """

    def __init__(
        self,
        model: ClassifierFreeDiffusion,
        num_timesteps: int = 1000,
        lr: float = 1e-4,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.device = device
        self.num_timesteps = num_timesteps
        self.mask_rates = torch.linspace(0.0, 1.0, num_timesteps + 1)[1:]
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    def train_step(
        self,
        x_0: torch.Tensor,
        condition: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, float]:
        """Single training step.

        Args:
            x_0: Clean token IDs, shape (B, L).
            condition: Class labels, shape (B,).
            attention_mask: Real token mask, shape (B, L).

        Returns:
            Metrics dictionary.
        """
        self.model.train()
        B, L = x_0.shape

        if attention_mask is None:
            attention_mask = torch.ones(B, L, device=self.device)

        x_0 = x_0.to(self.device)
        condition = condition.to(self.device)
        attention_mask = attention_mask.to(self.device)

        # Sample timesteps
        t_idx = torch.randint(0, self.num_timesteps, (B,), device=self.device)
        t = t_idx.float()

        # Apply masking
        mask_rate = self.mask_rates[t_idx].to(self.device).unsqueeze(1)
        noise = torch.rand(B, L, device=self.device)
        should_mask = (noise < mask_rate) & (attention_mask == 1)

        x_t = x_0.clone()
        x_t[should_mask] = self.model.mask_token_id

        # Forward pass (condition dropout happens inside the model)
        logits = self.model(x_t, t, condition, attention_mask)

        # Loss at masked positions only
        is_masked = should_mask
        if is_masked.sum() == 0:
            loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        else:
            loss = F.cross_entropy(logits[is_masked], x_0[is_masked])

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        return {"loss": loss.item()}


class ClassifierFreeSampler:
    """Sampler with classifier-free guidance.

    Generates text by running the model twice at each step (conditional
    and unconditional) and interpolating the predictions.

    Args:
        model: Trained ClassifierFreeDiffusion.
        guidance_scale: Guidance weight w. Higher = stronger conditioning.
        temperature: Sampling temperature.
    """

    def __init__(
        self,
        model: ClassifierFreeDiffusion,
        guidance_scale: float = 3.0,
        temperature: float = 1.0,
    ):
        self.model = model
        self.guidance_scale = guidance_scale
        self.temperature = temperature

    @torch.no_grad()
    def sample(
        self,
        condition: int,
        seq_len: int = 64,
        batch_size: int = 4,
        num_steps: int = 50,
        num_timesteps: int = 1000,
    ) -> torch.Tensor:
        """Generate text with classifier-free guidance.

        At each step computes:
            guided = (1 + w) * cond_logits - w * uncond_logits

        Args:
            condition: Target class index.
            seq_len: Sequence length.
            batch_size: Number of sequences.
            num_steps: Denoising steps.
            num_timesteps: Total timesteps.

        Returns:
            Generated token IDs, shape (batch_size, seq_len).
        """
        device = next(self.model.parameters()).device
        mask_rates = torch.linspace(0.0, 1.0, num_timesteps + 1)[1:]

        x = torch.full(
            (batch_size, seq_len), self.model.mask_token_id,
            dtype=torch.long, device=device,
        )
        attention_mask = torch.ones(batch_size, seq_len, device=device)
        cond_tensor = torch.full((batch_size,), condition, dtype=torch.long, device=device)

        timesteps = torch.linspace(num_timesteps - 1, 0, num_steps, dtype=torch.long)

        for step_t in timesteps:
            t = torch.full((batch_size,), step_t.item(), device=device)

            # Conditional prediction
            cond_logits = self.model(x, t, cond_tensor, attention_mask, force_uncond=False)

            # Unconditional prediction
            uncond_logits = self.model(x, t, cond_tensor, attention_mask, force_uncond=True)

            # Classifier-free guidance
            guided_logits = (
                (1 + self.guidance_scale) * cond_logits
                - self.guidance_scale * uncond_logits
            )

            # Sample at masked positions
            is_masked = x == self.model.mask_token_id
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
        if (x == self.model.mask_token_id).any():
            t = torch.zeros(batch_size, device=device)
            logits = self.model(x, t, cond_tensor, attention_mask)
            preds = logits.argmax(dim=-1)
            x = torch.where(x == self.model.mask_token_id, preds, x)

        return x
