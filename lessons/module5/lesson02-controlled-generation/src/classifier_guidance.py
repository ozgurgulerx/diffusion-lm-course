"""Classifier-guided sampling for diffusion language models.

Trains a classifier on noisy (masked) data and uses its gradients to steer
the denoising process toward desired attributes (e.g., positive sentiment).

The key idea: at each denoising step, shift the predicted token distribution
using gradients from a classifier that operates on noisy inputs.

References:
    - Dhariwal & Nichol (2021): "Diffusion Models Beat GANs on Image Synthesis"
    - Li et al. (2022): "Diffusion-LM Improves Controllable Text Generation"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class NoisyClassifier(nn.Module):
    """Classifier trained on noisy (partially masked) text.

    This classifier must work on inputs at any noise level t, because
    during guided sampling we need classifier gradients at every denoising step.

    Architecture: token embeddings -> Transformer encoder -> pooling -> label.

    Args:
        vocab_size: Number of tokens in vocabulary.
        d_model: Hidden dimension.
        nhead: Number of attention heads.
        num_layers: Number of Transformer layers.
        num_classes: Number of classification labels.
        max_seq_len: Maximum sequence length.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 2,
        num_classes: int = 2,
        max_seq_len: int = 128,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.time_embedding = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, num_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Classify noisy text.

        Args:
            x: Token IDs (potentially masked), shape (B, L).
            t: Noise level / timestep, shape (B,).
            attention_mask: 1 for real tokens, shape (B, L).

        Returns:
            Class logits, shape (B, num_classes).
        """
        B, L = x.shape
        positions = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)
        h = self.token_embedding(x) + self.position_embedding(positions)

        # Inject timestep information
        t_emb = self.time_embedding(t.float().unsqueeze(-1))  # (B, D)
        h = h + t_emb.unsqueeze(1)

        # Key padding mask: True means ignore
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = attention_mask == 0

        h = self.encoder(h, src_key_padding_mask=key_padding_mask)

        # Mean pooling over non-padding tokens
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).float()
            pooled = (h * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
        else:
            pooled = h.mean(dim=1)

        return self.classifier_head(pooled)


class ClassifierGuidedSampler:
    """Sampler that uses classifier gradients to guide diffusion generation.

    During each denoising step:
    1. The diffusion model predicts p(x_0 | x_t)
    2. The classifier computes p(y | x_t)
    3. We shift the predicted distribution using grad_x log p(y | x_t)

    For discrete tokens, we use the Gumbel-softmax trick to get gradients
    through the discrete token selection.

    Args:
        diffusion_model: The denoising diffusion model.
        classifier: Trained NoisyClassifier.
        mask_token_id: Token ID for [MASK].
        guidance_scale: How strongly to follow classifier gradients.
            Higher = more controlled but less diverse.
        temperature: Sampling temperature for token selection.
    """

    def __init__(
        self,
        diffusion_model: nn.Module,
        classifier: NoisyClassifier,
        mask_token_id: int,
        guidance_scale: float = 3.0,
        temperature: float = 1.0,
    ):
        self.diffusion_model = diffusion_model
        self.classifier = classifier
        self.mask_token_id = mask_token_id
        self.guidance_scale = guidance_scale
        self.temperature = temperature

    @torch.no_grad()
    def _get_diffusion_logits(
        self, x_t: torch.Tensor, t: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Get denoising model predictions."""
        return self.diffusion_model(x_t, t, attention_mask)

    def _get_classifier_guidance(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        target_class: int,
        attention_mask: torch.Tensor,
        diffusion_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Compute classifier guidance signal for shifting token distributions.

        Uses the straight-through Gumbel-softmax estimator to get gradients
        through discrete token selections.

        Args:
            x_t: Current noisy tokens, shape (B, L).
            t: Current timestep, shape (B,).
            target_class: Desired class index.
            attention_mask: Mask for real tokens, shape (B, L).
            diffusion_logits: Predicted logits from diffusion model, shape (B, L, V).

        Returns:
            Guidance logits to add to diffusion predictions, shape (B, L, V).
        """
        # Use soft token representations for gradient computation
        soft_tokens = F.gumbel_softmax(
            diffusion_logits / self.temperature, tau=1.0, hard=False
        )
        soft_tokens.requires_grad_(True)

        # Get classifier prediction on soft tokens
        # We embed the soft tokens by multiplying with the embedding matrix
        token_embeds = soft_tokens @ self.classifier.token_embedding.weight

        B, L = x_t.shape
        positions = torch.arange(L, device=x_t.device).unsqueeze(0).expand(B, -1)
        h = token_embeds + self.classifier.position_embedding(positions)
        t_emb = self.classifier.time_embedding(t.float().unsqueeze(-1))
        h = h + t_emb.unsqueeze(1)

        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = attention_mask == 0

        h = self.classifier.encoder(h, src_key_padding_mask=key_padding_mask)

        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).float()
            pooled = (h * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
        else:
            pooled = h.mean(dim=1)

        class_logits = self.classifier.classifier_head(pooled)
        log_prob = F.log_softmax(class_logits, dim=-1)
        target_log_prob = log_prob[:, target_class].sum()

        # Backpropagate to get guidance gradients
        target_log_prob.backward()

        # The gradient w.r.t. soft_tokens tells us how to shift the distribution
        guidance = soft_tokens.grad  # (B, L, V)
        return guidance if guidance is not None else torch.zeros_like(diffusion_logits)

    @torch.no_grad()
    def sample(
        self,
        target_class: int,
        seq_len: int = 64,
        batch_size: int = 4,
        num_steps: int = 50,
        num_timesteps: int = 1000,
    ) -> torch.Tensor:
        """Generate text guided toward a target class.

        Args:
            target_class: Desired class index (e.g., 1 for positive sentiment).
            seq_len: Length of sequence to generate.
            batch_size: Number of sequences to generate.
            num_steps: Number of denoising steps.
            num_timesteps: Total diffusion timesteps.

        Returns:
            Generated token IDs, shape (batch_size, seq_len).
        """
        device = next(self.diffusion_model.parameters()).device
        mask_rates = torch.linspace(0.0, 1.0, num_timesteps + 1)[1:]

        # Start fully masked
        x = torch.full(
            (batch_size, seq_len), self.mask_token_id,
            dtype=torch.long, device=device,
        )
        attention_mask = torch.ones(batch_size, seq_len, device=device)
        timesteps = torch.linspace(num_timesteps - 1, 0, num_steps, dtype=torch.long)

        for step_t in timesteps:
            t = torch.full((batch_size,), step_t.item(), device=device)

            # Step 1: Get diffusion model predictions
            diffusion_logits = self._get_diffusion_logits(x, t.float(), attention_mask)

            # Step 2: Get classifier guidance
            with torch.enable_grad():
                guidance = self._get_classifier_guidance(
                    x, t, target_class, attention_mask, diffusion_logits.detach()
                )

            # Step 3: Combine predictions with guidance
            guided_logits = diffusion_logits + self.guidance_scale * guidance

            # Step 4: Sample tokens at masked positions
            is_masked = x == self.mask_token_id
            probs = F.softmax(guided_logits / self.temperature, dim=-1)
            sampled = torch.multinomial(
                probs.view(-1, probs.shape[-1]), num_samples=1
            ).view(batch_size, seq_len)

            # Determine unmasking rate
            if step_t > 0:
                current_rate = mask_rates[int(step_t.item())]
                next_rate = mask_rates[max(0, int(step_t.item()) - 1)]
                unmask_prob = (current_rate - next_rate) / max(current_rate, 1e-8)
            else:
                unmask_prob = 1.0

            unmask = (torch.rand_like(x.float()) < unmask_prob) & is_masked
            x = torch.where(unmask, sampled, x)

        # Final: unmask remaining
        if (x == self.mask_token_id).any():
            t = torch.zeros(batch_size, device=device)
            logits = self._get_diffusion_logits(x, t, attention_mask)
            preds = logits.argmax(dim=-1)
            x = torch.where(x == self.mask_token_id, preds, x)

        return x


def train_noisy_classifier(
    classifier: NoisyClassifier,
    train_data: list[tuple[torch.Tensor, int]],
    mask_token_id: int,
    num_timesteps: int = 1000,
    epochs: int = 10,
    lr: float = 1e-3,
    device: str = "cpu",
) -> NoisyClassifier:
    """Train a classifier on noisy (masked) inputs.

    For each training sample, we randomly choose a noise level t and
    mask the input accordingly. The classifier must learn to predict
    the label regardless of noise level.

    Args:
        classifier: NoisyClassifier instance.
        train_data: List of (token_ids, label) pairs.
        mask_token_id: Token ID for masking.
        num_timesteps: Number of diffusion timesteps.
        epochs: Training epochs.
        lr: Learning rate.
        device: Training device.

    Returns:
        Trained classifier.
    """
    classifier = classifier.to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)
    mask_rates = torch.linspace(0.0, 1.0, num_timesteps + 1)[1:]

    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0

        for token_ids, label in train_data:
            token_ids = token_ids.unsqueeze(0).to(device)
            label_tensor = torch.tensor([label], device=device)

            # Random noise level
            t_idx = torch.randint(0, num_timesteps, (1,))
            t = t_idx.float().to(device)
            rate = mask_rates[t_idx.item()]

            # Apply masking
            x_t = token_ids.clone()
            mask = torch.rand_like(x_t.float()) < rate
            x_t[mask] = mask_token_id

            # Forward and loss
            logits = classifier(x_t, t)
            loss = F.cross_entropy(logits, label_tensor)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (logits.argmax(-1) == label_tensor).sum().item()
            total += 1

        acc = correct / max(1, total)
        print(f"Epoch {epoch + 1}/{epochs} | Loss: {total_loss / total:.4f} | Acc: {acc:.3f}")

    return classifier
