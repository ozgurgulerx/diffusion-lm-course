"""Generation pipeline using HuggingFace-style abstractions.

Provides a clean, user-friendly interface for generating text with
diffusion language models, following HuggingFace pipeline conventions.

Usage:
    pipe = DiffusionLMPipeline.from_pretrained("path/to/model")
    texts = pipe("Generate a short story", num_samples=4)
    for text in texts:
        print(text)
"""

import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer


class DiffusionLMPipeline:
    """HuggingFace-style pipeline for diffusion language model generation.

    Provides a simple interface for text generation that handles tokenization,
    denoising, and decoding internally.

    Args:
        model: Trained diffusion language model.
        tokenizer: HuggingFace tokenizer.
        mask_token_id: Token ID for [MASK].
        num_timesteps: Number of diffusion timesteps.
        device: Device to run on.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer,
        mask_token_id: int,
        num_timesteps: int = 1000,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.mask_token_id = mask_token_id
        self.num_timesteps = num_timesteps
        self.device = device
        self.mask_rates = torch.linspace(0.0, 1.0, num_timesteps + 1)[1:]

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str,
        model_class: type = None,
        device: str = "cpu",
    ) -> "DiffusionLMPipeline":
        """Load a pipeline from a saved model directory.

        Args:
            model_dir: Path to model directory (containing config.json and weights).
            model_class: Model class to instantiate. If None, tries to import
                MDLMTransformer from the training module.
            device: Device to load onto.

        Returns:
            Configured DiffusionLMPipeline ready for generation.
        """
        from hf_bridge import load_model_from_hub, DiffusionLMConfig

        if model_class is None:
            try:
                import sys
                parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                lesson01_src = os.path.join(parent, "lesson01-training-custom-data", "src")
                if lesson01_src not in sys.path:
                    sys.path.insert(0, lesson01_src)
                from train_custom import MDLMTransformer
                model_class = MDLMTransformer
            except ImportError:
                raise ImportError(
                    "Could not import MDLMTransformer. Please provide model_class."
                )

        model, config = load_model_from_hub(model_class, model_dir, device)

        # Load tokenizer
        tokenizer_dir = os.path.join(model_dir, "tokenizer")
        if os.path.exists(tokenizer_dir):
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
        else:
            tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)

        return cls(
            model=model,
            tokenizer=tokenizer,
            mask_token_id=config.mask_token_id,
            num_timesteps=config.num_timesteps,
            device=device,
        )

    def __call__(
        self,
        prompt: Optional[str] = None,
        num_samples: int = 1,
        max_length: int = 64,
        num_steps: int = 50,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
    ) -> list[str]:
        """Generate text, optionally conditioned on a prompt.

        Args:
            prompt: Optional text prompt. If provided, the generated text
                will start with the prompt tokens.
            num_samples: Number of text samples to generate.
            max_length: Maximum sequence length.
            num_steps: Number of denoising steps.
            temperature: Sampling temperature (higher = more random).
            top_k: If > 0, only sample from top-k tokens.
            top_p: If < 1.0, use nucleus (top-p) sampling.

        Returns:
            List of generated text strings.
        """
        return self.generate(
            prompt=prompt,
            num_samples=num_samples,
            max_length=max_length,
            num_steps=num_steps,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

    @torch.no_grad()
    def generate(
        self,
        prompt: Optional[str] = None,
        num_samples: int = 1,
        max_length: int = 64,
        num_steps: int = 50,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
    ) -> list[str]:
        """Generate text with full control over sampling parameters.

        Args:
            prompt: Optional text prompt.
            num_samples: Number of samples.
            max_length: Maximum length.
            num_steps: Denoising steps.
            temperature: Sampling temperature.
            top_k: Top-k filtering parameter.
            top_p: Top-p (nucleus) filtering parameter.

        Returns:
            List of generated text strings.
        """
        # Handle prompt
        prefix_len = 0
        prefix_ids = []
        if prompt is not None:
            prefix_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            prefix_len = len(prefix_ids)
            if prefix_len >= max_length:
                prefix_ids = prefix_ids[:max_length]
                prefix_len = max_length

        # Initialize fully masked
        x = torch.full(
            (num_samples, max_length), self.mask_token_id,
            dtype=torch.long, device=self.device,
        )

        # Set prompt tokens
        if prefix_len > 0:
            prefix_tensor = torch.tensor(prefix_ids, device=self.device)
            x[:, :prefix_len] = prefix_tensor.unsqueeze(0).expand(num_samples, -1)

        attention_mask = torch.ones(num_samples, max_length, device=self.device)

        # Denoising loop
        timesteps = torch.linspace(
            self.num_timesteps - 1, 0, num_steps, dtype=torch.long
        )

        for step_t in timesteps:
            t = torch.full(
                (num_samples,), step_t.item(), device=self.device, dtype=torch.float
            )
            logits = self.model(x, t, attention_mask)

            # Apply temperature
            logits = logits / max(temperature, 1e-8)

            # Apply top-k filtering
            if top_k > 0:
                logits = self._top_k_filter(logits, top_k)

            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                logits = self._top_p_filter(logits, top_p)

            # Sample
            probs = F.softmax(logits, dim=-1)
            sampled = torch.multinomial(
                probs.view(-1, probs.shape[-1]), num_samples=1
            ).view(num_samples, max_length)

            # Only update masked positions (not prompt)
            is_masked = x == self.mask_token_id

            # Determine unmasking rate
            if step_t > 0:
                curr = self.mask_rates[int(step_t.item())]
                nxt = self.mask_rates[max(0, int(step_t.item()) - 1)]
                unmask_prob = (curr - nxt) / max(curr, 1e-8)
            else:
                unmask_prob = 1.0

            unmask = (torch.rand_like(x.float()) < unmask_prob) & is_masked
            x = torch.where(unmask, sampled, x)

            # Restore prompt tokens
            if prefix_len > 0:
                prefix_tensor = torch.tensor(prefix_ids, device=self.device)
                x[:, :prefix_len] = prefix_tensor.unsqueeze(0).expand(num_samples, -1)

        # Final: unmask remaining
        if (x == self.mask_token_id).any():
            t = torch.zeros(num_samples, device=self.device)
            logits = self.model(x, t, attention_mask)
            preds = logits.argmax(dim=-1)
            still_masked = x == self.mask_token_id
            x = torch.where(still_masked, preds, x)

        # Decode
        results = []
        for i in range(num_samples):
            text = self.tokenizer.decode(
                x[i].cpu().tolist(), skip_special_tokens=True
            )
            results.append(text)

        return results

    @torch.no_grad()
    def infill(
        self,
        prefix: str,
        suffix: str,
        infill_length: int = 20,
        num_samples: int = 1,
        num_steps: int = 50,
        temperature: float = 1.0,
    ) -> list[str]:
        """Generate text to fill between a prefix and suffix.

        Args:
            prefix: Text before the gap.
            suffix: Text after the gap.
            infill_length: Number of tokens to generate in the gap.
            num_samples: Number of variants.
            num_steps: Denoising steps.
            temperature: Sampling temperature.

        Returns:
            List of complete texts (prefix + infill + suffix).
        """
        prefix_ids = self.tokenizer.encode(prefix, add_special_tokens=False)
        suffix_ids = self.tokenizer.encode(suffix, add_special_tokens=False)
        total_len = len(prefix_ids) + infill_length + len(suffix_ids)

        # Initialize
        x = torch.full(
            (num_samples, total_len), self.mask_token_id,
            dtype=torch.long, device=self.device,
        )

        # Set known positions
        prefix_t = torch.tensor(prefix_ids, device=self.device)
        suffix_t = torch.tensor(suffix_ids, device=self.device)
        x[:, : len(prefix_ids)] = prefix_t.unsqueeze(0)
        x[:, len(prefix_ids) + infill_length :] = suffix_t.unsqueeze(0)

        known_mask = torch.zeros(
            num_samples, total_len, dtype=torch.bool, device=self.device
        )
        known_mask[:, : len(prefix_ids)] = True
        known_mask[:, len(prefix_ids) + infill_length :] = True

        attention_mask = torch.ones(num_samples, total_len, device=self.device)
        timesteps = torch.linspace(
            self.num_timesteps - 1, 0, num_steps, dtype=torch.long
        )

        for step_t in timesteps:
            t = torch.full(
                (num_samples,), step_t.item(), device=self.device, dtype=torch.float
            )
            logits = self.model(x, t, attention_mask) / max(temperature, 1e-8)
            probs = F.softmax(logits, dim=-1)
            sampled = torch.multinomial(
                probs.view(-1, probs.shape[-1]), num_samples=1
            ).view(num_samples, total_len)

            is_free_masked = (~known_mask) & (x == self.mask_token_id)

            if step_t > 0:
                curr = self.mask_rates[int(step_t.item())]
                nxt = self.mask_rates[max(0, int(step_t.item()) - 1)]
                unmask_prob = (curr - nxt) / max(curr, 1e-8)
            else:
                unmask_prob = 1.0

            unmask = (torch.rand_like(x.float()) < unmask_prob) & is_free_masked
            x = torch.where(unmask, sampled, x)

            # Restore known tokens
            x[:, : len(prefix_ids)] = prefix_t.unsqueeze(0)
            x[:, len(prefix_ids) + infill_length :] = suffix_t.unsqueeze(0)

        # Final unmasking
        if (x == self.mask_token_id).any():
            t = torch.zeros(num_samples, device=self.device)
            logits = self.model(x, t, attention_mask)
            preds = logits.argmax(dim=-1)
            x = torch.where(x == self.mask_token_id, preds, x)

        results = []
        for i in range(num_samples):
            text = self.tokenizer.decode(
                x[i].cpu().tolist(), skip_special_tokens=True
            )
            results.append(text)
        return results

    @staticmethod
    def _top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
        """Zero out logits outside the top-k."""
        top_k_vals = logits.topk(k, dim=-1).values
        threshold = top_k_vals[..., -1:]
        return logits.masked_fill(logits < threshold, float("-inf"))

    @staticmethod
    def _top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
        """Nucleus (top-p) filtering."""
        sorted_logits, sorted_indices = torch.sort(
            logits, descending=True, dim=-1
        )
        cumulative_probs = torch.cumsum(
            F.softmax(sorted_logits, dim=-1), dim=-1
        )

        # Remove tokens with cumulative probability above the threshold
        sorted_mask = (
            cumulative_probs - F.softmax(sorted_logits, dim=-1) >= p
        )
        sorted_logits[sorted_mask] = float("-inf")

        # Scatter back
        return sorted_logits.scatter(-1, sorted_indices, sorted_logits)
