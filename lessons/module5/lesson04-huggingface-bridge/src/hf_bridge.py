"""Bridge between scratch implementations and HuggingFace ecosystem.

Provides utilities to:
- Convert our custom model format to HuggingFace-compatible format
- Load models from HuggingFace Hub
- Push trained models to HuggingFace Hub
- Create model cards

This bridge lets you share your diffusion LM models with the community
and use pre-trained models from the Hub.

References:
    - HuggingFace Diffusers: https://huggingface.co/docs/diffusers
    - HuggingFace Hub: https://huggingface.co/docs/hub
"""

import json
import os
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn


class DiffusionLMConfig:
    """Configuration for a diffusion language model.

    Stores all hyperparameters needed to reconstruct the model architecture.
    Serializable to/from JSON for HuggingFace Hub compatibility.

    Args:
        vocab_size: Vocabulary size.
        d_model: Hidden dimension.
        nhead: Number of attention heads.
        num_layers: Number of Transformer layers.
        max_seq_len: Maximum sequence length.
        dropout: Dropout rate.
        mask_token_id: Absorbing state token ID.
        num_timesteps: Number of diffusion timesteps.
        model_type: Type of diffusion model (e.g., "mdlm", "d3pm").
        tokenizer_name: Name/path of the tokenizer.
    """

    def __init__(
        self,
        vocab_size: int = 30522,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        max_seq_len: int = 128,
        dropout: float = 0.1,
        mask_token_id: int = 103,
        num_timesteps: int = 1000,
        model_type: str = "mdlm",
        tokenizer_name: str = "bert-base-uncased",
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.mask_token_id = mask_token_id
        self.num_timesteps = num_timesteps
        self.model_type = model_type
        self.tokenizer_name = tokenizer_name

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "nhead": self.nhead,
            "num_layers": self.num_layers,
            "max_seq_len": self.max_seq_len,
            "dropout": self.dropout,
            "mask_token_id": self.mask_token_id,
            "num_timesteps": self.num_timesteps,
            "model_type": self.model_type,
            "tokenizer_name": self.tokenizer_name,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DiffusionLMConfig":
        """Deserialize from dictionary."""
        return cls(**{k: v for k, v in d.items() if k in cls.__init__.__code__.co_varnames})

    def save(self, path: str):
        """Save config to a JSON file."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "DiffusionLMConfig":
        """Load config from a JSON file."""
        with open(path, "r") as f:
            return cls.from_dict(json.load(f))


def save_model_for_hub(
    model: nn.Module,
    config: DiffusionLMConfig,
    output_dir: str,
    tokenizer=None,
    model_card: Optional[str] = None,
):
    """Save a trained model in HuggingFace Hub-compatible format.

    Creates the directory structure expected by the Hub:
        output_dir/
            config.json        - Model configuration
            model.safetensors  - Model weights (or pytorch_model.bin)
            tokenizer/         - Tokenizer files (if provided)
            README.md          - Model card

    Args:
        model: Trained PyTorch model.
        config: DiffusionLMConfig with model hyperparameters.
        output_dir: Directory to save the model.
        tokenizer: Optional HuggingFace tokenizer to save.
        model_card: Optional model card content (Markdown string).
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save config
    config.save(os.path.join(output_dir, "config.json"))

    # Save model weights
    try:
        from safetensors.torch import save_file

        state_dict = {k: v.contiguous() for k, v in model.state_dict().items()}
        save_file(state_dict, os.path.join(output_dir, "model.safetensors"))
    except ImportError:
        torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))

    # Save tokenizer
    if tokenizer is not None:
        tokenizer_dir = os.path.join(output_dir, "tokenizer")
        os.makedirs(tokenizer_dir, exist_ok=True)
        tokenizer.save_pretrained(tokenizer_dir)

    # Save model card
    if model_card is None:
        model_card = _generate_model_card(config)
    with open(os.path.join(output_dir, "README.md"), "w") as f:
        f.write(model_card)

    print(f"Model saved to {output_dir}")


def load_model_from_hub(
    model_class: type,
    model_dir: str,
    device: str = "cpu",
) -> tuple[nn.Module, DiffusionLMConfig]:
    """Load a model from a local HuggingFace Hub-format directory.

    Args:
        model_class: The model class to instantiate (e.g., MDLMTransformer).
        model_dir: Path to the saved model directory.
        device: Device to load the model onto.

    Returns:
        Tuple of (loaded model, config).
    """
    # Load config
    config = DiffusionLMConfig.load(os.path.join(model_dir, "config.json"))

    # Instantiate model
    model = model_class(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        max_seq_len=config.max_seq_len,
        dropout=config.dropout,
        mask_token_id=config.mask_token_id,
    )

    # Load weights
    safetensors_path = os.path.join(model_dir, "model.safetensors")
    bin_path = os.path.join(model_dir, "pytorch_model.bin")

    if os.path.exists(safetensors_path):
        from safetensors.torch import load_file

        state_dict = load_file(safetensors_path, device=device)
        model.load_state_dict(state_dict)
    elif os.path.exists(bin_path):
        state_dict = torch.load(bin_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
    else:
        raise FileNotFoundError(
            f"No model weights found in {model_dir}. "
            "Expected model.safetensors or pytorch_model.bin."
        )

    model = model.to(device)
    return model, config


def push_to_hub(
    model: nn.Module,
    config: DiffusionLMConfig,
    repo_id: str,
    tokenizer=None,
    commit_message: str = "Upload diffusion LM model",
    private: bool = False,
):
    """Push a trained model to HuggingFace Hub.

    Requires the `huggingface_hub` package and being logged in
    (run `huggingface-cli login` first).

    Args:
        model: Trained model.
        config: Model configuration.
        repo_id: Hub repository ID (e.g., "username/my-diffusion-lm").
        tokenizer: Optional tokenizer to include.
        commit_message: Git commit message for the upload.
        private: Whether to make the repository private.
    """
    from huggingface_hub import HfApi, create_repo

    # Create repo if it doesn't exist
    create_repo(repo_id, exist_ok=True, private=private)

    # Save locally first
    local_dir = f"/tmp/hf_upload_{repo_id.replace('/', '_')}"
    save_model_for_hub(model, config, local_dir, tokenizer)

    # Upload
    api = HfApi()
    api.upload_folder(
        folder_path=local_dir,
        repo_id=repo_id,
        commit_message=commit_message,
    )
    print(f"Model pushed to https://huggingface.co/{repo_id}")


def convert_checkpoint_to_hf(
    checkpoint_path: str,
    model_class: type,
    config: DiffusionLMConfig,
    output_dir: str,
    tokenizer=None,
):
    """Convert a training checkpoint to HuggingFace format.

    Training checkpoints contain optimizer state and other training metadata.
    This function extracts just the model weights and saves them in HF format.

    Args:
        checkpoint_path: Path to the training checkpoint (.pt file).
        model_class: Model class to instantiate.
        config: Model configuration.
        output_dir: Directory to save the HF-format model.
        tokenizer: Optional tokenizer.
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    # Extract model state dict
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    # Instantiate model and load weights
    model = model_class(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        max_seq_len=config.max_seq_len,
        dropout=config.dropout,
        mask_token_id=config.mask_token_id,
    )
    model.load_state_dict(state_dict)

    # Save in HF format
    save_model_for_hub(model, config, output_dir, tokenizer)

    training_info = {}
    if "epoch" in checkpoint:
        training_info["final_epoch"] = checkpoint["epoch"]
    if "global_step" in checkpoint:
        training_info["global_step"] = checkpoint["global_step"]

    if training_info:
        with open(os.path.join(output_dir, "training_info.json"), "w") as f:
            json.dump(training_info, f, indent=2)

    print(f"Converted checkpoint to HF format at {output_dir}")


def _generate_model_card(config: DiffusionLMConfig) -> str:
    """Generate a model card for the HuggingFace Hub.

    Args:
        config: Model configuration.

    Returns:
        Markdown string for the model card.
    """
    num_params_estimate = (
        config.vocab_size * config.d_model  # Token embeddings
        + config.max_seq_len * config.d_model  # Position embeddings
        + config.num_layers * (4 * config.d_model**2 + 8 * config.d_model)  # Transformer
        + config.vocab_size * config.d_model  # Output projection
    )

    return f"""---
tags:
- diffusion-lm
- text-generation
- {config.model_type}
library_name: pytorch
---

# Diffusion Language Model ({config.model_type.upper()})

A discrete diffusion language model trained for text generation.

## Model Details

- **Model type**: {config.model_type.upper()} (Masked Discrete Language Model)
- **Parameters**: ~{num_params_estimate:,}
- **Hidden dimension**: {config.d_model}
- **Attention heads**: {config.nhead}
- **Layers**: {config.num_layers}
- **Max sequence length**: {config.max_seq_len}
- **Vocabulary size**: {config.vocab_size}
- **Diffusion timesteps**: {config.num_timesteps}
- **Tokenizer**: {config.tokenizer_name}

## Usage

```python
from hf_bridge import load_model_from_hub, DiffusionLMConfig
from train_custom import MDLMTransformer

model, config = load_model_from_hub(MDLMTransformer, "path/to/model")
```

## Training

This model was trained using the diffusion-lm-course training pipeline.
See the [course repository](https://github.com/your-repo/diffusion-lm-course) for details.

## Citation

If you use this model, please cite the relevant diffusion LM papers:

```bibtex
@article{{sahoo2024simple,
  title={{Simple and Effective Masked Diffusion Language Models}},
  author={{Sahoo et al.}},
  journal={{arXiv preprint arXiv:2406.07524}},
  year={{2024}}
}}
```
"""
