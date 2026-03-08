# Lesson 04: HuggingFace Bridge

## Prerequisites

- Completed Lessons 01-03 (training, controlled generation, infilling)
- A trained diffusion LM checkpoint from Lesson 01
- HuggingFace account (for pushing to Hub)
- `pip install transformers huggingface_hub safetensors`
- **Compute**: CPU is sufficient. No GPU needed for conversion and inference.

## Learning Objective

By the end of this lesson, you will be able to convert trained diffusion LM models to HuggingFace-compatible format, share them on the Hub, and use the HuggingFace ecosystem for inference with a clean pipeline interface.

## Concept

### Why Bridge to HuggingFace?

The HuggingFace ecosystem provides:
- **Model sharing**: Push your trained model to the Hub for others to use
- **Standardized format**: `config.json` + `model.safetensors` + tokenizer
- **Pipeline abstraction**: Simple `pipe("prompt")` interface for users
- **Community**: Discoverability, model cards, and version control

### Converting Models

Convert a training checkpoint to HuggingFace format:

```python
from hf_bridge import DiffusionLMConfig, convert_checkpoint_to_hf

config = DiffusionLMConfig(
    vocab_size=30522,
    d_model=512,
    nhead=8,
    num_layers=6,
    max_seq_len=128,
    mask_token_id=103,
    num_timesteps=1000,
    tokenizer_name="bert-base-uncased",
)

convert_checkpoint_to_hf(
    checkpoint_path="checkpoints/my_model/checkpoint_epoch50.pt",
    model_class=MDLMTransformer,
    config=config,
    output_dir="hf_model/",
    tokenizer=tokenizer,
)
```

This creates:
```
hf_model/
    config.json          # Model configuration
    model.safetensors    # Weights in safetensors format
    tokenizer/           # Tokenizer files
    README.md            # Auto-generated model card
```

### Loading Models

```python
from hf_bridge import load_model_from_hub

model, config = load_model_from_hub(
    model_class=MDLMTransformer,
    model_dir="hf_model/",
    device="cuda",
)
```

### Pushing to HuggingFace Hub

```python
from hf_bridge import push_to_hub

push_to_hub(
    model=model,
    config=config,
    repo_id="your-username/my-diffusion-lm",
    tokenizer=tokenizer,
    private=False,
)
# Model available at https://huggingface.co/your-username/my-diffusion-lm
```

### Generation Pipeline

The `DiffusionLMPipeline` provides a HuggingFace-style interface:

```python
from hf_generate import DiffusionLMPipeline

# Load from saved directory
pipe = DiffusionLMPipeline.from_pretrained("hf_model/", device="cuda")

# Unconditional generation
texts = pipe(num_samples=4, max_length=64, temperature=0.9)
for text in texts:
    print(text)

# Prompted generation
texts = pipe("Once upon a time", num_samples=4, max_length=128)

# Infilling
texts = pipe.infill(
    prefix="The scientist discovered",
    suffix="which changed everything.",
    infill_length=10,
    num_samples=4,
)

# Advanced sampling parameters
texts = pipe(
    num_samples=4,
    max_length=64,
    temperature=0.8,
    top_k=50,        # Top-k filtering
    top_p=0.95,      # Nucleus sampling
    num_steps=100,   # More denoising steps = higher quality
)
```

### Model Cards

The bridge automatically generates a model card with:
- Architecture details (dimensions, layers, parameters)
- Usage examples
- Citation information

You can customize it:

```python
from hf_bridge import save_model_for_hub

save_model_for_hub(
    model=model,
    config=config,
    output_dir="hf_model/",
    tokenizer=tokenizer,
    model_card="# My Custom Model Card\n\nTrained on poetry data...",
)
```

## Paper Reference

- HuggingFace Diffusers documentation: [huggingface.co/docs/diffusers](https://huggingface.co/docs/diffusers)
- Safetensors format: [huggingface.co/docs/safetensors](https://huggingface.co/docs/safetensors)

## Exercises

1. **Convert and Load**: Take a checkpoint from Lesson 01, convert it to HF format, then load it back and verify the model produces the same outputs.

2. **Pipeline Usage**: Create a `DiffusionLMPipeline` and experiment with different sampling parameters (`temperature`, `top_k`, `top_p`). Document how each parameter affects output quality and diversity.

3. **Model Card**: Write a custom model card for your trained model that includes training details, dataset description, and example outputs.

4. **Hub Upload** (optional): Push your model to HuggingFace Hub. Share the link with a classmate and have them load and use your model.

## What's Next

In [Lab 01](../lab01-end-to-end-project/), you will bring everything together in a capstone project: training a diffusion LM, implementing controlled generation, and packaging it as a reusable pipeline.
