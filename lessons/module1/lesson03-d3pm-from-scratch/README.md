# Lesson 03: D3PM from Scratch

Build a complete Discrete Denoising Diffusion Probabilistic Model (D3PM), train it on text, and generate samples.

## Prerequisites

- Lesson 01: Forward Corruption Process
- Lesson 02: Reverse Denoising Process

## Learning Objective

After this lesson you will be able to implement a full D3PM model with a transformer-based denoiser, train it on a small text dataset using the variational lower bound loss, and generate text samples via the reverse diffusion chain.

## Concept: Putting It All Together

In Lessons 01 and 02, we built the forward and reverse processes separately. Now we combine them into a complete model:

1. **Denoiser network:** A transformer that takes corrupted tokens `(x_t, t)` and predicts the clean tokens `x_0`.
2. **Training loss:** Cross-entropy between the predicted and true `x_0`, which bounds the variational lower bound (VLB).
3. **Sampling:** Run the reverse chain from `t=T` (fully corrupted) to `t=0` (clean).

### The Denoiser Architecture

```python
from src.d3pm_model import D3PMDenoiser

denoiser = D3PMDenoiser(
    vocab_size=50,        # vocabulary size
    d_model=128,          # hidden dimension
    n_heads=4,            # attention heads
    n_layers=4,           # transformer blocks
    max_seq_len=64,       # max sequence length
)

# Forward pass: predict x_0 from (x_t, t)
logits = denoiser(x_t, t)  # (batch, seq_len, vocab_size)
```

The denoiser combines:
- Token embeddings for the corrupted input
- Sinusoidal positional encoding for position
- Sinusoidal timestep embedding for the diffusion step
- A stack of pre-norm transformer blocks
- A linear output head

### The Training Loss

D3PM's variational lower bound decomposes into per-timestep KL divergences. In practice, we use a simpler and often more effective loss: directly predict `x_0` with cross-entropy.

```python
from src.d3pm_model import D3PM

d3pm = D3PM(
    denoiser=denoiser,
    vocab_size=50,
    num_timesteps=100,
    schedule="absorbing",
    mask_token_id=2,
)

# Training step
loss = d3pm.train_loss(x_0)  # x_0 shape: (batch, seq_len)
loss.backward()
```

### Sampling

Generation runs the reverse chain:

```python
samples = d3pm.sample(batch_size=4, seq_len=64, temperature=0.8)
```

Starting from all `[MASK]` tokens (absorbing) or random tokens (uniform), the model iteratively predicts and denoises until reaching clean text.

### Training on Real Text

```python
from shared.datasets.text import SimpleTokenizer, TextDataset, load_text_dataset

texts = load_text_dataset("tinystories", max_samples=5000)
tokenizer = SimpleTokenizer(texts, level="char")
dataset = TextDataset(texts, tokenizer, seq_len=64)
```

## Paper Reference

Austin, J., Johnson, D. D., Ho, J., Tarlow, D., & van den Berg, R. (2021). **Structured Denoising Diffusion Models in Discrete State-Spaces (D3PM)**. *NeurIPS 2021*. Section 4: Model and Training.

[https://arxiv.org/abs/2107.03006](https://arxiv.org/abs/2107.03006)

## Exercises

1. **Train and sample:** Train the D3PM on a small character-level dataset for 10 epochs. Generate 5 samples and inspect the quality.

2. **Compare schedules:** Train two D3PM models — one with "absorbing" and one with "uniform" schedule — on the same data. Compare sample quality.

3. **Vary model size:** Try d_model=64 vs d_model=256. How does model capacity affect generation quality for the same training budget?

## What's Next

In [Lesson 04: MDLM](../lesson04-mdlm/), we learn a simpler approach that uses only masking and achieves strong results with less complexity.
