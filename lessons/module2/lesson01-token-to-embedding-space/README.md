# Lesson 01: From Tokens to Embedding Space

**Why move from discrete tokens to continuous embeddings for diffusion?**

## Prerequisites

- Module 0: Foundations (all lessons)
- Module 1: Discrete Diffusion (all lessons and lab)

## Learning Objective

After this lesson you will be able to embed discrete tokens into a continuous vector space, visualize the embedding geometry, and explain why continuous space enables smoother diffusion than operating directly on token IDs.

## Concept

### The Problem with Discrete Diffusion

In Module 1, we built discrete diffusion models (D3PM, MDLM) that operate directly on token IDs. The forward process swaps tokens via a transition matrix, and the reverse process predicts which tokens to restore. This works, but it has a fundamental limitation: **there is no natural notion of "small perturbation" in discrete space.** Swapping token 42 for token 43 might change "cat" to "dog" -- a huge semantic jump, not a small step.

In continuous diffusion (for images), we add small amounts of Gaussian noise. The key insight is that small noise = small perturbation, and the denoising model can learn smooth, gradual refinement. We want that smoothness for text too.

### The Solution: Embed, Diffuse, Round Back

The pipeline for continuous text diffusion has three stages:

```
Tokens  -->  Embed  -->  Diffuse in R^d  -->  Round back  -->  Tokens
[42, 7, 13]  [v1,v2,v3]   [v1+noise, ...]    [nearest]     [42, 7, 13]
```

1. **Embed**: Map each token to a learned vector in R^d using an embedding table.
2. **Diffuse**: Add/remove Gaussian noise in this continuous space (lessons 02-03).
3. **Round**: Map the denoised continuous vectors back to discrete tokens by finding the nearest embedding (lesson 05).

### Implementing a Token Embedder

```python
import torch
import torch.nn as nn
import math

class TokenEmbedder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # Scale init so embeddings start with reasonable variance
        nn.init.normal_(self.embedding.weight, std=1.0 / math.sqrt(embed_dim))

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(token_ids)

    def round_to_nearest(self, continuous: torch.Tensor) -> torch.Tensor:
        weight = self.embedding.weight  # (V, D)
        logits = torch.einsum("bsd,vd->bsv", continuous, weight)
        return logits.argmax(dim=-1)
```

The embedding maps each token ID to a point in R^d. Similar tokens should end up nearby in this space -- and training will encourage this.

### What Embeddings Look Like

We can visualize token embeddings by projecting them to 2D with PCA:

```python
from src.embedding_space import TokenEmbedder, reduce_to_2d, visualize_embeddings_2d

embedder = TokenEmbedder(vocab_size=100, embed_dim=32)
all_embs = embedder.get_all_embeddings().detach()
pts_2d = reduce_to_2d(all_embs)
visualize_embeddings_2d(pts_2d, title="Random Init Embeddings")
```

At initialization, embeddings are roughly uniform. After training, semantically similar tokens cluster together -- this structure is what makes continuous diffusion work.

### Why Smoothness Matters

Adding Gaussian noise to an embedding produces a point that is still "close" to the original in a meaningful way:

```python
x_0 = embedder(torch.tensor([[5, 10, 15]]))  # Clean embeddings
x_noisy = x_0 + 0.1 * torch.randn_like(x_0)  # Small perturbation

# The noisy version still rounds back to the correct tokens
recovered = embedder.round_to_nearest(x_noisy)
# With small noise, recovered == [5, 10, 15]
```

As noise increases, at some point the noisy embedding is closer to a different token's embedding -- that is when errors start. The denoiser's job is to push the noisy embedding back toward the correct token's embedding.

## Paper Link

This embedding-based approach was introduced in:
- Li et al., "Diffusion-LM Improves Controllable Text Generation" (2022), Section 3.1 -- embedding and rounding.

## Exercises

1. **Rounding robustness**: Create a `TokenEmbedder` with `vocab_size=50, embed_dim=64`. Embed a batch of random token IDs, then add increasing Gaussian noise (sigma = 0.1, 0.5, 1.0, 2.0). At each level, round back and measure the fraction of tokens that match the original. At what noise level does accuracy drop below 50%?

2. **Embedding dimension effect**: Repeat exercise 1 with `embed_dim=8` vs `embed_dim=128`. How does the embedding dimension affect rounding robustness? Why?

3. **Visualization**: Use `show_noisy_embeddings()` to visualize a small set of token embeddings (pick 10-20 tokens) at noise levels [0.0, 0.3, 1.0, 3.0]. Observe how the clusters dissolve as noise increases.

## What's Next

In [Lesson 02: Forward and Reverse SDE](../lesson02-forward-reverse-sde/), we will formalize the noise-adding and denoising processes using stochastic differential equations (SDEs), building the mathematical framework for continuous diffusion in embedding space.
