# Lesson 04: CDCD -- Continuous Diffusion for Categorical Data

**Improving Diffusion-LM with a categorical loss that keeps predictions close to valid token embeddings.**

## Prerequisites

- Module 2, Lesson 03: Diffusion-LM from Scratch

## Learning Objective

After this lesson you will be able to explain the limitations of pure MSE training for text diffusion, implement the CDCD categorical projection loss, and train a model that produces denoised embeddings closer to valid tokens.

## Concept

### The Problem with Pure MSE

Diffusion-LM trains with MSE loss: `||x_0_pred - x_0||^2`. This works, but has a subtle flaw: **the model can minimize MSE by predicting a point between two token embeddings, rather than committing to one.**

Imagine two tokens "cat" and "hat" with embeddings e_cat and e_hat. If the model is uncertain, it might predict (e_cat + e_hat) / 2 -- which has low MSE to both but does not correspond to any real token. When we round this prediction, we get an arbitrary result.

### CDCD's Solution: Categorical Projection Loss

CDCD adds an auxiliary cross-entropy loss that forces the predicted embedding to "choose" a token:

```
Loss = MSE(x_0_pred, x_0) + lambda * CE(x_0_pred @ E^T, target_tokens)
```

where E is the embedding matrix. The cross-entropy term computes a softmax over dot-product similarities between the prediction and all token embeddings, then penalizes incorrect token choices.

This has two effects:
1. The model learns to produce predictions that are close to specific token embeddings, not averages.
2. At generation time, rounding is more reliable because predictions already lie near valid tokens.

### Implementation

```python
from src.cdcd import CDCD

model = CDCD(
    vocab_size=100,
    embed_dim=64,
    n_heads=4,
    n_layers=4,
    seq_len=32,
    categorical_weight=1.0,  # Weight for the CE loss
)

# The training loss includes both MSE and categorical terms
token_ids = torch.randint(0, 100, (8, 32))
loss = model.train_loss(token_ids)
```

### The Categorical Projection Loss in Detail

```python
def categorical_projection_loss(x_0_pred, target_ids, embedding_weight):
    # Compute logits: similarity of prediction to each token embedding
    logits = torch.einsum("bsd,vd->bsv", x_0_pred, embedding_weight)
    # Cross-entropy: the prediction should be most similar to the true token
    return F.cross_entropy(logits.reshape(-1, vocab_size), target_ids.reshape(-1))
```

This is essentially saying: "treat the denoiser output as a soft token prediction, and train it like a classifier." The MSE loss ensures the prediction is close in L2 distance; the CE loss ensures it is close in terms of token identity.

### Embedding Distance Loss (Alternative)

CDCD also motivates a simpler variant: penalize the minimum L2 distance between the prediction and any token embedding:

```python
def embedding_distance_loss(x_0_pred, embedding_weight):
    # For each prediction, find distance to nearest embedding
    dists = pairwise_distances(x_0_pred, embedding_weight)
    min_dist = dists.min(dim=-1).values
    return min_dist.mean()
```

This does not require knowing the target token -- it just pushes predictions onto the embedding manifold.

### Comparing Diffusion-LM vs CDCD

| Aspect | Diffusion-LM | CDCD |
|--------|-------------|------|
| Training loss | MSE only | MSE + categorical CE |
| Rounding quality | Can produce ambiguous predictions | Predictions align with token embeddings |
| Training cost | Slightly faster per step | Extra forward pass through embedding layer |
| Generation | Same reverse SDE | Same reverse SDE |

## Paper Link

- Dieleman et al., "Continuous Diffusion for Categorical Data" (2022) -- the categorical loss and score interpolation (Section 4).

## Exercises

1. **Compare losses**: Train both a `DiffusionLM` and a `CDCD` model (same architecture, same data, same epochs). After training, embed test tokens, add moderate noise (sigma=0.5), denoise with each model, and compare rounding accuracy.

2. **Categorical weight sweep**: Train CDCD with `categorical_weight` values of 0.1, 1.0, and 10.0. How does the weight affect (a) training convergence, (b) rounding accuracy, and (c) text quality?

3. **Visualize the embedding distance**: After training CDCD, generate samples and measure the average L2 distance between denoised embeddings and their nearest token embedding. Compare with Diffusion-LM. CDCD should produce smaller distances.

## What's Next

In [Lesson 05: Rounding and Decoding](../lesson05-rounding-decoding/), we will explore different strategies for converting continuous embeddings back to discrete tokens, from simple nearest-neighbor to iterative refinement methods.
