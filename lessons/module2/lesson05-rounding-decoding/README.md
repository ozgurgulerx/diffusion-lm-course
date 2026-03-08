# Lesson 05: Rounding and Decoding

**Converting continuous embeddings back to discrete tokens -- the critical last step in continuous text diffusion.**

## Prerequisites

- Module 2, Lesson 03: Diffusion-LM from Scratch
- Module 2, Lesson 04: CDCD

## Learning Objective

After this lesson you will be able to implement and compare multiple rounding strategies (nearest-neighbor, clamped, projection-based, self-conditioning) and explain how each affects generation quality.

## Concept

### The Rounding Problem

After running the reverse SDE, we have continuous vectors in R^d. But we need discrete tokens. This "rounding" step is unique to continuous text diffusion -- image diffusion does not need it because pixels are already continuous (or can be trivially quantized).

Rounding errors are a major source of quality loss. A perfect denoiser with bad rounding produces bad text.

### Strategy 1: Nearest Neighbor (Baseline)

The simplest approach: for each position, find the token whose embedding is closest.

```python
def nearest_neighbor_round(continuous, embedding_weight):
    logits = torch.einsum("bsd,vd->bsv", continuous, embedding_weight)
    return logits.argmax(dim=-1)
```

This works well when the denoised embedding is very close to a token embedding. It fails when the prediction is ambiguous or noisy.

### Strategy 2: Clamped Rounding

The denoiser can produce embeddings with extreme values that are far from any token. Clamping clips these values before rounding:

```python
def clamped_round(continuous, embedding_weight, clamp_value=3.0):
    e_std = embedding_weight.std().item()
    clamped = continuous.clamp(-clamp_value * e_std, clamp_value * e_std)
    logits = torch.einsum("bsd,vd->bsv", clamped, embedding_weight)
    return logits.argmax(dim=-1)
```

This is a simple but effective trick from the Diffusion-LM paper. It prevents outlier predictions from dominating the rounding.

### Strategy 3: Softmax Sampling

Instead of hard argmax, sample from the softmax distribution over token similarities:

```python
def softmax_round(continuous, embedding_weight, temperature=0.5):
    logits = torch.einsum("bsd,vd->bsv", continuous, embedding_weight)
    probs = F.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs.reshape(-1, V), 1).reshape(B, S)
```

Lower temperature = more deterministic (approaches argmax). Higher temperature = more diverse but noisier.

### Strategy 4: Iterative Projection

Repeatedly move the prediction toward the nearest embedding:

```python
def projection_round(continuous, embedding_weight, n_steps=5, step_size=0.1):
    x = continuous.clone()
    for _ in range(n_steps):
        nearest_ids = nearest_neighbor_round(x, embedding_weight)
        nearest_embs = embedding_weight[nearest_ids]
        x = x + step_size * (nearest_embs - x)  # Step toward nearest
    return nearest_neighbor_round(x, embedding_weight)
```

This smoothly pulls ambiguous predictions toward valid tokens over multiple steps.

### Strategy 5: Self-Conditioning Rounding

Use the denoiser itself for iterative refinement at very low noise levels:

```python
def self_conditioning_round(continuous, embedding_weight, denoise_fn):
    x = continuous
    for t_val in [0.05, 0.02, 0.01]:
        noise = torch.randn_like(x) * (t_val ** 0.5)
        x = denoise_fn(x + noise, t=t_val)  # Re-denoise
    return nearest_neighbor_round(x, embedding_weight)
```

The idea: the denoiser has already learned to map noisy embeddings to clean ones. By running it a few extra times at very low noise, we refine the prediction.

### Comparing Strategies

```python
from src.compare_rounding import compare_rounding_strategies, print_comparison_table

results = compare_rounding_strategies(model, test_token_ids)
print_comparison_table(results)
```

Typical results (accuracy at different noise levels):

| Strategy         | s=0.0 | s=0.1 | s=0.5 | s=1.0 |
|-----------------|-------|-------|-------|-------|
| nearest_neighbor | 1.000 | 0.98  | 0.72  | 0.31  |
| clamped          | 1.000 | 0.98  | 0.75  | 0.35  |
| projection       | 1.000 | 0.99  | 0.78  | 0.38  |
| softmax t=0.5    | 0.998 | 0.97  | 0.70  | 0.29  |

Key takeaways:
- All strategies are nearly identical at low noise (the prediction is already close to a token).
- Clamping and projection help most at moderate noise.
- Softmax sampling trades accuracy for diversity.
- Self-conditioning is the most powerful but requires extra model forward passes.

## Paper Link

- Li et al., "Diffusion-LM Improves Controllable Text Generation" (2022), Section 3.2 -- clamping trick and rounding.
- Dieleman et al., "Continuous Diffusion for Categorical Data" (2022) -- analysis of rounding challenges.

## Exercises

1. **Rounding benchmark**: Train a Diffusion-LM, generate 100 samples using each rounding strategy. Manually inspect 10 samples from each. Which strategy produces the most coherent text?

2. **Temperature sweep**: Use `softmax_round` with temperatures [0.1, 0.3, 0.5, 1.0, 2.0]. For each, generate 20 samples and measure (a) the fraction of unique tokens used and (b) subjective coherence. What is the best temperature?

3. **Self-conditioning depth**: Implement self-conditioning with 1, 3, 5, and 10 refinement steps. Measure rounding accuracy on held-out data. Is there a point of diminishing returns?

## What's Next

You have now completed all lessons in Module 2. Head to [Lab 01: Controlled Generation](../lab01-controlled-generation/) to put everything together and build a classifier-guided text generation system using Diffusion-LM.
