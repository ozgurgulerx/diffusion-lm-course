# Lesson 2: Discrete vs. Continuous Noise

Two approaches to corrupting text for diffusion: replace tokens randomly (discrete) or add Gaussian noise to embedding vectors (continuous).

## Prerequisites

- [Lesson 1: Autoregressive vs. Diffusion](../lesson01-autoregressive-vs-diffusion/)

## Learning Objective

After this lesson you will be able to implement both discrete and continuous noise processes for text, and explain the trade-offs between them.

## Concept

Diffusion models need a **forward process** that gradually converts clean data into noise. For images, this is straightforward: add Gaussian noise to pixel values. For text, there are two main strategies.

### Discrete Noise: Corrupt the Tokens

Text is inherently discrete (a sequence of token IDs). The most natural approach is to add noise in discrete space:

**Uniform corruption** — randomly replace tokens with tokens drawn uniformly from the vocabulary:

```python
def uniform_corrupt(token_ids, noise_level, vocab_size):
    mask = torch.rand_like(token_ids, dtype=torch.float) < noise_level
    random_tokens = torch.randint_like(token_ids, 0, vocab_size)
    return torch.where(mask, random_tokens, token_ids)
```

**Mask corruption** — replace tokens with a special `[MASK]` token (absorbing state):

```python
def mask_corrupt(token_ids, noise_level, mask_token_id):
    mask = torch.rand_like(token_ids, dtype=torch.float) < noise_level
    return torch.where(mask, mask_token_id, token_ids)
```

At `noise_level=0`, the sequence is clean. At `noise_level=1`, every token is replaced. The diffusion model learns to reverse this corruption.

### Continuous Noise: Corrupt the Embeddings

An alternative approach (used by Diffusion-LM) maps tokens into a continuous embedding space first, then adds standard Gaussian noise:

```python
def add_gaussian_noise(embeddings, noise_level):
    alpha = 1.0 - noise_level
    noise = torch.randn_like(embeddings)
    noisy = sqrt(alpha) * embeddings + sqrt(1 - alpha) * noise
    return noisy, noise
```

This reuses the well-understood math of continuous diffusion (DDPM, score matching) but requires a way to map back from continuous vectors to discrete tokens at the end.

### Comparing the Two Approaches

| Aspect | Discrete | Continuous |
|---|---|---|
| Noise space | Token IDs (integers) | Embedding vectors (floats) |
| Forward process | Replace tokens randomly | Add Gaussian noise to vectors |
| Math framework | Discrete Markov chains | Standard diffusion SDEs |
| Back to text | Direct (tokens are already discrete) | Requires rounding/projection |
| Example models | D3PM, MDLM, SEDD | Diffusion-LM |

Most recent work favors **discrete** diffusion because it avoids the rounding problem and achieves better perplexity, but understanding both approaches is essential.

## Paper Links

- [Structured Denoising Diffusion Models in Discrete State-Spaces (D3PM, Austin et al., 2021)](https://arxiv.org/abs/2107.03006)
- [Diffusion-LM Improves Controllable Text Generation (Li et al., 2022)](https://arxiv.org/abs/2205.14217)

## Exercises

1. **Corruption histogram**: For a fixed sentence, apply uniform corruption at noise_level=0.5 a hundred times. Plot a histogram of how often each position gets corrupted. Is it uniform?

2. **Embedding distance**: Using `continuous_noise.py`, compute the average L2 distance between clean and noisy embeddings at noise levels [0.0, 0.25, 0.5, 0.75, 1.0]. How does distance grow with noise level?

3. **Mask vs. uniform**: For the same noise level, which corruption type destroys more information — mask or uniform? Think about why. (Hint: with mask corruption, you at least know *which* positions are corrupted.)

## What's Next

[Lesson 3: The Denoising Objective](../lesson03-denoising-objective/) — training a model to reverse the noise process.
