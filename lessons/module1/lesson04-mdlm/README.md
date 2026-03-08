# Lesson 04: MDLM — Masked Discrete Language Model

MDLM simplifies discrete diffusion by using only masking transitions and a streamlined loss, often achieving better results with less complexity.

## Prerequisites

- Lesson 03: D3PM from Scratch (complete D3PM implementation, training, and sampling).

## Learning Objective

After this lesson you will be able to implement MDLM with its continuous-time masking schedule and simplified loss, and understand why masking-only diffusion is often preferred over general transition matrices.

## Concept: Simplifying Discrete Diffusion

D3PM defines the forward process using general transition matrices Q_t, supporting both uniform and absorbing corruption. MDLM makes a simplifying choice: **use only absorbing (mask) transitions**.

This means:
- The forward process just masks more tokens over time
- The model's job is to predict the original tokens at masked positions
- The loss simplifies to cross-entropy on masked positions

### The Masking Schedule

MDLM uses continuous time `t` in `[0, 1]`:
- `t = 0`: fully clean (no tokens masked)
- `t = 1`: fully masked (all tokens are [MASK])

The masking rate `gamma(t)` controls what fraction of tokens are masked at time `t`:

```python
# Cosine schedule (preferred — gradual at start, rapid at end)
gamma(t) = 1 - cos(t * pi / 2)

# Linear schedule (simpler)
gamma(t) = t
```

### The Forward Process

Masking is dead simple — each token is independently replaced with [MASK] with probability `gamma(t)`:

```python
from src.mdlm import MDLM

mdlm = MDLM(denoiser, vocab_size=50, mask_token_id=2)

# Mask at time t=0.5
x_t = mdlm.mask_tokens(x_0, t=torch.tensor([0.5]))
```

### The Loss

The MDLM loss is cross-entropy on masked positions only:

```python
loss = mdlm.train_loss(x_0)
```

This is simpler than D3PM's VLB because:
1. No need to compute posterior distributions
2. No need to track transition matrices
3. The model directly learns to fill in masks (like BERT)

### Sampling: Iterative Unmasking

Generation starts from fully masked and progressively unmasks:

```python
samples = mdlm.sample(batch_size=4, seq_len=64, temperature=0.8)
```

At each step:
1. The model predicts probabilities for all positions
2. A fraction of masked tokens are unmasked (sampled from the predictions)
3. Repeat until all tokens are unmasked

### Why MDLM Works Better

1. **Simpler optimization landscape:** Only one type of corruption to handle
2. **Efficient training:** Loss only on masked positions, no need to predict unchanged tokens
3. **Better sampling:** Unmasking schedule provides natural curriculum
4. **Connection to BERT:** Leverages insights from masked language modeling

## Paper Reference

Sahoo, S., Arriola, M., Schiff, Y., Gokaslan, A., Marroquin, E., Chiu, J. T., Rush, A., & Kuleshov, V. (2024). **Simple and Effective Masked Diffusion Language Models**. *NeurIPS 2024*.

[https://arxiv.org/abs/2406.07524](https://arxiv.org/abs/2406.07524)

## Exercises

1. **Compare schedules:** Train MDLM with linear vs. cosine masking schedule. Plot the masking rate curves and compare sample quality.

2. **Vary number of sampling steps:** Generate with 10, 50, and 200 sampling steps. How does step count affect quality?

3. **MDLM vs D3PM:** Train both on the same data with similar model sizes. Compare training speed and sample quality.

## What's Next

In [Lesson 05: Training and Sampling](../lesson05-training-and-sampling/), we dive into practical training details: learning rate schedules, loss weighting, and advanced sampling strategies.
