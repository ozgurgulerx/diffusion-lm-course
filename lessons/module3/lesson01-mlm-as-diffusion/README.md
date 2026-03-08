# Lesson 01: Masked Language Modeling as Diffusion

## Prerequisites

- Module 0: Foundations (autoregressive vs diffusion, discrete noise, denoising objective)
- Module 1: Discrete diffusion basics (forward corruption, reverse denoising, D3PM)
- Module 2: Continuous diffusion for text
- Familiarity with BERT-style masked language modeling

## Learning Objective

Understand that masked language modeling (MLM) is a special case of discrete diffusion. By varying the masking rate, we turn a one-step MLM into a multi-step diffusion model where masking rate = noise level and unmasking = denoising.

## Concept

### MLM Is One-Step Diffusion

BERT masks ~15% of tokens and predicts them. If we think of this as a diffusion model:
- **Forward process**: mask tokens with probability gamma
- **Reverse process**: predict original tokens at masked positions
- **One step**: gamma is fixed at 0.15

### Variable Masking Rate = Multi-Step Diffusion

The key insight: if we vary gamma(t) from 0 to 1 as t goes from 0 to 1, we get a full diffusion model:

```python
def forward_corrupt(x_0, t, mask_token_id):
    """Forward process: mask each token with probability gamma(t)."""
    gamma_t = cosine_masking_schedule(t)  # gamma(t) in [0, 1]
    mask = torch.rand_like(x_0, dtype=torch.float) < gamma_t
    x_t = x_0.clone()
    x_t[mask] = mask_token_id
    return x_t, mask
```

Compare this to D3PM's forward process, which requires a full (V x V) transition matrix Q_t. Here, the forward process is trivially simple: each token independently stays or becomes [MASK].

### Why This Works

The forward process defines a distribution:

    q(x_t | x_0) = gamma(t) * delta(x_t = MASK) + (1 - gamma(t)) * delta(x_t = x_0)

This is an **absorbing state** Markov chain -- once a token becomes [MASK], it stays [MASK]. The reverse process learns to predict the original token at each masked position:

    p_theta(x_0 | x_t, t) = Transformer(x_t, t)

Training loss = cross-entropy on masked positions only.

### Sampling: Iterative Unmasking

Start from a fully masked sequence (t=1) and iteratively unmask:
1. Predict tokens at all masked positions
2. Unmask the most confident predictions
3. Repeat until all tokens are revealed

```python
# At each step, unmask a fraction of positions
for step in range(num_steps):
    logits = model(x_t, t)
    probs = softmax(logits)
    sampled = multinomial(probs)
    # Unmask most confident predictions
    confidence = probs.gather(sampled)
    top_k_positions = confidence.topk(n_to_unmask)
    x_t[top_k_positions] = sampled[top_k_positions]
```

## Code

See `src/mlm_diffusion.py` for the full implementation:
- `cosine_masking_schedule` / `linear_masking_schedule`: gamma(t) functions
- `TransformerDenoiser`: backbone network
- `MLMDiffusion`: complete model with `forward_corrupt`, `train_loss`, and `sample`

## Exercises

1. **Schedule comparison**: Plot cosine vs linear masking schedules. Which one spends more time at intermediate masking rates? Why might that matter?

2. **Connection to BERT**: Modify `MLMDiffusion` to use a fixed masking rate (gamma=0.15) and verify that `train_loss` becomes equivalent to standard MLM loss.

3. **Masking visualization**: Use `forward_corrupt` at t=0.1, 0.3, 0.5, 0.7, 0.9 on a real sentence. Observe how the sentence degrades.

4. **Number of sampling steps**: Generate samples with num_steps=5, 10, 25, 50, 100. How does sample quality change? What is the minimum number of steps for reasonable output?

## Paper Reference

This lesson builds the intuition for:
- Shi et al. "Simplified and Generalized Masked Diffusion for Discrete Data" (2024) -- formalized in Lesson 02

## What's Next

Lesson 02 formalizes this connection using continuous-time Markov chains (MDM), giving us a principled ELBO and proper loss weighting.
