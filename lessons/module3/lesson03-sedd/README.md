# Lesson 03: SEDD -- Score Entropy Discrete Diffusion

## Prerequisites

- Lesson 01-02: Masked diffusion models (forward masking, reverse unmasking)
- Module 2: Continuous diffusion and score matching (helpful for analogy)
- Understanding of score functions in continuous diffusion (grad log p)

## Learning Objective

Understand how SEDD estimates discrete "scores" (probability ratios) instead of directly predicting masked tokens. Learn the score entropy loss and how sampling uses a Tweedie-type correction to convert scores into reverse transition rates.

## Concept

### The Discrete Score

In continuous diffusion, the score is the gradient of the log-density: nabla_x log p(x). For discrete data, there is no gradient. Instead, SEDD defines the **concrete score**:

    s(x, t)_{i,y} = p(x_0^i = y | x_t) / p(x_0^i = x_t^i | x_t)

This tells us: "how much more likely is token y at position i compared to the current token?" This is the discrete analog of the continuous score.

### Why Scores Instead of Direct Prediction?

MDM predicts p(x_0 | x_t) directly. SEDD predicts **ratios** of probabilities. Advantages:

1. **Unnormalized**: scores don't need to sum to 1, so the network has more freedom
2. **Score matching theory**: connects discrete diffusion to the well-understood continuous score matching framework
3. **Better gradients**: the score entropy loss provides different gradient dynamics than cross-entropy

### The Score Entropy Loss

For the absorbing (mask) process, the true score at a masked position with original token x_0^i is:

    s*(x_t, t)_{i,y} = 1 if y == x_0^i, else 0

The score entropy loss trains the network to match this:

```python
def score_entropy_loss(self, x_0):
    # ... corrupt x_0 to x_t ...
    log_scores = self.score_net(x_t, t)  # (B, L, V)

    # At masked positions:
    log_s = log_scores[mask]  # (N_masked, V)

    # sum_{y != mask} exp(log_s_y)  -- encourages small scores for wrong tokens
    exp_sum = log_s[:, non_mask_tokens].exp().sum(dim=-1)

    # -log_s_{x_0}  -- encourages large score for correct token
    log_s_target = log_s.gather(1, targets.unsqueeze(1)).squeeze(1)

    loss = (exp_sum - log_s_target).mean()
    return loss
```

The loss has two terms:
- **exp_sum**: penalizes the model for assigning high scores to incorrect tokens
- **-log_s_target**: rewards the model for assigning high score to the correct token

### Sampling via Tweedie Correction

From the estimated scores, we reconstruct the denoising distribution:

    p_theta(x_0^i = y | x_t) proportional to exp(log_s_{i,y})

This is the discrete analog of the Tweedie formula in continuous diffusion. We then use iterative unmasking, same as MDM:

```python
# Convert log-scores to probabilities
probs = softmax(log_scores / temperature)
probs[:, :, MASK_ID] = 0  # don't sample [MASK]
probs = probs / probs.sum()  # renormalize
```

### SEDD vs MDM

| Aspect | MDM | SEDD |
|--------|-----|------|
| Output | p(x_0 \| x_t) directly | Score ratios s(x_t) |
| Loss | Cross-entropy | Score entropy |
| Normalization | Softmax (normalized) | Unnormalized ratios |
| Theory | ELBO on likelihood | Score matching |
| Sampling | Same iterative unmasking | Same (via Tweedie) |

## Code

See `src/sedd.py`:
- `AbsorbingSchedule`: masking schedule with alpha(t) and gamma(t)
- `ScoreTransformer`: outputs log-scores (B, L, V)
- `SEDD`: full model with `score_entropy_loss` and `sample`

## Paper Reference

Lou et al. "Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution" (2024)
https://arxiv.org/abs/2310.16834

## Exercises

1. **Score interpretation**: For a trained SEDD model, visualize the scores at masked positions. Verify that the highest-scoring token matches the most likely original token.

2. **Loss comparison**: Train the same architecture with both cross-entropy (MDM-style) and score entropy (SEDD-style) losses. Compare convergence speed and final loss values.

3. **Score entropy derivation**: Starting from the KL divergence between the true score s* and the model score s_theta, derive the score entropy loss. (Hint: expand the KL and drop terms that don't depend on theta.)

4. **Temperature sweep**: Generate samples at temperature 0.5, 0.8, 1.0, 1.2, 1.5. How does temperature affect the diversity-quality tradeoff for SEDD compared to MDM?

## What's Next

Lesson 04 directly compares masked diffusion (MDM) vs general discrete diffusion (D3PM), showing when each approach is preferable.
