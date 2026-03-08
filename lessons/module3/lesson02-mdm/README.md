# Lesson 02: MDM -- Masked Diffusion Model

## Prerequisites

- Lesson 01: MLM as diffusion (variable masking rates, iterative unmasking)
- Understanding of continuous-time Markov chains (helpful but not required)
- Basic probability (Bayes' rule, ELBO)

## Learning Objective

Implement MDM (Masked Diffusion Model) from scratch, understanding the continuous-time formulation where the masking rate gamma(t) defines a CTMC. Learn how the training loss simplifies to weighted cross-entropy on masked positions.

## Concept

### From Discrete Steps to Continuous Time

In Lesson 01, we used discrete timesteps. MDM formalizes this using a continuous-time Markov chain (CTMC):

- Each token independently transitions to [MASK] at **rate** beta(t)
- The probability of being masked by time t is: gamma(t) = 1 - exp(-integral_0^t beta(s) ds)
- The reverse process unmasks tokens, with rates derived from Bayes' rule

### The Forward Process

The CTMC transition rate from any token to [MASK] is beta(t). This gives a masking probability:

```python
class MaskingSchedule:
    def gamma(self, t):
        """Probability a token is masked at time t."""
        # Cosine schedule (same as lesson 01)
        f_t = cos((t + s) / (1 + s) * pi/2) ** 2
        return 1 - f_t / f_0

    def gamma_prime(self, t):
        """d/dt gamma(t) -- needed for loss weighting."""
        return (self.gamma(t + eps) - self.gamma(t - eps)) / (2 * eps)
```

### The Simplified Loss

Shi et al. (2024) show that the continuous-time ELBO simplifies to:

    L = E_t[ (gamma'(t) / gamma(t)) * E_{x_0, x_t}[ sum_{masked i} -log p_theta(x_0^i | x_t, t) ] ]

The key components:
1. **Sample t uniformly** from [0, 1]
2. **Corrupt** x_0 to x_t by masking with probability gamma(t)
3. **Predict** original tokens at masked positions
4. **Weight** the cross-entropy loss by gamma'(t) / gamma(t)

```python
def train_loss(self, x_0):
    t = torch.rand(B, device=x_0.device).clamp(0.01, 0.99)
    x_t, mask = self.forward_corrupt(x_0, t)
    logits = self.denoiser(x_t, t)

    # Cross-entropy on masked positions
    ce = F.cross_entropy(logits[mask], x_0[mask], reduction="none")

    # Weight by gamma'(t) / gamma(t)
    weight = gamma_prime(t) / gamma(t)
    loss = (ce * weight_per_token).mean()
    return loss
```

### Why the Weighting Matters

The weight gamma'(t)/gamma(t) ensures that:
- Early timesteps (low masking) contribute less (few tokens masked)
- Late timesteps (high masking) contribute proportionally more
- The total loss equals a proper ELBO on log-likelihood

Without this weighting, the model would spend equal effort on easy (low-masking) and hard (high-masking) cases.

### Sampling

Same iterative unmasking as Lesson 01, but now grounded in the reverse-time CTMC:

1. Start fully masked (t=1)
2. At each step, predict tokens and unmask the most confident ones
3. The number to unmask follows the schedule: proportional to gamma(t) - gamma(t-dt)

## Code

See `src/mdm.py`:
- `MaskingSchedule`: continuous-time schedule with gamma(t) and gamma'(t)
- `MDM`: full model with weighted `train_loss` and `sample`

## Paper Reference

Shi et al. "Simplified and Generalized Masked Diffusion for Discrete Data" (2024)
https://arxiv.org/abs/2406.04329

## Exercises

1. **Loss weighting ablation**: Train MDM with and without the gamma'(t)/gamma(t) weighting. Compare training loss curves and sample quality.

2. **Schedule comparison**: Implement a linear schedule (gamma(t) = t) and compare with cosine. Which converges faster?

3. **Verify the ELBO**: For a tiny vocabulary (V=4, L=3), compute the exact ELBO by enumeration and compare it with the Monte Carlo estimate from `train_loss`.

4. **Adaptive step sizes**: Modify `sample` to use non-uniform timestep spacing (more steps where gamma changes fastest). Does this improve sample quality for a fixed total number of steps?

## What's Next

Lesson 03 introduces SEDD, which takes a different approach: instead of predicting masked tokens, it estimates discrete scores (probability ratios).
