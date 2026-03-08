# Lesson 02: Forward and Reverse SDE

**How to formally add and remove noise in continuous embedding space using stochastic differential equations.**

## Prerequisites

- Module 2, Lesson 01: From Tokens to Embedding Space

## Learning Objective

After this lesson you will be able to implement the variance-preserving (VP) SDE forward process that adds noise to token embeddings, compute closed-form noisy samples at any timestep, and take reverse SDE steps using a score function to denoise.

## Concept

### From Discrete Steps to Continuous Time

In Module 1, we used a discrete number of timesteps (e.g., T=1000) and added noise step by step. The continuous-time formulation using SDEs is more elegant: instead of discrete steps, noise is added continuously over time t in [0, 1].

The **forward SDE** describes how clean data x_0 is gradually corrupted:

```
dx = f(x, t) dt + g(t) dw
```

where `f` is the drift, `g` is the diffusion coefficient, and `dw` is an infinitesimal Wiener process (Brownian motion).

### The VP-SDE

The Variance-Preserving SDE uses:
- Drift: `f(x, t) = -0.5 * beta(t) * x`
- Diffusion: `g(t) = sqrt(beta(t))`
- Noise schedule: `beta(t) = beta_min + t * (beta_max - beta_min)`

The drift term shrinks x toward zero, while the diffusion term adds noise. Together, they transform any data distribution into a standard Gaussian N(0, I) as t goes to 1.

### Closed-Form Forward Marginal

The VP-SDE has a convenient closed form. At any time t, we can sample x_t directly without simulating the SDE step by step:

```
x_t = sqrt(alpha_bar(t)) * x_0 + sqrt(1 - alpha_bar(t)) * epsilon
```

where `epsilon ~ N(0, I)` and `alpha_bar(t) = exp(-0.5 * integral_0^t beta(s) ds)`.

```python
from src.sde import VPSDE

sde = VPSDE(beta_min=0.1, beta_max=20.0)

# Sample noisy embeddings at time t
x_0 = torch.randn(4, 32, 64)  # batch of embeddings
t = torch.tensor([0.1, 0.3, 0.5, 0.9])
x_t, noise, alpha_bar_t = sde.forward_marginal(x_0, t)

# alpha_bar decreases as t increases (more noise)
print(f"alpha_bar at t=0.1: {sde.alpha_bar(torch.tensor(0.1)):.3f}")  # ~0.995
print(f"alpha_bar at t=0.9: {sde.alpha_bar(torch.tensor(0.9)):.3f}")  # ~0.000
```

### The Score Function

The **score function** is the gradient of the log-density: `score(x, t) = nabla_x log p_t(x)`.

Intuitively, the score at any point x tells you: "which direction should I move to reach higher-density (more data-like) regions?" It is a vector field that points from noise toward clean data.

For the VP-SDE forward marginal, the score has a simple relationship to the noise:

```
score(x_t, t) = -epsilon / sqrt(1 - alpha_bar(t))
```

So if we train a model to predict the noise epsilon, we can compute the score. This is exactly what a denoiser does.

### The Reverse SDE

The reverse SDE runs time backward from t=1 to t=0, denoising the data:

```
dx = [-0.5 * beta(t) * x - beta(t) * score(x, t)] dt + sqrt(beta(t)) dw_bar
```

The key insight: **if we know the score function, we can reverse the noise process.** In practice, we train a neural network to approximate the score.

```python
# One reverse step (Euler-Maruyama discretization)
def reverse_step(sde, x_t, t, score, dt):
    beta_t = sde.beta(t)
    drift = -0.5 * beta_t * x_t - beta_t * score
    diffusion = torch.sqrt(beta_t)
    x_next = x_t + drift * dt  # dt is negative
    noise = torch.randn_like(x_t)
    x_next += diffusion * math.sqrt(abs(dt)) * noise
    return x_next
```

### Putting It Together: Forward and Reverse

```python
sde = VPSDE()

# Forward: embed -> noise
x_0 = embedder(token_ids)
x_t, noise, _ = sde.forward_marginal(x_0, t=torch.tensor([0.5]))

# Reverse: noise -> clean (with a trained score model)
# score = trained_model(x_t, t)
# x_denoised = sde.reverse_step(x_t, t, score, dt=-0.01)
```

## Paper Link

- Song et al., "Score-Based Generative Modeling through Stochastic Differential Equations" (2021) -- the VP-SDE framework (Section 3).
- This paper unified DDPM and score matching under the SDE framework.

## Exercises

1. **Visualize the forward process**: Use `VPSDE.forward_marginal` to noise a batch of embeddings at t = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]. Plot the mean and variance of x_t at each timestep. Verify that the mean shrinks to 0 and the variance approaches 1.

2. **Score from noise**: Given a known noise vector epsilon and alpha_bar, compute the score using `score_from_noise`. Verify that `noise_from_score(score_from_noise(noise, t), t)` recovers the original noise.

3. **Reverse trajectory with oracle score**: Use the ground-truth noise as the score (since we know epsilon in the forward pass). Run 100 reverse steps from x_1 back to x_0 and measure the MSE between the recovered x_0 and the original. How close does it get?

## What's Next

In [Lesson 03: Diffusion-LM from Scratch](../lesson03-diffusion-lm-from-scratch/), we will combine the embedding layer and the VP-SDE into a complete Diffusion-LM that can be trained on text and used to generate new sequences.
