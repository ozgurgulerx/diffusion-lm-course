# Lesson 01: The Forward Corruption Process

The forward process in discrete diffusion is a Markov chain that gradually corrupts clean token sequences into noise.

## Prerequisites

- Module 0: Understand the difference between autoregressive and diffusion models, discrete vs. continuous noise, and the denoising objective.

## Learning Objective

After this lesson you will be able to implement the forward corruption process for discrete diffusion, including uniform and absorbing-state transition matrices, and visualize how text degrades over diffusion timesteps.

## Concept: Corrupting Token Sequences Step by Step

In autoregressive models, we generate text left-to-right. In discrete diffusion, we instead learn to **reverse a corruption process**. This lesson focuses on the corruption (forward) direction.

### The Setup

Given a sequence of clean tokens `x_0 = [w_1, w_2, ..., w_L]` from a vocabulary of size `K`, the forward process defines a Markov chain:

```
x_0 -> x_1 -> x_2 -> ... -> x_T
```

At each step `t`, every token independently transitions according to a **transition matrix** `Q_t` of shape `(K, K)`, where `Q_t[i, j] = P(x_t = j | x_{t-1} = i)`.

### Two Corruption Schedules

**Uniform corruption:** Each token may be replaced by any random token:

```python
# Q_t for uniform corruption
Q_t = (1 - beta_t) * I + (beta_t / K) * ones(K, K)
```

With probability `(1 - beta_t)` the token stays the same; with probability `beta_t` it is replaced by a uniformly random token.

**Absorbing state corruption:** Each token may be replaced by a special `[MASK]` token:

```python
# Q_t for absorbing corruption
Q_t = (1 - beta_t) * I
Q_t[:, mask_id] += beta_t
```

With probability `(1 - beta_t)` the token stays; with probability `beta_t` it becomes `[MASK]`.

### Cumulative Corruption

The key insight is that we can skip intermediate steps. The cumulative transition matrix from time 0 to time `t` is:

```
Q_bar_t = Q_1 * Q_2 * ... * Q_t
```

For our schedules, this has a clean closed form. Let `alpha_bar_t = prod(1 - beta_s, s=1..t)`:

- **Uniform:** `Q_bar_t = alpha_bar_t * I + (1 - alpha_bar_t) / K * ones`
- **Absorbing:** `Q_bar_t = alpha_bar_t * I + (1 - alpha_bar_t) * e_mask`

This means we can sample `x_t` directly from `x_0` without simulating all intermediate steps:

```python
from src.forward_process import DiscreteForwardProcess

# Create forward process with absorbing schedule
fp = DiscreteForwardProcess(vocab_size=100, num_timesteps=1000, schedule="absorbing")

# Sample corrupted tokens at time t
x_0 = torch.tensor([[10, 20, 30, 40, 50]])  # clean sequence
t = torch.tensor([500])                       # halfway through
x_t = fp.sample_q_t(x_0, t)                  # corrupted sequence
```

### Visualizing Corruption Over Time

As `t` increases from 0 to `T`, the text degrades from fully clean to fully corrupted:

```python
from src.forward_process import visualize_corruption

# Show corruption at t=0, T/4, T/2, T
visualize_corruption(x_0[0], fp, timesteps=[1, 250, 500, 1000])
```

For absorbing corruption, you will see more and more `[MASK]` tokens. For uniform corruption, the text becomes increasingly random.

## Paper Reference

Austin, J., Johnson, D. D., Ho, J., Tarlow, D., & van den Berg, R. (2021). **Structured Denoising Diffusion Models in Discrete State-Spaces (D3PM)**. *NeurIPS 2021*. Section 3: Forward Process.

[https://arxiv.org/abs/2107.03006](https://arxiv.org/abs/2107.03006)

## Exercises

1. **Verify the transition matrix:** Create a `DiscreteForwardProcess` with `vocab_size=5` and `schedule="absorbing"`. Compute `Q_bar_t` at `t=T` and verify that every row sends all probability mass to the mask token.

2. **Compare schedules:** Corrupt the same sentence with both "uniform" and "absorbing" schedules at `t=T/2`. Which one is more readable? Why?

3. **Custom schedule:** Modify `_compute_beta_schedule` to use a cosine schedule instead of linear. How does this affect the corruption rate at different timesteps?

## What's Next

In [Lesson 02: The Reverse Denoising Process](../lesson02-reverse-denoising-process/), we will learn how to undo this corruption step by step, which is the foundation of discrete diffusion generation.
