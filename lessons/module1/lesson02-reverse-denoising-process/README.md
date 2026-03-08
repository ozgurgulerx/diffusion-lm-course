# Lesson 02: The Reverse Denoising Process

The reverse process learns to undo corruption step by step, enabling generation of clean text from noise.

## Prerequisites

- Lesson 01: Forward Corruption Process (transition matrices, Q_bar_t, sample_q_t).

## Learning Objective

After this lesson you will be able to derive and implement the reverse posterior q(x_{t-1} | x_t, x_0) for discrete diffusion, and understand how predicting x_0 from x_t enables denoising.

## Concept: Reversing the Corruption

In Lesson 01, we corrupted clean text into noise. Now we go the other direction: given corrupted tokens x_t, we want to recover x_{t-1} (one step less corrupted).

### The Posterior

Using Bayes' rule, the reverse posterior is:

```
q(x_{t-1} = j | x_t, x_0) ∝ q(x_t | x_{t-1} = j) * q(x_{t-1} = j | x_0)
```

Where:
- `q(x_t | x_{t-1} = j) = Q_t[j, x_t]` — the single-step forward transition
- `q(x_{t-1} = j | x_0) = Q_bar_{t-1}[x_0, j]` — the cumulative transition from x_0 to x_{t-1}

This requires knowing x_0, which we don't have during generation. The key insight of D3PM is: **train a neural network to predict x_0 from x_t**, then use that prediction to compute the posterior.

### Computing the Posterior in Code

```python
from src.reverse_process import compute_posterior, sample_reverse_step

# Given corrupted x_t and a model's prediction of x_0
posterior = compute_posterior(x_t, x_0_probs, t, forward_process)
# posterior shape: (batch, seq_len, vocab_size)

# Sample x_{t-1} from this posterior
x_t_minus_1 = sample_reverse_step(x_t, x_0_probs, t, forward_process)
```

### Oracle Sanity Check

To verify our posterior computation, we can use an "oracle" that knows the true x_0. With perfect knowledge, the reverse process should recover the original sequence:

```python
from src.reverse_process import demo_reverse_with_oracle

recovered = demo_reverse_with_oracle(x_0, forward_process, t_start=50)
# recovered should be very close to x_0
```

### From Oracle to Neural Network

In practice, we replace the oracle with a learned model:
1. The model takes (x_t, t) as input
2. It outputs a probability distribution over x_0 for each position
3. We use this prediction in the posterior formula to take one reverse step
4. Repeat from t=T down to t=1 to generate clean text

This is exactly what we build in Lesson 03.

## Paper Reference

Austin, J., Johnson, D. D., Ho, J., Tarlow, D., & van den Berg, R. (2021). **Structured Denoising Diffusion Models in Discrete State-Spaces (D3PM)**. *NeurIPS 2021*. Section 3.2: Reverse Process.

[https://arxiv.org/abs/2107.03006](https://arxiv.org/abs/2107.03006)

## Exercises

1. **Oracle recovery:** Run `demo_reverse_with_oracle` with both uniform and absorbing schedules. Does the oracle always perfectly recover x_0? Why or why not?

2. **Noisy predictions:** Instead of using a perfect oracle, add noise to the x_0 prediction (e.g., mix the true one-hot with uniform noise). How does this affect the quality of the reverse process?

3. **Posterior analysis:** For the absorbing schedule, examine the posterior at t=2. When x_t is a mask token, which tokens get high posterior probability?

## What's Next

In [Lesson 03: D3PM from Scratch](../lesson03-d3pm-from-scratch/), we build a complete discrete diffusion model with a transformer-based denoiser, train it on text, and generate samples.
