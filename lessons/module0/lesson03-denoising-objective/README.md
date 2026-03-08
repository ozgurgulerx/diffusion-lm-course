# Lesson 3: The Denoising Objective

The core training signal for diffusion models: given a noisy version of data, predict the clean version.

## Prerequisites

- [Lesson 1: Autoregressive vs. Diffusion](../lesson01-autoregressive-vs-diffusion/)
- [Lesson 2: Discrete vs. Continuous Noise](../lesson02-discrete-vs-continuous-noise/)

## Learning Objective

After this lesson you will be able to implement and train both discrete and continuous denoising objectives, and explain why "predict the clean data from noisy data" is a valid training signal for generative models.

## Concept

### The Key Idea

A diffusion model is trained to **reverse the noise process**. During training:

1. Take a clean data sample x_0.
2. Add noise at a random level t to get x_t.
3. Train the model to predict x_0 (or the noise) from x_t.

At generation time, we start from pure noise and repeatedly apply the denoiser to recover clean data. The training objective teaches the model what clean data looks like at every noise level.

### Discrete Denoising: Cross-Entropy Loss

For discrete diffusion, the denoiser sees corrupted token IDs and predicts the original token at each position. This is a classification problem — cross-entropy loss:

```python
# Corrupt tokens
corrupted = corrupt(clean_ids, noise_level, vocab_size)

# Predict clean tokens
logits = model(corrupted)  # (batch, seq_len, vocab_size)

# Cross-entropy loss against clean targets
loss = F.cross_entropy(logits.view(-1, vocab_size), clean_ids.view(-1))
```

The random baseline loss is `log(vocab_size)` — if the model cannot do better than random guessing, that is the loss you would see.

### Continuous Denoising: MSE Loss

For continuous diffusion, the denoiser sees noisy embedding vectors and predicts either:
- **x_0-prediction**: the clean embeddings (predict the signal).
- **epsilon-prediction**: the noise that was added (predict the noise).

Both use MSE loss:

```python
# Add noise to embeddings
noisy = sqrt(alpha) * clean_emb + sqrt(1 - alpha) * noise

# Predict clean embeddings (x0-prediction)
predicted = model(noisy, noise_level)

# MSE loss
loss = F.mse_loss(predicted, clean_emb)
```

These two targets are mathematically equivalent (you can derive one from the other given the noise level), but they can have different training dynamics.

### Training Loop

The full training procedure is:

```
for each batch of clean data:
    1. Sample random noise level t ~ Uniform(0, 1)
    2. Add noise: x_t = noise(x_0, t)
    3. Predict: x_0_hat = model(x_t, t)
    4. Loss = compare(x_0_hat, x_0)
    5. Backpropagate
```

In the notebook, we train a tiny denoiser on a toy dataset and watch the loss decrease — confirming that the model is learning the data distribution.

## Paper Links

- [Denoising Diffusion Probabilistic Models (Ho et al., 2020)](https://arxiv.org/abs/2006.11239) — the DDPM paper that established the modern denoising framework.
- [Structured Denoising Diffusion Models in Discrete State-Spaces (Austin et al., 2021)](https://arxiv.org/abs/2107.03006) — extends the framework to discrete data.

## Exercises

1. **Random baseline**: Before training, compute the loss. For the discrete denoiser, verify it is close to `log(vocab_size)`. What is the equivalent baseline for the continuous denoiser?

2. **Noise level matters**: Train the discrete denoiser but only use a fixed noise level (e.g., always 0.5). Then train with random noise levels sampled uniformly. Which model generalizes better to different noise levels at test time?

3. **x_0 vs. epsilon prediction**: Train the continuous denoiser with both targets. Plot the loss curves. Do they converge to different values? Which one is lower?

## What's Next

You have completed Module 0 (Foundations). In Module 1, you will build a complete discrete diffusion language model from scratch using these building blocks.
