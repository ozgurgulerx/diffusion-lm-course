# Lesson 05: Training and Sampling Strategies

Practical details for training discrete diffusion models well: learning rate schedules, loss weighting, sampling strategies, and model comparison.

## Prerequisites

- Lesson 03: D3PM from Scratch
- Lesson 04: MDLM

## Learning Objective

After this lesson you will be able to apply practical training improvements (LR warmup, loss weighting, gradient clipping) and advanced sampling strategies (temperature, top-k, top-p) to discrete diffusion models, and compare D3PM vs MDLM on the same data.

## Concept: Making Training and Sampling Work Well

Having a correct implementation is necessary but not sufficient. This lesson covers the practical details that make discrete diffusion models actually produce good text.

### Loss Weighting Across Timesteps

Not all timesteps are equally important. Very early timesteps (t near 0) have almost no corruption, so the loss is trivially low. Very late timesteps (t near T) are nearly fully corrupted, making prediction nearly impossible.

The most informative timesteps are in the middle range. We can weight the loss accordingly:

```python
from src.training_utils import importance_weight_timesteps

# SNR-based weighting: emphasize intermediate timesteps
weights = importance_weight_timesteps(t, num_timesteps=100, strategy="snr")
weighted_loss = (per_sample_loss * weights).mean()
```

### Learning Rate Schedules

Cosine annealing with linear warmup is the standard choice:

```python
from src.training_utils import get_cosine_schedule_with_warmup

scheduler = get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps=100, num_training_steps=10000
)
```

### Sampling Strategies

Beyond temperature, we can apply top-k and top-p (nucleus) filtering to the model's predictions:

```python
from src.training_utils import sample_with_temperature

# Top-k: only consider the 50 most likely tokens
tokens = sample_with_temperature(logits, temperature=0.8, top_k=50)

# Top-p: only consider tokens in the top 90% of probability mass
tokens = sample_with_temperature(logits, temperature=0.8, top_p=0.9)
```

### Comparing D3PM vs MDLM

```python
from src.compare_models import print_comparison

print_comparison(d3pm_samples, mdlm_samples, reference_texts)
```

### Practical Recommendations

1. **Start with MDLM** — simpler to implement and tune
2. **Use cosine masking schedule** — better than linear
3. **Cosine LR with warmup** — prevents early training instability
4. **Gradient clipping** at 1.0 — standard for transformers
5. **Temperature 0.7-0.9** for sampling — good quality-diversity tradeoff
6. **100-200 sampling steps** — diminishing returns beyond this

## Exercises

1. **Loss weighting experiment:** Train MDLM with "uniform", "snr", and "truncated" loss weighting. Compare the training curves and final sample quality.

2. **Sampling strategy sweep:** Fix a trained model and generate samples with different (temperature, top_k, top_p) combinations. Rank the outputs.

3. **Head-to-head comparison:** Train D3PM (absorbing) and MDLM on the same data with the same model size. Use `compare_models.py` to produce a quantitative comparison.

## What's Next

You now have all the tools needed for the [Lab: Train a Discrete Diffusion Model](../lab01-train-discrete-diffusion/) on TinyStories. In Module 2 (coming later), we will explore continuous diffusion in embedding space.
