# Lab 01: Train a Discrete Diffusion Model on TinyStories

## Problem Statement

Train a discrete diffusion model (your choice of D3PM or MDLM) on the TinyStories dataset and generate 10 coherent short stories.

## Success Criteria

1. Your model trains without errors on TinyStories (character-level tokenization).
2. You generate 10 text samples of at least 64 characters each.
3. Your model achieves a perplexity proxy score below 3.5 on a held-out evaluation batch.
4. At least 5 of the 10 generated samples contain recognizable English words and basic sentence structure.

## Setup

- Use the `shared/` utilities for data loading and tokenization.
- Use the model implementations from Lessons 03 (D3PM) and 04 (MDLM).
- Use the training utilities from Lesson 05.

## Recommended Configuration

```
Dataset: TinyStories, max_samples=5000
Tokenization: character-level
Sequence length: 64
Model: d_model=128, n_heads=4, n_layers=4
Timesteps: 100 (D3PM) or 100 sampling steps (MDLM)
Learning rate: 3e-4 with cosine schedule + warmup
Batch size: 64
Epochs: 30-50
Temperature: 0.7-0.9 for generation
```

This configuration should train in under 10 minutes on a consumer GPU (T4, RTX 3060, or M1/M2 Mac).

## Deliverables

Complete the notebook by filling in all `# TODO` sections:

1. Data preparation
2. Model construction
3. Training loop with LR scheduling
4. Sample generation
5. Quality evaluation

## Hints

- Start with MDLM — it is simpler and often works better out of the box.
- Use gradient clipping (max norm 1.0) for stable training.
- Monitor the training loss; it should decrease steadily.
- If samples are poor, try lower temperature (0.5-0.7) and more training epochs.
- The cosine masking schedule generally outperforms linear.
