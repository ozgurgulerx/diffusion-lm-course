# Lab 01: Classifier-Guided Controlled Generation

**Use Diffusion-LM and classifier gradients to steer text generation toward a target attribute (positive sentiment).**

## Prerequisites

- Module 2, Lessons 01-05 (all completed)

## Problem Statement

In this lab, you will implement the complete classifier-guided generation pipeline from the Diffusion-LM paper:

1. **Train a Diffusion-LM** on a small text dataset (TinyStories or similar).
2. **Train a sentiment classifier** that operates on noisy embeddings at any noise level.
3. **Implement classifier-guided sampling** that modifies the reverse SDE to steer toward positive-sentiment text.
4. **Evaluate** by comparing the sentiment of guided vs. unguided samples.

### Background

The key insight from Dhariwal & Nichol (2021) and Li et al. (2022): during reverse diffusion sampling, we can add the gradient of a classifier to the score estimate:

```
score_guided(x, t) = score(x, t) + s * grad_x log p(y=target | x, t)
```

where `s` is the guidance scale. This steers the generation toward samples that the classifier rates as having the target attribute.

For this to work, the classifier must be trained on **noisy** embeddings at all noise levels, not just clean data. Otherwise it cannot provide useful gradients during the reverse process.

## Success Criteria

1. Your Diffusion-LM trains and produces text (even if not perfectly coherent at this small scale).
2. Your classifier achieves > 60% accuracy on held-out noisy-embedding sentiment classification.
3. Guided samples (guidance_scale > 0) contain more positive-sentiment words than unguided samples (guidance_scale = 0).
4. All tests in `tests/test_solution.py` pass.

## Getting Started

Open `notebook.ipynb` and follow the TODO markers. The notebook is structured in four parts:

1. **Part 1**: Train the Diffusion-LM
2. **Part 2**: Create labeled data and train the classifier
3. **Part 3**: Implement classifier-guided sampling
4. **Part 4**: Evaluate guided vs. unguided generation

## Files

```
lab01-controlled-generation/
  README.md              # This file
  notebook.ipynb         # Starter notebook with TODOs
  solutions/
    solution.py          # Reference implementation
  tests/
    test_solution.py     # Automated tests
```

## Hints

- Keep the model small: embed_dim=64, n_layers=4, seq_len=32. This trains in minutes on a single GPU.
- Use word-level tokenization with a small vocabulary (500-1000 tokens).
- For the classifier, a simple MLP that mean-pools over sequence positions works well.
- Start with guidance_scale=1.0 and increase if needed. Too high a scale can produce degenerate outputs.
- The classifier must be trained with noise added at random timesteps matching the diffusion process.

## Paper References

- Li et al., "Diffusion-LM Improves Controllable Text Generation" (2022), Section 4 -- classifier-guided generation.
- Dhariwal & Nichol, "Diffusion Models Beat GANs on Image Synthesis" (2021), Section 4 -- classifier guidance (originally for images).
