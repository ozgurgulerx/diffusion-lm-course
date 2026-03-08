# Lesson 01 -- Score Matching for Text

## Prerequisites

- Module 2 (Continuous Diffusion for Text): token embeddings, forward/reverse SDE, Diffusion-LM
- Module 3 (Masked Diffusion): familiarity with SEDD and discrete corruption
- Basic understanding of probability density and gradients

## Learning Objectives

By the end of this lesson you will be able to:

1. Explain what the score function is and why it avoids the intractable normalizing constant.
2. Derive the equivalence between noise prediction (epsilon-parameterization) and score estimation.
3. Implement a denoising score matching training loop for text embeddings.
4. Describe the concrete score from SEDD as the discrete analog of the continuous score.

## Concept

### What is the score?

For a probability density p(x), the **score function** is the gradient of the log-density:

```
s(x) = nabla_x log p(x)
```

The key advantage: the normalizing constant Z in `p(x) = exp(f(x)) / Z` vanishes when we take the gradient of the log, so we never need to compute Z.

### Denoising score matching

Direct score matching requires access to the true score, which we do not have. **Denoising score matching** (Vincent, 2011) sidesteps this by:

1. Corrupting clean data x_0 with known Gaussian noise: `x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon`
2. Training a network to predict the noise epsilon from x_t.
3. The optimal noise predictor implicitly learns the score because:

```
nabla_{x_t} log p(x_t | x_0) = -epsilon / sqrt(1 - alpha_bar_t)
```

So **noise prediction IS score estimation**, up to a known scaling factor.

### Score matching for text

In Diffusion-LM (Li et al., 2022), text tokens are embedded into continuous space and then a standard continuous diffusion process is applied. The model is trained with MSE on x_0 (or equivalently, on epsilon). Our `ContinuousScoreNet` makes this connection explicit by showing that the same epsilon-prediction loss is denoising score matching.

### Discrete score (SEDD)

For discrete tokens, there is no gradient to take. SEDD (Lou et al., 2024) defines the **concrete score** as a ratio of transition probabilities:

```
s(x, y, t) = p(x_t = y | x_0) / p(x_t = x_i | x_0)
```

This measures how likely each token replacement is under the forward corruption. The model learns these ratios, enabling score-based generation directly in discrete token space.

## Code

See `src/score_matching.py` which implements:

- `ContinuousScoreNet` -- Transformer-based noise/score predictor for continuous embeddings
- `DenoisingScoreMatchingTrainer` -- training loop with explicit score-noise equivalence
- `concrete_score_example` -- illustration of discrete concrete scores (SEDD concept)

## Paper References

- **Denoising Score Matching**: Vincent, "A Connection Between Score Matching and Denoising Autoencoders" (2011)
- **Score-based Generative Modeling**: Song & Ermon, "Generative Modeling by Estimating Gradients of the Data Distribution" (NeurIPS 2019)
- **Diffusion-LM**: Li et al., "Diffusion-LM Improves Controllable Text Generation" (NeurIPS 2022)
- **SEDD**: Lou et al., "Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution" (ICML 2024)

## Exercises

1. **Verify the equivalence**: Run the demo in `score_matching.py` and confirm that the two score computation paths give identical results.
2. **Vary the noise level**: Modify the demo to compute scores at t=10, t=50, t=90. How does the score magnitude change with noise level? Why?
3. **Weight the loss**: In practice, score matching losses are weighted by sigma_t^2 to balance contributions across timesteps. Add this weighting to `train_step` and observe the effect on training dynamics.
4. **Concrete score**: Extend `concrete_score_example` to use absorbing (mask) corruption instead of uniform corruption. How do the score ratios differ?

## What's Next

In the next lesson, we introduce **flow matching** -- a simpler alternative to score-based diffusion that replaces the SDE with a deterministic ODE along straight paths.
