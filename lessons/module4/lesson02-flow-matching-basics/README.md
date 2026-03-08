# Lesson 02 -- Flow Matching Basics

## Prerequisites

- Lesson 01 (Score Matching for Text): understanding of score functions and denoising objectives
- Module 2 (Continuous Diffusion): familiarity with SDEs for text generation
- Basic knowledge of ODEs and vector fields

## Learning Objectives

By the end of this lesson you will be able to:

1. Explain how flow matching differs from score-based diffusion (ODE vs SDE, velocity vs score).
2. Implement the conditional flow matching training objective with linear interpolation.
3. Sample from a trained flow model using the Euler method.
4. Articulate why flow matching often requires fewer sampling steps than diffusion.

## Concept

### From SDE to ODE

Score-based diffusion models define a **stochastic** differential equation (SDE) with noise injection during both forward and reverse processes. Flow matching takes a simpler approach: define a **deterministic** ordinary differential equation (ODE) that transports noise to data.

### The Flow Matching Framework

Given data $x_1$ and noise $x_0 \sim \mathcal{N}(0, I)$, flow matching defines:

1. **Linear interpolation path**: $x_t = (1-t) \cdot x_0 + t \cdot x_1$ for $t \in [0, 1]$
2. **Target velocity**: $v = x_1 - x_0$ (the direction from noise to data)
3. **Training loss**: $\mathcal{L} = \mathbb{E}_{t, x_0, x_1} \| v_\theta(x_t, t) - (x_1 - x_0) \|^2$

The model $v_\theta$ learns to predict the velocity field that moves samples from noise toward data.

### Sampling via ODE

Generation integrates the learned velocity field from $t=0$ to $t=1$:

$$x_{t+\Delta t} = x_t + \Delta t \cdot v_\theta(x_t, t)$$

This is the **Euler method**. Because the paths are straight, even a coarse discretization works well.

### Why is this simpler?

- **No noise schedule**: No $\beta_t$, $\bar{\alpha}_t$, or variance schedules to tune.
- **Deterministic**: No stochastic noise during sampling.
- **Straight paths**: The interpolation is linear, leading to straighter ODE trajectories and fewer steps.
- **Simpler loss**: Direct MSE on velocity, no weighting schemes needed.

## Code

See `src/flow_matching.py` which implements:

- `VelocityNet` -- MLP velocity predictor for 2D demos
- `SequenceVelocityNet` -- Transformer velocity predictor for sequences
- `FlowMatcher` -- training with linear interpolation and Euler ODE sampling

## Paper Reference

- Lipman et al., "Flow Matching for Generative Modeling" (ICLR 2023)
- Liu et al., "Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow" (ICLR 2023)

## Exercises

1. **2D visualization**: Run the 2D demo and plot the generated samples. Do they cluster around the 4 target modes?
2. **Vary n_steps**: Sample with n_steps=10, 50, 100, 200. How does sample quality change? Plot the results side by side.
3. **Visualize trajectories**: Use `sample_trajectory` to plot how samples evolve from noise to data. Are the paths approximately straight?
4. **Compare with diffusion**: Implement DDPM-style sampling (add noise at each step) and compare the number of function evaluations needed to match flow matching quality.
5. **sigma_min**: Set `sigma_min=0.01` and retrain. Does this improve or hurt sample quality? Why might a small amount of noise help?

## What's Next

In the next lesson, we adapt flow matching to work with **token sequences** -- embedding tokens into continuous space, training flow matching, and rounding back to discrete tokens.
