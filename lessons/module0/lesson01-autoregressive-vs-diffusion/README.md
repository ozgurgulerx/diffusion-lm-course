# Lesson 1: Autoregressive vs. Diffusion Language Models

Two fundamentally different ways to generate text: one token at a time, or all tokens in parallel through iterative refinement.

## Prerequisites

None. This is the first lesson in the course.

## Learning Objective

After this lesson you will be able to explain the key difference between autoregressive and diffusion approaches to text generation, and implement a simple generation loop for each.

## Concept

### Autoregressive Generation

Standard language models (GPT, LLaMA, etc.) generate text **left to right**, one token at a time. At each step, the model sees all previous tokens and predicts the next one:

```python
sequence = [start_token]
for _ in range(max_length):
    logits = model(sequence)       # condition on everything so far
    next_token = sample(logits[-1]) # predict ONE new token
    sequence.append(next_token)     # grow the sequence by 1
```

Key properties:
- **Sequential**: token N cannot be generated until tokens 1..N-1 exist.
- **Causal**: the model only looks backward (left context).
- **Generation time** scales linearly with sequence length.

### Diffusion Generation

Diffusion language models take a completely different approach. Instead of building a sequence token by token, they:

1. **Start from noise** (random tokens or random vectors).
2. **Denoise all positions simultaneously** using a trained model.
3. **Repeat** for a fixed number of steps, gradually refining the output.

```python
x = random_tokens(seq_len)          # start from pure noise
for step in range(num_denoise_steps):
    logits = denoiser(x)             # predict clean tokens for ALL positions
    x = sample(logits)               # update ALL positions at once
```

Key properties:
- **Parallel**: all positions are generated/refined simultaneously.
- **Non-causal**: the model sees (noisy versions of) all positions.
- **Generation time** depends on number of denoising steps, not sequence length.

### Why Diffusion for Language?

| Property | Autoregressive | Diffusion |
|---|---|---|
| Generation order | Fixed (left to right) | Flexible (all at once) |
| Parallel generation | No | Yes |
| Can revise earlier tokens | No | Yes (iterative refinement) |
| Controllability | Hard (must plan ahead) | Easier (can guide at any step) |

The ability to revise and refine is particularly interesting for tasks like constrained generation, infilling, and editing.

## Paper Link

- [Diffusion-LM Improves Controllable Text Generation (Li et al., 2022)](https://arxiv.org/abs/2205.14217) — the foundational paper on diffusion for language.

## Exercises

1. **Vary the temperature**: In `toy_autoregressive.py`, change the `temperature` parameter to 0.1, 1.0, and 5.0. How does the output change? Why?

2. **Count the steps**: In the diffusion loop (`toy_diffusion_concept.py`), the model processes the full sequence at each step. If the sequence length is L and we use S denoising steps, how many total model forward passes are needed? Compare with the autoregressive case.

3. **Observe the trajectory**: Run the diffusion generation with different random seeds. Does the sequence converge to the same output? Why or why not?

## What's Next

[Lesson 2: Discrete vs. Continuous Noise](../lesson02-discrete-vs-continuous-noise/) — how to add noise to text for the forward diffusion process.
