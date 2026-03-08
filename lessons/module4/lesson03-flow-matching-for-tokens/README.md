# Lesson 03 -- Flow Matching for Tokens

## Prerequisites

- Lesson 02 (Flow Matching Basics): linear interpolation, velocity prediction, Euler sampling
- Module 2 (Continuous Diffusion): token embedding, rounding/decoding
- Familiarity with tokenizers and embedding tables

## Learning Objectives

By the end of this lesson you will be able to:

1. Adapt flow matching to operate on token sequences via embedding space.
2. Implement the full pipeline: embed tokens, train flow matching, sample via ODE, round to tokens.
3. Compare ODE-based (flow matching) vs SDE-based (Diffusion-LM) text generation.
4. Explain the trade-offs between embedding-space and simplex-based discrete flow matching.

## Concept

### Two Approaches to Flow Matching for Text

**Approach 1: Flow matching in embedding space** (implemented here)
- Embed discrete tokens into continuous vectors using a learned embedding table.
- Train flow matching on these embeddings, exactly as in Lesson 02.
- Sample via ODE integration (Euler method).
- Round the generated embeddings back to discrete tokens via nearest-neighbor lookup.
- This is essentially Diffusion-LM with an ODE instead of an SDE.

**Approach 2: Discrete flow matching on the probability simplex**
- Define flows directly on probability distributions over the vocabulary.
- Each position carries a probability vector over V tokens.
- The flow transports from uniform distribution to one-hot data distribution.
- More principled for discrete data, but more complex to implement.

We implement Approach 1 because it directly builds on Module 2 knowledge and clearly shows the SDE-to-ODE transition.

### ODE vs SDE for Text

| Aspect | SDE (Diffusion-LM) | ODE (Flow Matching) |
|--------|--------------------|--------------------|
| Forward process | Gradual noise addition | Linear interpolation |
| Reverse process | Stochastic (noise at each step) | Deterministic (no noise) |
| Sampling steps | ~1000 typical | ~50-100 typical |
| Generation | Non-deterministic | Deterministic (same noise -> same output) |
| Rounding | Same nearest-neighbor | Same nearest-neighbor |

### Rounding to Tokens

Both approaches need to map continuous embeddings back to discrete tokens. We use cosine similarity against the embedding table to find the nearest token. This is the same rounding step used in Diffusion-LM (Module 2, Lesson 05).

## Compute Requirements

Training on real text data (e.g., TinyStories) with d_model=256 and 4 Transformer layers takes approximately 10-30 minutes on a single GPU. The notebook uses small synthetic data for quick demonstration; the lab uses real text data.

## Code

See `src/text_flow_matching.py` which implements:

- `TextVelocityNet` -- Transformer velocity predictor for token embeddings
- `TextFlowMatcher` -- complete pipeline (embed, train, sample, round)
- Trajectory visualization to see how tokens evolve during generation

## Paper References

- Lipman et al., "Flow Matching for Generative Modeling" (ICLR 2023)
- Li et al., "Diffusion-LM Improves Controllable Text Generation" (NeurIPS 2022) -- the SDE-based baseline
- Campbell et al., "A Continuous Time Framework for Discrete Denoising Models" (NeurIPS 2022) -- discrete flow perspective

## Exercises

1. **Compare ODE vs SDE sampling**: Generate text with both 50-step ODE and 1000-step SDE (from Module 2). Are the outputs qualitatively different?
2. **Vary ODE steps**: Generate with n_steps in [10, 25, 50, 100, 200]. At what point does quality plateau?
3. **Rounding strategies**: Try L2 distance instead of cosine similarity for rounding. Does it change the output?
4. **Embedding regularization**: Add an L2 penalty on the embedding table to keep embeddings close to a unit sphere. Does this help rounding accuracy?
5. **Token trajectory**: Use `generate_with_trajectory` to visualize how tokens change during generation. Do most tokens stabilize early, or do they keep changing until the end?

## What's Next

In the lab, you will build a complete flow-matching text generator trained on TinyStories and compare it against the SDE-based approach from Module 2.
