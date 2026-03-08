# Lab 01 -- Flow Matching Text Generator

> **Compute Requirements**: Training on TinyStories with d_model=256 and 4 Transformer layers takes approximately 20-40 minutes on a single GPU. CPU-only training is possible but will be significantly slower (several hours). Consider using Google Colab with a GPU runtime if you do not have a local GPU.

## Problem Statement

Build a **flow-matching text generator** from scratch and compare it against SDE-based Diffusion-LM from Module 2.

You will:

1. **Implement** a `FlowMatchingTextGenerator` class that:
   - Tokenizes text from TinyStories using a simple word-level tokenizer.
   - Embeds tokens into continuous space.
   - Trains flow matching (linear interpolation, velocity prediction).
   - Samples via Euler ODE integration.
   - Rounds embeddings back to tokens.

2. **Implement** an SDE-based baseline for comparison:
   - Uses the same architecture but with DDPM-style noise addition and reverse sampling.

3. **Compare** the two approaches on:
   - **Sample quality**: Are the generated sentences coherent?
   - **Number of function evaluations (NFE)**: How many model forward passes are needed?
   - **Wall-clock generation time**: Which is faster?

## Files

- `notebook.ipynb` -- Starter code with `# TODO` markers. Fill in the marked sections.
- `solutions/solution.py` -- Reference implementation (do not peek until you have tried!).
- `tests/test_solution.py` -- Tests to verify your implementation.

## Tasks

### Task 1: Build the Tokenizer
Create a simple word-level tokenizer with `<pad>`, `<unk>`, `<bos>`, `<eos>` special tokens. Implement `encode` and `decode` methods.

### Task 2: Implement Flow Matching Training
Implement the `train_step` method:
- Embed token IDs to get x_1 (target embeddings).
- Sample Gaussian noise x_0.
- Sample time t uniformly from [0, 1].
- Compute interpolation x_t = (1 - t) * x_0 + t * x_1.
- Compute target velocity v = x_1 - x_0.
- Predict velocity with the model and compute MSE loss.

### Task 3: Implement ODE Sampling
Implement the `sample` method:
- Start from Gaussian noise.
- Integrate the velocity field using Euler method: x += dt * v(x, t).
- Round final embeddings to tokens.

### Task 4: Implement SDE Baseline
Implement a simplified DDPM-style baseline for comparison:
- Forward: add noise according to a variance schedule.
- Reverse: denoise step by step with noise injection.

### Task 5: Compare and Analyze
- Train both models on the same data.
- Generate samples with varying step counts.
- Measure NFE and wall-clock time.
- Report your findings.

## Evaluation Criteria

Your solution should:
1. Train without errors on the provided data.
2. Generate grammatically plausible (not necessarily perfect) text.
3. Show that ODE sampling needs fewer steps than SDE sampling for comparable quality.
4. Include timing measurements demonstrating the speed difference.

## Hints

- Start with a very small model (d_model=128, 2 layers) to debug, then scale up.
- The tokenizer can be as simple as splitting on whitespace.
- For the SDE baseline, you do not need a full DDPM implementation -- a simplified version with 100-1000 steps is sufficient.
- Use `torch.no_grad()` during sampling to save memory.
