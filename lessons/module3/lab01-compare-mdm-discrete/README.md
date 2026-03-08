# Lab 01: Compare MDM vs D3PM on WikiText-2

## Problem Statement

In this lab, you will:

1. **Implement MDM from scratch** -- build a Masked Diffusion Model with cosine masking schedule, training loss, and iterative unmasking sampler
2. **Train on WikiText-2** -- use a real text dataset to test your implementation
3. **Compare with D3PM** -- a D3PM baseline is provided; compare training efficiency and sample quality

## What You'll Build

- `TransformerDenoiser`: Transformer backbone that takes (masked_tokens, timestep) and outputs logits
- `cosine_masking_schedule`: gamma(t) function controlling masking rate
- `MDM` class with:
  - `forward_corrupt(x_0, t)`: mask tokens with probability gamma(t)
  - `train_loss(x_0)`: cross-entropy on masked positions
  - `sample(batch_size, seq_len, num_steps)`: iterative unmasking

## Setup

```bash
pip install torch datasets transformers
```

## Instructions

1. Open `notebook.ipynb` and follow the guided implementation
2. Fill in all `# TODO` sections
3. The notebook provides:
   - A D3PM baseline (pre-implemented)
   - Data loading utilities for WikiText-2
   - Evaluation metrics (distinct n-grams, self-perplexity)
   - Plotting code for loss curves

## Evaluation Criteria

Your implementation should:
- [ ] `forward_corrupt` correctly masks tokens proportional to gamma(t)
- [ ] `train_loss` computes cross-entropy only on masked positions
- [ ] `sample` generates sequences with no remaining [MASK] tokens
- [ ] MDM converges faster than D3PM (in wall-clock time)
- [ ] Generated samples show reasonable diversity (distinct-2 > 0.5)

## Model Configuration

Use these defaults for consumer GPU compatibility:
- `d_model=256`, `n_heads=4`, `n_layers=4`
- `max_seq_len=128`
- `batch_size=32`, `lr=3e-4`
- Training: 5-10 epochs on WikiText-2

## Files

- `notebook.ipynb` -- guided implementation with TODO markers
- `solutions/solution.py` -- reference solution
- `tests/test_solution.py` -- unit tests (run with `pytest tests/`)

## Running Tests

```bash
cd lessons/module3/lab01-compare-mdm-discrete
pytest tests/test_solution.py -v
```

To test your own implementation, update the import path in the test file.
