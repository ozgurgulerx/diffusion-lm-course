# Lab 01: End-to-End Diffusion LM Project

## Problem Statement

Build a complete, end-to-end text generation system using diffusion language models. This capstone lab brings together everything from Module 5: data preparation, model training, controlled generation, and packaging.

**Your task**: Build a **poetry generation pipeline with style control**.

You will:
1. Prepare a poetry dataset with style labels (romantic, nature, melancholy)
2. Train a diffusion LM with classifier-free guidance support
3. Generate style-controlled poetry using CFG
4. Package everything as a reusable `PoetryPipeline` class

### Suggested Alternatives

If you prefer a different domain, you can adapt the same architecture for:
- **Code docstring generation**: Train on function/docstring pairs, condition on code complexity
- **Story completion**: Train on story paragraphs, condition on genre
- **Haiku generation**: Train on haiku with season/theme labels

The core pattern is the same: data with labels, conditional diffusion model, guided sampling, pipeline packaging.

## Compute Requirements

- **Minimum**: CPU-only, using sample data provided (~15 poems). Training takes ~2 minutes.
- **Recommended**: GPU (any CUDA-capable GPU). Allows training on larger poetry corpora.
- **For production quality**: GPU with 8GB+ VRAM. Train for 100+ epochs on thousands of poems.

## Setup

```bash
pip install torch transformers
```

## Instructions

Open `notebook.ipynb` and follow the guided steps. Each section has `# TODO` markers where you need to implement the solution.

### Step 1: Data Preparation
- Implement `PoetryDataset` that tokenizes poems and returns `input_ids`, `attention_mask`, and `label`
- Create or load poetry data with style labels
- Build a PyTorch DataLoader

### Step 2: Model Architecture
- Implement `PoetryDiffusionModel` with:
  - Token, position, and timestep embeddings
  - Style conditioning with random dropout (for classifier-free guidance)
  - Transformer encoder backbone
  - Output projection to vocabulary

### Step 3: Training
- Implement the MDLM training loop:
  - Sample random timesteps
  - Apply masking at the corresponding rate
  - Compute cross-entropy loss at masked positions
  - Track training metrics

### Step 4: Generation with CFG
- Implement classifier-free guided sampling:
  - Run model twice (conditional and unconditional)
  - Interpolate: `guided = (1 + w) * cond - w * uncond`
  - Progressive unmasking schedule

### Step 5: Pipeline Packaging
- Create a `PoetryPipeline` class that bundles model + tokenizer + generation
- Implement `__call__` for simple usage: `pipe(style="romantic", num_samples=4)`
- Add save/load functionality

## Evaluation Criteria

Your solution should:
1. **Work end-to-end**: From raw text to generated output without errors
2. **Support all three styles**: Romantic, nature, and melancholy generation
3. **Show CFG effect**: Conditional generation should differ from unconditional
4. **Be reusable**: The pipeline class should work as a standalone component

## Testing

Run the test suite to verify your solution:

```bash
cd tests
pytest test_solution.py -v
```

Tests check:
- Data preparation produces correct tensor shapes
- Model forward pass returns expected output dimensions
- Training produces decreasing (or stable) loss
- Generation returns non-empty strings for all styles
- Pipeline is callable and produces correct number of outputs

## Solution

The reference solution is in `solutions/solution.py`. Try to complete the lab on your own first before looking at it.
