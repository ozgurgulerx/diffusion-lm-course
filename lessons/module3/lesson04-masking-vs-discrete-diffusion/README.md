# Lesson 04: Masking vs General Discrete Diffusion

## Prerequisites

- Lesson 01-03: MLM as diffusion, MDM, SEDD
- Module 1 Lesson 03: D3PM (transition matrices, uniform/absorbing corruption)
- Understanding of both masked and general discrete diffusion

## Learning Objective

Directly compare MDM (masked diffusion) and D3PM (general discrete diffusion) on the same dataset with the same compute budget. Understand when each approach is better and the tradeoffs involved.

## Concept

### Two Approaches to Discrete Diffusion

**MDM (Masked Diffusion)**:
- Forward process: each token independently becomes [MASK]
- Corruption is binary: original or masked
- Loss: cross-entropy on masked positions only
- Sampling: iterative unmasking (reveal tokens one group at a time)

**D3PM (General Discrete Diffusion)**:
- Forward process: each token can become ANY other token (via transition matrix Q)
- Corruption is gradual: token -> similar token -> random token
- Loss: cross-entropy on ALL positions (every token may be corrupted)
- Sampling: iterative denoising (refine all tokens simultaneously)

### Key Differences

| Aspect | MDM | D3PM (uniform) |
|--------|-----|-----------------|
| Forward process | Binary (keep/mask) | Continuous (any transition) |
| Transition matrix | Trivial (2-state) | Full V x V matrix |
| Loss positions | Masked only | All positions |
| Training efficiency | Faster (sparse loss) | Slower (dense loss) |
| Generality | Mask corruption only | Any corruption type |
| Start of sampling | All [MASK] | Random tokens |

### When MDM Wins

1. **Training speed**: Computing loss only on masked positions is cheaper than all positions
2. **Simplicity**: No need to handle transition matrices or their products
3. **Text generation**: The mask-then-reveal paradigm naturally suits language
4. **Fewer parameters**: No need to learn complex transition dynamics

### When D3PM Wins

1. **Non-text domains**: For data without a natural "mask" token (e.g., protein sequences where any amino acid is meaningful)
2. **Smooth corruption**: Gradual token-to-token transitions can preserve more structure
3. **Flexibility**: Can design task-specific transition matrices

### Empirical Comparison

The code in `src/comparison.py` trains both models with:
- Same Transformer architecture (shared backbone)
- Same number of parameters
- Same training budget (epochs, batch size, learning rate)

Metrics compared:
- **Training loss convergence**: How fast each model learns
- **Distinct-2**: Diversity of generated samples (unique bigram ratio)
- **Self-perplexity**: Model's own confidence in its generations

Typical findings:
- MDM converges ~1.5-2x faster in wall-clock time
- MDM achieves lower training loss in fewer epochs
- D3PM generates more diverse samples (due to random-token starting point)
- Both reach similar quality with enough training

## Code

See `src/comparison.py`:
- `MDMComparison`: streamlined MDM for fair comparison
- `D3PMComparison`: simplified D3PM with uniform transitions
- `train_and_compare`: trains both and reports metrics
- `distinct_ngrams`, `sample_perplexity`: evaluation metrics

## Exercises

1. **Fair comparison**: Run `train_and_compare` with d_model=256, n_layers=4. Which model reaches lower loss first? Does the winner change with more training?

2. **Compute-matched**: Instead of matching epochs, match total FLOPs. Since MDM only computes loss on masked positions (~50% of tokens on average), give D3PM 50% fewer steps. Is the comparison still favorable for MDM?

3. **Absorbing D3PM**: Modify D3PMComparison to use absorbing (mask) transitions instead of uniform. How does it compare to MDM? (Hint: it should be very similar -- the difference is in the loss weighting.)

4. **Scale experiment**: Try d_model in {64, 128, 256, 512}. Does the relative advantage of MDM vs D3PM change with model size?

## What's Next

Lab 01 puts this into practice: implement MDM from scratch, train on WikiText-2, and run a full comparison against D3PM.
