# Lesson 03: Infilling and Constrained Generation

## Prerequisites

- Completed Lessons 01-02 (training and controlled generation)
- Understanding of the MDLM denoising process
- Familiarity with masking schedules
- **Compute**: CPU is sufficient for inference. GPU recommended for training.

## Learning Objective

By the end of this lesson, you will be able to use diffusion LMs for text infilling (generating middle text given beginning and end) and constrained generation (enforcing hard requirements on the output). These are capabilities that set diffusion LMs apart from autoregressive models.

## Concept

### Why Diffusion LMs Excel Here

Autoregressive (AR) models generate text left-to-right. They cannot naturally:
- Fill in a gap between known text (infilling)
- Guarantee specific tokens appear at specific positions
- Generate text matching a template with fixed and free slots

Diffusion LMs denoise all positions simultaneously, making these tasks natural.

### Infilling with Repaint

The Repaint algorithm (Lugmayr et al., 2022) adapts diffusion models for inpainting. The key idea:

1. **Fix known positions**: At every denoising step, replace known positions with their correct values
2. **Denoise unknown positions**: Let the model predict tokens at unknown positions
3. **Re-noise for coherence**: Periodically re-noise everything and repeat, so the model can harmonize the boundary between known and generated text

```python
from infilling import InfillingSampler, RepaintScheduler

scheduler = RepaintScheduler(
    num_timesteps=1000,
    resample_steps=10,   # Number of re-noising jumps
    jump_length=10,      # How far to jump back
)

sampler = InfillingSampler(model, mask_token_id=103, scheduler=scheduler)

# Infill between prefix and suffix
prefix = tokenizer.encode("The cat sat on")
suffix = tokenizer.encode("and looked out the window")
result = sampler.infill(
    prefix=torch.tensor(prefix),
    suffix=torch.tensor(suffix),
    infill_length=5,
    num_steps=50,
)
# Result: "The cat sat on [the warm cushion] and looked out the window"
```

You can also infill arbitrary positions using a mask:

```python
tokens = tokenizer.encode("The ___ cat ___ on the ___")
mask = torch.tensor([False, True, False, True, False, False, True])
result = sampler.infill_with_mask(tokens, mask, num_steps=50)
```

### Constrained Generation

Three types of hard constraints are supported:

#### 1. Token Constraints
Fix specific tokens at specific positions:

```python
from constrained import TokenConstraint, ConstrainedSampler

constraint = TokenConstraint(
    positions=[0, 5, 10],
    token_ids=[tokenizer.encode("The")[0],
               tokenizer.encode("beautiful")[0],
               tokenizer.encode("sunset")[0]],
)

sampler = ConstrainedSampler(model, mask_token_id=103)
result = sampler.sample_with_token_constraint(constraint, seq_len=20)
# "The" at pos 0, "beautiful" at pos 5, "sunset" at pos 10, rest generated
```

#### 2. Template Constraints
Define a template with fixed and free slots:

```python
from constrained import TemplateConstraint

# None = generate, int = fixed token
template = TemplateConstraint([
    the_id, None, None, None, is_id, None, None, period_id
])
result = sampler.sample_with_template(template)
# "The [generated] [generated] [generated] is [generated] [generated] ."
```

#### 3. Keyword Constraints
Require that specific keywords appear somewhere in the output:

```python
from constrained import KeywordConstraint

kw = KeywordConstraint(keyword_token_ids=[
    tokenizer.encode("ocean"),
    tokenizer.encode("waves"),
])
result = sampler.sample_with_keywords(kw, seq_len=64)
# Generated text guaranteed to contain "ocean" and "waves"
```

### How Constraints Work

At each denoising step, after the model predicts and tokens are unmasked:
1. **Token/Template constraints**: Overwrite constrained positions with required values
2. **Keyword constraints**: Periodically project the current sequence to include required keywords, preferring currently-masked positions for insertion

The model learns to generate text that is coherent *around* the constraints because it sees the constraint tokens as context during subsequent denoising steps.

## Paper Reference

- Lugmayr et al. (2022), "RePaint: Inpainting using Denoising Diffusion Probabilistic Models" - [arXiv:2201.09865](https://arxiv.org/abs/2201.09865)
- Li et al. (2022), "Diffusion-LM Improves Controllable Text Generation" - [arXiv:2205.14217](https://arxiv.org/abs/2205.14217)

## Exercises

1. **Basic Infilling**: Given the sentence "I went to the ___ and bought some ___", use the infilling sampler to generate 10 different completions. How diverse are the results?

2. **Repaint Parameters**: Experiment with `resample_steps` (1 vs 5 vs 20) and `jump_length` (5 vs 10 vs 20). How do these affect coherence at the boundary between known and generated text?

3. **Template Generation**: Create a template for a simple sentence pattern (e.g., "The [ADJ] [NOUN] [VERB] the [ADJ] [NOUN].") and generate 10 completions. Do they follow the intended grammatical structure?

4. **Keyword Inclusion**: Generate text that must contain three specific keywords. Compare the output quality when the keywords are semantically related vs unrelated.

## What's Next

In [Lesson 04](../lesson04-huggingface-bridge/), you will learn how to bridge from these scratch implementations to the HuggingFace ecosystem, making it easy to share and reuse your trained models.
