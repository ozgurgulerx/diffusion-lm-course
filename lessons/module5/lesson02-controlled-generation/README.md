# Lesson 02: Controlled Generation

## Prerequisites

- Completed Lesson 01 (training on custom data)
- Understanding of MDLM denoising process
- Familiarity with gradient-based optimization
- **Compute**: GPU recommended. Classifier guidance requires backpropagation through the model at each denoising step.

## Learning Objective

By the end of this lesson, you will be able to steer diffusion LM generation toward desired attributes using three complementary approaches: classifier guidance, classifier-free guidance, and plug-and-play constraints.

## Concept

Controlled generation makes diffusion LMs practical. Instead of generating random text, you can specify attributes like sentiment, topic, style, or even hard constraints like keyword inclusion.

### Approach 1: Classifier Guidance

**Idea**: Train a separate classifier on noisy data, then use its gradients to steer generation.

```
At each denoising step t:
1. Diffusion model predicts: p(x_0 | x_t)
2. Classifier computes: p(y | x_t)   (y = desired attribute)
3. Shift prediction: p(x_0 | x_t, y) ~ p(x_0 | x_t) * p(y | x_t)^w
```

The classifier must work on partially masked inputs at any noise level:

```python
from classifier_guidance import NoisyClassifier, ClassifierGuidedSampler

# Train classifier on noisy data
classifier = NoisyClassifier(vocab_size=30522, num_classes=2)
classifier = train_noisy_classifier(classifier, train_data, mask_token_id=103)

# Use for guided sampling
sampler = ClassifierGuidedSampler(
    diffusion_model=model,
    classifier=classifier,
    mask_token_id=103,
    guidance_scale=3.0,  # Higher = stronger control
)
tokens = sampler.sample(target_class=1, seq_len=64, batch_size=4)
```

**Pros**: Can use any pre-trained classifier. Separates generation from control.
**Cons**: Requires a classifier trained on noisy data. Extra compute per step.

### Approach 2: Classifier-Free Guidance (CFG)

**Idea**: Train a single model that handles both conditional and unconditional generation. No separate classifier needed.

```
During training: randomly drop the condition 10-15% of the time
During inference: guided = (1 + w) * conditional - w * unconditional
```

```python
from classifier_free import ClassifierFreeDiffusion, ClassifierFreeSampler

model = ClassifierFreeDiffusion(
    vocab_size=30522,
    num_classes=2,
    cond_drop_prob=0.1,  # 10% condition dropout
)

# Train with condition dropout (happens automatically)
trainer = ClassifierFreeTrainer(model)
trainer.train_step(x_0, condition=labels)

# Sample with guidance
sampler = ClassifierFreeSampler(model, guidance_scale=3.0)
tokens = sampler.sample(condition=1, seq_len=64)
```

**Pros**: No separate classifier. Often higher quality. Simple to implement.
**Cons**: Must be trained from scratch with condition dropout. Only works for conditions seen during training.

### Approach 3: Plug-and-Play Guidance

**Idea**: Use any differentiable function as a constraint. No training required for the constraint itself.

```python
from plug_and_play import PlugAndPlaySampler, make_keyword_constraint

# Keyword constraint: generated text must contain specific words
constraint = make_keyword_constraint(
    target_token_ids=[2293, 3407],  # e.g., "happy", "morning"
    vocab_size=30522,
)

sampler = PlugAndPlaySampler(model, mask_token_id=103, guidance_scale=5.0)
tokens = sampler.sample(constraint_fn=constraint, seq_len=64)
```

**Pros**: Works with any differentiable constraint. No training needed for the constraint. Highly flexible.
**Cons**: Slower (multiple gradient steps per denoising step). Constraint must be differentiable.

### Comparison

| Method | Separate Training | Flexibility | Speed | Quality |
|--------|------------------|-------------|-------|---------|
| Classifier guidance | Yes (classifier) | Medium | Medium | Good |
| Classifier-free | No | Low (fixed classes) | Fast | Best |
| Plug-and-play | No | High | Slow | Good |

## Paper Reference

- Dhariwal & Nichol (2021), "Diffusion Models Beat GANs on Image Synthesis" - [arXiv:2105.05233](https://arxiv.org/abs/2105.05233)
- Ho & Salimans (2022), "Classifier-Free Diffusion Guidance" - [arXiv:2207.12598](https://arxiv.org/abs/2207.12598)
- Li et al. (2022), "Diffusion-LM Improves Controllable Text Generation" - [arXiv:2205.14217](https://arxiv.org/abs/2205.14217)

## Exercises

1. **Classifier Guidance**: Train a noisy classifier for sentiment on a small dataset. Compare generation with guidance_scale=1.0 vs 5.0 vs 10.0. What happens at very high guidance scales?

2. **CFG Training**: Train a classifier-free model with different `cond_drop_prob` values (0.05, 0.1, 0.2). How does this affect the quality/diversity tradeoff?

3. **Plug-and-Play**: Create a custom constraint function that encourages generated text to have a specific average word length. Use `make_keyword_constraint` as a template.

4. **Comparison**: Generate 50 samples with each method for the same target attribute. Qualitatively compare coherence and attribute adherence.

## What's Next

In [Lesson 03](../lesson03-infilling-constrained/), you will learn about infilling and constrained generation -- capabilities unique to diffusion LMs that autoregressive models cannot easily replicate.
