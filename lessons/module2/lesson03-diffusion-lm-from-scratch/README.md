# Lesson 03: Diffusion-LM from Scratch

**Build a complete continuous diffusion language model: embed tokens, train a denoiser, and generate text.**

## Prerequisites

- Module 2, Lesson 01: From Tokens to Embedding Space
- Module 2, Lesson 02: Forward and Reverse SDE

## Learning Objective

After this lesson you will be able to implement a full Diffusion-LM that embeds tokens into continuous space, trains a transformer to denoise noisy embeddings using MSE loss, and generates new text by reverse SDE sampling followed by rounding.

## Concept

### The Diffusion-LM Architecture

Diffusion-LM combines three components we have already studied:

1. **Token Embedding** (Lesson 01): Map discrete tokens to continuous vectors.
2. **VP-SDE Forward Process** (Lesson 02): Add noise at random timesteps during training.
3. **Transformer Denoiser**: Predict the clean embedding from the noisy one, conditioned on the timestep.

The training loop is simple:

```
1. Sample a batch of token sequences
2. Embed them: x_0 = Embed(tokens)
3. Sample random timestep t ~ Uniform(0, 1)
4. Add noise: x_t = sqrt(alpha_bar(t)) * x_0 + sqrt(1 - alpha_bar(t)) * eps
5. Predict clean: x_0_pred = Denoiser(x_t, t)
6. Loss = MSE(x_0_pred, x_0)
```

### The Denoiser Architecture

The denoiser is a transformer that takes noisy embeddings + timestep and outputs predicted clean embeddings. Key design choices:

- **x_0-prediction**: The model directly predicts the clean embedding (not the noise). This is the formulation used in the original Diffusion-LM paper.
- **Time conditioning**: The timestep t is embedded via sinusoidal encoding + MLP, then added to the input.
- **Positional encoding**: Standard sinusoidal positional encoding for sequence position.

```python
from src.diffusion_lm import DiffusionLM

model = DiffusionLM(
    vocab_size=100,
    embed_dim=64,
    n_heads=4,
    n_layers=4,
    seq_len=64,
)

# Training step
token_ids = torch.randint(0, 100, (8, 64))
loss = model.train_loss(token_ids)
loss.backward()
```

### Training on Real Text

```python
from shared.datasets.text import SimpleTokenizer, TextDataset, load_text_dataset
from torch.utils.data import DataLoader

# Load data
texts = load_text_dataset("tinystories", max_samples=5000)
tokenizer = SimpleTokenizer(texts, level="word", max_vocab=1000)
dataset = TextDataset(texts, tokenizer, seq_len=32)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Create model
model = DiffusionLM(
    vocab_size=tokenizer.vocab_size,
    embed_dim=64, n_heads=4, n_layers=4, seq_len=32,
)

# Train
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
for epoch in range(10):
    for batch in dataloader:
        loss = model.train_loss(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### Generating Text

Generation runs the reverse SDE: start from Gaussian noise, iteratively denoise, then round to tokens.

```python
# Generate 4 sequences
token_ids = model.sample(batch_size=4, n_steps=100)

# Decode to text
for i in range(4):
    text = tokenizer.decode(token_ids[i].tolist())
    print(text)
```

The sampling process:
1. Start with x_T ~ N(0, I)
2. For each reverse step t -> t - dt:
   - Predict x_0 from x_t using the denoiser
   - Compute the score from the x_0-prediction
   - Take a reverse SDE step
3. After all steps, do a final denoise at t ≈ 0
4. Round continuous vectors to nearest token embeddings

### Understanding the Loss

The simplified MSE loss works because:
- At large t (high noise), the model learns the overall data distribution (which tokens are common, what patterns exist).
- At small t (low noise), the model learns fine-grained corrections (which specific token goes here).
- The x_0-prediction formulation means the loss directly penalizes incorrect token predictions -- if x_0_pred is close to x_0, rounding will recover the correct tokens.

## Paper Link

- Li et al., "Diffusion-LM Improves Controllable Text Generation" (2022) -- the full Diffusion-LM model (Sections 3-4).
- The paper also introduces controllable generation via classifier guidance, which we will implement in the lab.

## Exercises

1. **Train and generate**: Train a DiffusionLM on TinyStories (word-level, vocab=500, seq_len=32) for 20 epochs. Generate 10 samples and inspect the output. Is the text coherent? What patterns does the model capture?

2. **Effect of diffusion steps**: Generate samples with n_steps = 10, 50, 200, 500. How does sample quality change with more reverse steps? Is there a point of diminishing returns?

3. **x_0-prediction vs noise-prediction**: Modify the training loss to predict noise instead of x_0 (change `MSE(x_0_pred, x_0)` to `MSE(noise_pred, noise)`). Train both variants for the same number of epochs. Compare generation quality. Which converges faster?

## What's Next

In [Lesson 04: CDCD](../lesson04-cdcd/), we will see how CDCD improves on Diffusion-LM by adding a categorical loss term that encourages the denoiser to produce outputs close to valid token embeddings.
