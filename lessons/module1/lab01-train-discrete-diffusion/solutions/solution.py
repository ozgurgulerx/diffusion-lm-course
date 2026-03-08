"""Lab 01 Solution: Train a discrete diffusion model on TinyStories.

This solution uses MDLM (simpler and often more effective than D3PM).
"""

import sys

sys.path.insert(0, "../../../..")
sys.path.insert(0, "../../lesson04-mdlm")
sys.path.insert(0, "../../lesson05-training-and-sampling")

import torch
from torch.utils.data import DataLoader

from shared.datasets.text import SimpleTokenizer, TextDataset, load_text_dataset
from shared.utils.device import get_device
from shared.utils.seed import set_seed
from src.mdlm import MDLM, MDLMDenoiser
from src.training_utils import compute_perplexity_proxy, get_cosine_schedule_with_warmup

# ============================================================
# Configuration
# ============================================================
SEED = 42
MAX_SAMPLES = 5000
SEQ_LEN = 64
BATCH_SIZE = 64
D_MODEL = 128
N_HEADS = 4
N_LAYERS = 4
NUM_EPOCHS = 30
LR = 3e-4
TEMPERATURE = 0.8
NUM_TIMESTEPS = 100

set_seed(SEED)
device = get_device()
print(f"Using device: {device}")

# ============================================================
# Step 1: Data Preparation
# ============================================================
print("\n--- Step 1: Loading data ---")
texts = load_text_dataset("tinystories", max_samples=MAX_SAMPLES)
tokenizer = SimpleTokenizer(texts, level="char")
dataset = TextDataset(texts, tokenizer, seq_len=SEQ_LEN)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

print(f"Vocab size: {tokenizer.vocab_size}")
print(f"Dataset size: {len(dataset)}")
print(f"Sample: '{tokenizer.decode(dataset[0].tolist())[:60]}...'")

# ============================================================
# Step 2: Model Construction
# ============================================================
print("\n--- Step 2: Building model ---")
denoiser = MDLMDenoiser(
    vocab_size=tokenizer.vocab_size,
    d_model=D_MODEL,
    n_heads=N_HEADS,
    n_layers=N_LAYERS,
    max_seq_len=SEQ_LEN,
    dropout=0.1,
).to(device)

model = MDLM(
    denoiser=denoiser,
    vocab_size=tokenizer.vocab_size,
    mask_token_id=tokenizer.mask_id,
    num_timesteps=NUM_TIMESTEPS,
    schedule_type="cosine",
    device=device,
)

n_params = sum(p.numel() for p in denoiser.parameters())
print(f"Model parameters: {n_params:,}")

# ============================================================
# Step 3: Training
# ============================================================
print("\n--- Step 3: Training ---")
optimizer = torch.optim.Adam(denoiser.parameters(), lr=LR)
total_steps = NUM_EPOCHS * len(dataloader)
scheduler = get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps
)

losses = []
for epoch in range(NUM_EPOCHS):
    epoch_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        batch = batch.to(device)

        loss = model.train_loss(batch)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(denoiser.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        epoch_loss += loss.item()
        n_batches += 1

    avg_loss = epoch_loss / n_batches
    losses.append(avg_loss)
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Loss: {avg_loss:.4f}")

# ============================================================
# Step 4: Generate Samples
# ============================================================
print("\n--- Step 4: Generating samples ---")
generated_samples = model.sample(
    batch_size=10, seq_len=SEQ_LEN, temperature=TEMPERATURE
)
generated_texts = [tokenizer.decode(s.cpu().tolist()) for s in generated_samples]

print("\nGenerated stories:")
for i, text in enumerate(generated_texts):
    print(f"\n[{i+1}] '{text}'")

# ============================================================
# Step 5: Evaluate
# ============================================================
print("\n--- Step 5: Evaluation ---")
eval_batch = next(iter(dataloader)).to(device)
ppl_proxy = compute_perplexity_proxy(
    denoiser, eval_batch, tokenizer.mask_id, device=device
)
print(f"Perplexity proxy: {ppl_proxy:.4f}")

if ppl_proxy < 3.5:
    print("PASSED: Perplexity proxy below threshold.")
else:
    print(f"NOTE: Perplexity proxy {ppl_proxy:.4f} exceeds 3.5. Try training longer.")

# Count samples with English words
common_words = {"the", "a", "is", "and", "in", "on", "to", "it", "was", "he", "she"}
n_good = sum(
    1
    for text in generated_texts
    if any(word in text.lower() for word in common_words)
)
print(f"\nSamples with English words: {n_good}/10")
if n_good >= 5:
    print("PASSED: Sufficient quality.")
else:
    print("NOTE: Fewer than 5 good samples. Try lower temperature or more training.")
