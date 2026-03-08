"""Solution for Lab 01: End-to-End Diffusion LM Project.

Task: Build a poetry generation pipeline with style control.
- Train a masked diffusion LM on a poetry corpus
- Implement classifier-free guidance for style control
- Package as a reusable generation pipeline

This solution demonstrates the complete workflow from data to deployment.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer


# ---------------------------------------------------------------------------
# Step 1: Data Preparation
# ---------------------------------------------------------------------------

class PoetryDataset(torch.utils.data.Dataset):
    """Dataset for poetry with style labels.

    Each sample is a poem tokenized and padded, with a style label
    (e.g., 0=romantic, 1=nature, 2=melancholy).

    Args:
        poems: List of (text, style_label) tuples.
        tokenizer: HuggingFace tokenizer.
        max_seq_len: Maximum sequence length.
    """

    def __init__(
        self,
        poems: list[tuple[str, int]],
        tokenizer,
        max_seq_len: int = 128,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.samples = []

        for text, label in poems:
            ids = tokenizer.encode(text, add_special_tokens=False, truncation=True,
                                   max_length=max_seq_len)
            if len(ids) >= 5:  # Skip very short poems
                self.samples.append((ids, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ids, label = self.samples[idx]
        length = min(len(ids), self.max_seq_len)

        if len(ids) >= self.max_seq_len:
            padded = ids[: self.max_seq_len]
            mask = [1] * self.max_seq_len
        else:
            pad_len = self.max_seq_len - len(ids)
            padded = ids + [self.tokenizer.pad_token_id or 0] * pad_len
            mask = [1] * len(ids) + [0] * pad_len

        return {
            "input_ids": torch.tensor(padded, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.long),
            "length": torch.tensor(length, dtype=torch.long),
        }


def create_sample_poetry_data() -> list[tuple[str, int]]:
    """Create sample poetry data for demonstration.

    In a real project, you would load a proper poetry corpus.
    Style labels: 0=romantic, 1=nature, 2=melancholy.

    Returns:
        List of (poem_text, style_label) tuples.
    """
    romantic = [
        ("How do I love thee let me count the ways "
         "I love thee to the depth and breadth and height"),
        ("Shall I compare thee to a summer's day "
         "thou art more lovely and more temperate"),
        ("She walks in beauty like the night "
         "of cloudless climes and starry skies"),
        ("My heart leaps up when I behold "
         "a rainbow in the sky so it was when life began"),
        ("Love is not love which alters when it alteration finds "
         "or bends with the remover to remove"),
    ]

    nature = [
        ("I wandered lonely as a cloud that floats on high "
         "o'er vales and hills when all at once I saw a crowd"),
        ("The world is too much with us late and soon "
         "getting and spending we lay waste our powers"),
        ("Season of mists and mellow fruitfulness "
         "close bosom friend of the maturing sun"),
        ("Tyger tyger burning bright in the forests of the night "
         "what immortal hand or eye could frame thy fearful symmetry"),
        ("I think that I shall never see a poem lovely as a tree "
         "a tree whose hungry mouth is pressed against the earth"),
    ]

    melancholy = [
        ("Do not go gentle into that good night "
         "old age should burn and rave at close of day"),
        ("Because I could not stop for death "
         "he kindly stopped for me the carriage held but just ourselves"),
        ("I have been one acquainted with the night "
         "I have walked out in rain and back in rain"),
        ("The fog comes on little cat feet it sits looking "
         "over harbor and city on silent haunches and then moves on"),
        ("Whose woods these are I think I know "
         "his house is in the village though he will not see me stopping here"),
    ]

    data = []
    for poem in romantic:
        data.append((poem, 0))
    for poem in nature:
        data.append((poem, 1))
    for poem in melancholy:
        data.append((poem, 2))

    return data


# ---------------------------------------------------------------------------
# Step 2: Model with Classifier-Free Guidance
# ---------------------------------------------------------------------------

class PoetryDiffusionModel(nn.Module):
    """Diffusion LM for poetry with style conditioning.

    Supports classifier-free guidance by randomly dropping
    the style condition during training.

    Args:
        vocab_size: Vocabulary size.
        d_model: Hidden dimension.
        nhead: Attention heads.
        num_layers: Transformer layers.
        num_styles: Number of style classes.
        max_seq_len: Maximum sequence length.
        cond_drop_prob: Condition dropout probability for CFG training.
        mask_token_id: Absorbing state token ID.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 4,
        num_styles: int = 3,
        max_seq_len: int = 128,
        cond_drop_prob: float = 0.15,
        mask_token_id: int = 103,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.mask_token_id = mask_token_id
        self.cond_drop_prob = cond_drop_prob
        self.null_style_id = num_styles  # Last ID = unconditional

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.time_emb = nn.Sequential(
            nn.Linear(1, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )
        self.style_emb = nn.Embedding(num_styles + 1, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model * 4, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        style: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        force_uncond: bool = False,
    ) -> torch.Tensor:
        B, L = x_t.shape
        device = x_t.device

        pos = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
        h = self.token_emb(x_t) + self.pos_emb(pos)

        t_emb = self.time_emb(t.float().unsqueeze(-1))

        if style is None or force_uncond:
            s_ids = torch.full((B,), self.null_style_id, device=device, dtype=torch.long)
        else:
            s_ids = style.clone()
            if self.training:
                drop = torch.rand(B, device=device) < self.cond_drop_prob
                s_ids[drop] = self.null_style_id

        s_emb = self.style_emb(s_ids)
        h = h + (t_emb + s_emb).unsqueeze(1)

        kpm = None
        if attention_mask is not None:
            kpm = attention_mask == 0

        h = self.encoder(h, src_key_padding_mask=kpm)
        h = self.norm(h)
        return self.out_proj(h)


# ---------------------------------------------------------------------------
# Step 3: Training Loop
# ---------------------------------------------------------------------------

def train_poetry_model(
    model: PoetryDiffusionModel,
    dataset: PoetryDataset,
    epochs: int = 30,
    batch_size: int = 8,
    lr: float = 3e-4,
    num_timesteps: int = 500,
    device: str = "cpu",
) -> list[float]:
    """Train the poetry diffusion model.

    Args:
        model: PoetryDiffusionModel instance.
        dataset: PoetryDataset with poems and labels.
        epochs: Number of training epochs.
        batch_size: Batch size.
        lr: Learning rate.
        num_timesteps: Number of diffusion timesteps.
        device: Training device.

    Returns:
        List of per-epoch average losses.
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    mask_rates = torch.linspace(0.0, 1.0, num_timesteps + 1)[1:]

    # Train/val split
    val_size = max(1, len(dataset) // 5)
    train_size = len(dataset) - val_size
    train_ds, _ = random_split(dataset, [train_size, val_size])
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    loss_history = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in loader:
            x_0 = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            B, L = x_0.shape

            # Random timesteps
            t_idx = torch.randint(0, num_timesteps, (B,), device=device)
            rate = mask_rates[t_idx].unsqueeze(1).to(device)

            # Apply masking
            noise = torch.rand(B, L, device=device)
            should_mask = (noise < rate) & (attn_mask == 1)
            x_t = x_0.clone()
            x_t[should_mask] = model.mask_token_id

            # Forward
            logits = model(x_t, t_idx.float(), labels, attn_mask)

            # Loss at masked positions
            if should_mask.sum() > 0:
                loss = F.cross_entropy(logits[should_mask], x_0[should_mask])
            else:
                loss = torch.tensor(0.0, device=device, requires_grad=True)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(1, n_batches)
        loss_history.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.4f}")

    return loss_history


# ---------------------------------------------------------------------------
# Step 4: Guided Generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_poetry(
    model: PoetryDiffusionModel,
    style: int,
    tokenizer,
    num_samples: int = 4,
    seq_len: int = 64,
    num_steps: int = 50,
    num_timesteps: int = 500,
    guidance_scale: float = 3.0,
    temperature: float = 0.9,
    device: str = "cpu",
) -> list[str]:
    """Generate poetry with classifier-free guidance.

    Args:
        model: Trained PoetryDiffusionModel.
        style: Style label (0=romantic, 1=nature, 2=melancholy).
        tokenizer: Tokenizer for decoding.
        num_samples: Number of poems to generate.
        seq_len: Sequence length.
        num_steps: Denoising steps.
        num_timesteps: Total diffusion timesteps.
        guidance_scale: CFG guidance weight.
        temperature: Sampling temperature.
        device: Generation device.

    Returns:
        List of generated poem strings.
    """
    model = model.to(device)
    # Switch to inference mode
    was_training = model.training
    model.train(False)
    mask_rates = torch.linspace(0.0, 1.0, num_timesteps + 1)[1:]

    x = torch.full(
        (num_samples, seq_len), model.mask_token_id,
        dtype=torch.long, device=device,
    )
    attn_mask = torch.ones(num_samples, seq_len, device=device)
    style_t = torch.full((num_samples,), style, dtype=torch.long, device=device)

    timesteps = torch.linspace(num_timesteps - 1, 0, num_steps, dtype=torch.long)

    for step_t in timesteps:
        t = torch.full((num_samples,), step_t.item(), device=device)

        # Conditional prediction
        cond_logits = model(x, t, style_t, attn_mask, force_uncond=False)
        # Unconditional prediction
        uncond_logits = model(x, t, style_t, attn_mask, force_uncond=True)

        # CFG interpolation
        guided = (1 + guidance_scale) * cond_logits - guidance_scale * uncond_logits

        is_masked = x == model.mask_token_id
        probs = F.softmax(guided / temperature, dim=-1)
        sampled = torch.multinomial(
            probs.view(-1, probs.shape[-1]), num_samples=1
        ).view(num_samples, seq_len)

        if step_t > 0:
            curr = mask_rates[int(step_t.item())]
            nxt = mask_rates[max(0, int(step_t.item()) - 1)]
            unmask_prob = (curr - nxt) / max(curr, 1e-8)
        else:
            unmask_prob = 1.0

        unmask = (torch.rand_like(x.float()) < unmask_prob) & is_masked
        x = torch.where(unmask, sampled, x)

    # Final unmasking
    if (x == model.mask_token_id).any():
        t = torch.zeros(num_samples, device=device)
        logits = model(x, t, style_t, attn_mask)
        preds = logits.argmax(dim=-1)
        x = torch.where(x == model.mask_token_id, preds, x)

    model.train(was_training)

    results = []
    for i in range(num_samples):
        text = tokenizer.decode(x[i].cpu().tolist(), skip_special_tokens=True)
        results.append(text)
    return results


# ---------------------------------------------------------------------------
# Step 5: Complete Pipeline
# ---------------------------------------------------------------------------

class PoetryPipeline:
    """Reusable end-to-end pipeline for poetry generation.

    Packages the model, tokenizer, and generation logic into a single
    callable object.

    Args:
        model: Trained PoetryDiffusionModel.
        tokenizer: HuggingFace tokenizer.
        num_timesteps: Diffusion timesteps.
        device: Device.
    """

    STYLE_NAMES = {0: "romantic", 1: "nature", 2: "melancholy"}

    def __init__(self, model, tokenizer, num_timesteps=500, device="cpu"):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.num_timesteps = num_timesteps
        self.device = device

    def __call__(
        self,
        style: str = "romantic",
        num_samples: int = 4,
        guidance_scale: float = 3.0,
        temperature: float = 0.9,
    ) -> list[str]:
        """Generate poems in a given style.

        Args:
            style: Style name ("romantic", "nature", or "melancholy").
            num_samples: Number of poems.
            guidance_scale: CFG guidance strength.
            temperature: Sampling temperature.

        Returns:
            List of generated poems.
        """
        style_to_id = {v: k for k, v in self.STYLE_NAMES.items()}
        style_id = style_to_id.get(style.lower(), 0)

        return generate_poetry(
            model=self.model,
            style=style_id,
            tokenizer=self.tokenizer,
            num_samples=num_samples,
            num_timesteps=self.num_timesteps,
            guidance_scale=guidance_scale,
            temperature=temperature,
            device=self.device,
        )

    def save(self, path: str):
        """Save pipeline to disk."""
        os.makedirs(path, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(path, "model.pt"))
        self.tokenizer.save_pretrained(os.path.join(path, "tokenizer"))
        print(f"Pipeline saved to {path}")


# ---------------------------------------------------------------------------
# Main: run the complete project
# ---------------------------------------------------------------------------

def main():
    """Run the complete end-to-end poetry generation project."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Step 1: Prepare data
    print("\n=== Step 1: Data Preparation ===")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    poems = create_sample_poetry_data()
    dataset = PoetryDataset(poems, tokenizer, max_seq_len=64)
    print(f"Dataset: {len(dataset)} poems, 3 styles")

    # Step 2: Build model
    print("\n=== Step 2: Build Model ===")
    model = PoetryDiffusionModel(
        vocab_size=len(tokenizer),
        d_model=256,
        nhead=4,
        num_layers=4,
        num_styles=3,
        max_seq_len=64,
        mask_token_id=tokenizer.mask_token_id,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Step 3: Train
    print("\n=== Step 3: Training ===")
    losses = train_poetry_model(
        model, dataset, epochs=30, batch_size=4,
        lr=3e-4, num_timesteps=500, device=device,
    )

    # Step 4: Generate
    print("\n=== Step 4: Generation ===")
    for style_name, style_id in [("romantic", 0), ("nature", 1), ("melancholy", 2)]:
        print(f"\n--- Style: {style_name} ---")
        generated = generate_poetry(
            model, style_id, tokenizer, num_samples=2,
            seq_len=64, num_timesteps=500, guidance_scale=3.0,
            device=device,
        )
        for i, poem in enumerate(generated):
            print(f"  {i + 1}: {poem[:120]}")

    # Step 5: Package as pipeline
    print("\n=== Step 5: Package Pipeline ===")
    pipeline = PoetryPipeline(model, tokenizer, num_timesteps=500, device=device)
    results = pipeline(style="nature", num_samples=2)
    print("Pipeline output:")
    for r in results:
        print(f"  {r[:120]}")

    print("\nProject complete!")


if __name__ == "__main__":
    main()
