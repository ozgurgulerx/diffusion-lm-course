"""Complete training script for diffusion language models on custom data.

Supports MDLM (Masked Discrete Language Model) training with:
- Configurable model architecture (Transformer backbone)
- Wandb logging for training monitoring
- Checkpointing and resume
- Mixed precision training
- Learning rate scheduling

Usage:
    python train_custom.py \
        --data_path data/my_corpus.txt \
        --output_dir checkpoints/my_model \
        --tokenizer bert-base-uncased \
        --max_seq_len 128 \
        --epochs 50 \
        --batch_size 32

References:
    - MDLM (Sahoo et al., 2024): https://arxiv.org/abs/2406.07524
"""

import argparse
import math
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from data_pipeline import CustomDataPipeline


# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------

class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal positional embeddings for timestep and position encoding."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t.float().unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb


class DiffusionTransformerBlock(nn.Module):
    """Single Transformer block with pre-norm and timestep conditioning.

    Uses pre-LayerNorm convention and injects timestep information
    via an additive bias after the first normalization.
    """

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.time_proj = nn.Linear(d_model, d_model)

    def forward(
        self, x: torch.Tensor, t_emb: torch.Tensor, mask: torch.Tensor = None
    ) -> torch.Tensor:
        # Add timestep conditioning
        t_bias = self.time_proj(t_emb).unsqueeze(1)  # (B, 1, D)
        h = self.norm1(x + t_bias)
        h2, _ = self.self_attn(h, h, h, key_padding_mask=mask)
        x = x + h2
        x = x + self.ff(self.norm2(x))
        return x


class MDLMTransformer(nn.Module):
    """Transformer backbone for Masked Discrete Language Model (MDLM).

    Predicts the clean token distribution p(x_0 | x_t, t) given a
    masked sequence x_t and diffusion timestep t.

    Args:
        vocab_size: Number of tokens in the vocabulary.
        d_model: Hidden dimension of the Transformer.
        nhead: Number of attention heads.
        num_layers: Number of Transformer blocks.
        max_seq_len: Maximum sequence length.
        dropout: Dropout probability.
        mask_token_id: Token ID for the [MASK] absorbing state.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        max_seq_len: int = 128,
        dropout: float = 0.1,
        mask_token_id: int = 103,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.mask_token_id = mask_token_id

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.time_embedding = nn.Sequential(
            SinusoidalPositionEmbedding(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

        self.blocks = nn.ModuleList(
            [
                DiffusionTransformerBlock(d_model, nhead, d_model * 4, dropout)
                for _ in range(num_layers)
            ]
        )

        self.final_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        attention_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Forward pass: predict clean token logits.

        Args:
            x_t: Masked token IDs, shape (B, L).
            t: Diffusion timesteps, shape (B,).
            attention_mask: 1 for real tokens, 0 for padding, shape (B, L).

        Returns:
            Logits over vocabulary, shape (B, L, V).
        """
        B, L = x_t.shape

        # Embeddings
        positions = torch.arange(L, device=x_t.device).unsqueeze(0).expand(B, -1)
        h = self.token_embedding(x_t) + self.position_embedding(positions)
        t_emb = self.time_embedding(t)

        # Padding mask for attention (True = ignore)
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = attention_mask == 0

        # Transformer blocks
        for block in self.blocks:
            h = block(h, t_emb, key_padding_mask)

        h = self.final_norm(h)
        logits = self.output_proj(h)  # (B, L, V)
        return logits


# ---------------------------------------------------------------------------
# MDLM Training logic
# ---------------------------------------------------------------------------

class MDLMTrainer:
    """Trainer for Masked Discrete Language Models.

    Implements the MDLM training objective: mask tokens with a schedule,
    then train the model to predict the original tokens at masked positions.

    Args:
        model: MDLMTransformer instance.
        mask_token_id: Token ID used for masking.
        num_timesteps: Number of discrete diffusion timesteps.
        lr: Learning rate.
        weight_decay: AdamW weight decay.
        warmup_steps: Linear warmup steps for the learning rate.
        use_wandb: Whether to log metrics to wandb.
        device: Device to train on.
    """

    def __init__(
        self,
        model: MDLMTransformer,
        mask_token_id: int,
        num_timesteps: int = 1000,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 500,
        use_wandb: bool = False,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.mask_token_id = mask_token_id
        self.num_timesteps = num_timesteps
        self.device = device
        self.use_wandb = use_wandb
        self.warmup_steps = warmup_steps

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.global_step = 0

        # Noise schedule: linear from small masking rate to high masking rate
        self.mask_rates = torch.linspace(0.0, 1.0, num_timesteps + 1)[1:]

    def _get_lr_scale(self) -> float:
        """Linear warmup then constant learning rate."""
        if self.global_step < self.warmup_steps:
            return self.global_step / max(1, self.warmup_steps)
        return 1.0

    def _apply_mask(
        self, x_0: torch.Tensor, t: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Apply absorbing-state masking to create x_t from x_0.

        Args:
            x_0: Clean token IDs, shape (B, L).
            t: Timestep per sample, shape (B,), values in [0, T-1].
            attention_mask: 1 for real tokens, shape (B, L).

        Returns:
            Masked token IDs x_t, shape (B, L).
        """
        mask_rate = self.mask_rates[t].to(self.device)  # (B,)
        mask_prob = mask_rate.unsqueeze(1)  # (B, 1)

        # Randomly mask tokens (only real tokens, not padding)
        rand = torch.rand_like(x_0.float())
        should_mask = (rand < mask_prob) & (attention_mask == 1)

        x_t = x_0.clone()
        x_t[should_mask] = self.mask_token_id
        return x_t

    def train_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """Execute a single training step.

        Args:
            batch: Dictionary with input_ids, attention_mask, length.

        Returns:
            Dictionary of metrics (loss, accuracy, etc.).
        """
        self.model.train()

        x_0 = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        B = x_0.shape[0]

        # Sample random timesteps
        t = torch.randint(0, self.num_timesteps, (B,), device=self.device)

        # Create masked input
        x_t = self._apply_mask(x_0, t, attention_mask)

        # Forward pass
        logits = self.model(x_t, t.float(), attention_mask)

        # Compute loss only at masked positions
        is_masked = (x_t == self.mask_token_id) & (attention_mask == 1)

        if is_masked.sum() == 0:
            # Edge case: nothing was masked (very low t)
            loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            accuracy = 1.0
        else:
            # Cross-entropy at masked positions
            loss = F.cross_entropy(
                logits[is_masked], x_0[is_masked], reduction="mean"
            )
            # Accuracy at masked positions
            preds = logits[is_masked].argmax(dim=-1)
            accuracy = (preds == x_0[is_masked]).float().mean().item()

        # Backward pass with LR warmup
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

        # Apply LR scaling
        lr_scale = self._get_lr_scale()
        for param_group in self.optimizer.param_groups:
            param_group["lr_scale"] = lr_scale

        self.optimizer.step()
        self.global_step += 1

        metrics = {
            "loss": loss.item(),
            "accuracy": accuracy,
            "masked_tokens": is_masked.sum().item(),
            "mask_rate": self.mask_rates[t].mean().item(),
            "lr_scale": lr_scale,
        }

        if self.use_wandb:
            try:
                import wandb

                wandb.log(metrics, step=self.global_step)
            except ImportError:
                pass

        return metrics

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> dict[str, float]:
        """Run validation on a held-out set.

        Args:
            dataloader: Validation DataLoader.

        Returns:
            Dictionary of average metrics.
        """
        self.model.eval()
        total_loss = 0.0
        total_acc = 0.0
        num_batches = 0

        for batch in dataloader:
            x_0 = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            B = x_0.shape[0]

            # Use middle timestep for consistent validation
            t = torch.full((B,), self.num_timesteps // 2, device=self.device)
            x_t = self._apply_mask(x_0, t, attention_mask)

            logits = self.model(x_t, t.float(), attention_mask)
            is_masked = (x_t == self.mask_token_id) & (attention_mask == 1)

            if is_masked.sum() > 0:
                loss = F.cross_entropy(
                    logits[is_masked], x_0[is_masked], reduction="mean"
                )
                preds = logits[is_masked].argmax(dim=-1)
                acc = (preds == x_0[is_masked]).float().mean().item()
                total_loss += loss.item()
                total_acc += acc
                num_batches += 1

        if num_batches == 0:
            return {"val_loss": 0.0, "val_accuracy": 0.0}

        return {
            "val_loss": total_loss / num_batches,
            "val_accuracy": total_acc / num_batches,
        }

    @torch.no_grad()
    def sample(
        self,
        seq_len: int = 64,
        batch_size: int = 4,
        num_steps: int = 50,
    ) -> torch.Tensor:
        """Generate text by iteratively unmasking tokens.

        Starts from fully masked sequence and progressively unmasks.

        Args:
            seq_len: Length of sequence to generate.
            batch_size: Number of sequences to generate.
            num_steps: Number of denoising steps.

        Returns:
            Generated token IDs, shape (batch_size, seq_len).
        """
        self.model.eval()

        # Start fully masked
        x = torch.full(
            (batch_size, seq_len), self.mask_token_id,
            dtype=torch.long, device=self.device,
        )
        attention_mask = torch.ones(batch_size, seq_len, device=self.device)

        # Reverse diffusion: go from high t to low t
        timesteps = torch.linspace(
            self.num_timesteps - 1, 0, num_steps, dtype=torch.long
        )

        for step_t in timesteps:
            t = torch.full((batch_size,), step_t.item(), device=self.device)
            logits = self.model(x, t.float(), attention_mask)

            # Only update currently masked positions
            is_masked = x == self.mask_token_id

            # Sample from predicted distribution
            probs = F.softmax(logits, dim=-1)
            sampled = torch.multinomial(
                probs.view(-1, probs.shape[-1]), num_samples=1
            ).view(batch_size, seq_len)

            # Determine how many tokens to unmask at this step
            if step_t > 0:
                current_rate = self.mask_rates[int(step_t.item())]
                next_idx = max(0, int(step_t.item()) - 1)
                next_rate = self.mask_rates[next_idx]
                unmask_prob = (current_rate - next_rate) / max(current_rate, 1e-8)
            else:
                unmask_prob = 1.0

            # Randomly unmask a fraction of masked tokens
            unmask_mask = (torch.rand_like(x.float()) < unmask_prob) & is_masked
            x = torch.where(unmask_mask, sampled, x)

        # Final pass: unmask everything remaining
        if (x == self.mask_token_id).any():
            t = torch.zeros(batch_size, device=self.device)
            logits = self.model(x, t, attention_mask)
            is_masked = x == self.mask_token_id
            preds = logits.argmax(dim=-1)
            x = torch.where(is_masked, preds, x)

        return x

    def save_checkpoint(self, path: str, epoch: int):
        """Save model checkpoint.

        Args:
            path: Directory to save the checkpoint.
            epoch: Current epoch number.
        """
        os.makedirs(path, exist_ok=True)
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "epoch": epoch,
        }
        torch.save(checkpoint, os.path.join(path, f"checkpoint_epoch{epoch}.pt"))
        # Also save as latest
        torch.save(checkpoint, os.path.join(path, "checkpoint_latest.pt"))
        print(f"Saved checkpoint at epoch {epoch}, step {self.global_step}")

    def load_checkpoint(self, path: str) -> int:
        """Load a checkpoint.

        Args:
            path: Path to checkpoint file.

        Returns:
            Epoch number from the checkpoint.
        """
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.global_step = checkpoint["global_step"]
        return checkpoint["epoch"]


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(args):
    """Main training function.

    Args:
        args: Parsed command-line arguments.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Initialize wandb if requested
    if args.use_wandb:
        try:
            import wandb

            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                config=vars(args),
            )
        except ImportError:
            print("wandb not installed, skipping logging")
            args.use_wandb = False

    # Build data pipeline
    print(f"Loading data from {args.data_path}...")
    pipeline = CustomDataPipeline(
        tokenizer_name=args.tokenizer,
        max_seq_len=args.max_seq_len,
        chunk_overlap=args.chunk_overlap,
    )
    dataset = pipeline.build_from_file(args.data_path, min_length=args.min_length)
    print(f"Dataset size: {len(dataset)} chunks")

    # Train/val split
    val_size = max(1, int(len(dataset) * args.val_fraction))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    print(f"Train: {train_size} samples, Val: {val_size} samples")

    # Build model
    model = MDLMTransformer(
        vocab_size=pipeline.vocab_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        max_seq_len=args.max_seq_len,
        dropout=args.dropout,
        mask_token_id=pipeline.mask_token_id,
    )
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Build trainer
    trainer = MDLMTrainer(
        model=model,
        mask_token_id=pipeline.mask_token_id,
        num_timesteps=args.num_timesteps,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        use_wandb=args.use_wandb,
        device=device,
    )

    # Resume from checkpoint if available
    start_epoch = 0
    if args.resume:
        ckpt_path = os.path.join(args.output_dir, "checkpoint_latest.pt")
        if os.path.exists(ckpt_path):
            start_epoch = trainer.load_checkpoint(ckpt_path) + 1
            print(f"Resumed from epoch {start_epoch}")

    # Training loop
    best_val_loss = float("inf")
    for epoch in range(start_epoch, args.epochs):
        epoch_loss = 0.0
        epoch_acc = 0.0
        num_batches = 0
        start_time = time.time()

        for batch in train_loader:
            metrics = trainer.train_step(batch)
            epoch_loss += metrics["loss"]
            epoch_acc += metrics["accuracy"]
            num_batches += 1

        epoch_time = time.time() - start_time
        avg_loss = epoch_loss / max(1, num_batches)
        avg_acc = epoch_acc / max(1, num_batches)

        # Validation
        val_metrics = trainer.validate(val_loader)

        print(
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"Loss: {avg_loss:.4f} | Acc: {avg_acc:.3f} | "
            f"Val Loss: {val_metrics['val_loss']:.4f} | "
            f"Val Acc: {val_metrics['val_accuracy']:.3f} | "
            f"Time: {epoch_time:.1f}s"
        )

        if args.use_wandb:
            try:
                import wandb

                wandb.log(
                    {
                        "epoch": epoch + 1,
                        "epoch_loss": avg_loss,
                        "epoch_accuracy": avg_acc,
                        **val_metrics,
                    }
                )
            except ImportError:
                pass

        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            trainer.save_checkpoint(args.output_dir, epoch + 1)

        # Save best model
        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            trainer.save_checkpoint(os.path.join(args.output_dir, "best"), epoch + 1)

        # Generate samples periodically
        if (epoch + 1) % args.sample_every == 0:
            print("\n--- Generated samples ---")
            samples = trainer.sample(
                seq_len=args.max_seq_len, batch_size=2, num_steps=50
            )
            for i, s in enumerate(samples):
                text = pipeline.tokenizer.decode(
                    s.cpu().tolist(), skip_special_tokens=True
                )
                print(f"  Sample {i + 1}: {text[:200]}")
            print("---\n")

    print("Training complete!")
    if args.use_wandb:
        try:
            import wandb

            wandb.finish()
        except ImportError:
            pass


def main():
    parser = argparse.ArgumentParser(
        description="Train a diffusion language model on custom data"
    )

    # Data arguments
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to data file"
    )
    parser.add_argument("--tokenizer", type=str, default="bert-base-uncased")
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--min_length", type=int, default=5)
    parser.add_argument("--chunk_overlap", type=int, default=16)
    parser.add_argument("--val_fraction", type=float, default=0.1)

    # Model arguments
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num_timesteps", type=int, default=1000)

    # Training arguments
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=500)

    # Logging and saving
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--sample_every", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="diffusion-lm-custom")
    parser.add_argument("--wandb_run_name", type=str, default=None)

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
