"""Solution for Lab 01: Classifier-Guided Controlled Generation with Diffusion-LM.

This implements:
1. A Diffusion-LM trained on a small text dataset.
2. A sentiment classifier trained on noisy embeddings.
3. Classifier-guided sampling that steers generation toward positive sentiment.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Part 1: Diffusion-LM (reused from lesson03)
# ---------------------------------------------------------------------------
class DiffusionLM(nn.Module):
    """Diffusion language model for continuous text generation."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        n_heads: int = 4,
        n_layers: int = 4,
        seq_len: int = 32,
        beta_min: float = 0.1,
        beta_max: float = 20.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.beta_min = beta_min
        self.beta_max = beta_max

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        nn.init.normal_(self.token_embedding.weight, std=1.0 / math.sqrt(embed_dim))

        pe = torch.zeros(seq_len, embed_dim)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

        self.time_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

        self.input_proj = nn.Linear(embed_dim, embed_dim)
        self.layers = nn.ModuleList(
            [_Block(embed_dim, n_heads, dropout) for _ in range(n_layers)]
        )
        self.final_norm = nn.LayerNorm(embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)

    def embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.token_embedding(token_ids)

    def _time_emb(self, t: torch.Tensor) -> torch.Tensor:
        half = self.embed_dim // 2
        freqs = torch.exp(
            torch.arange(half, device=t.device, dtype=torch.float)
            * (-math.log(10000.0) / half)
        )
        args = t.unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.time_mlp(emb)

    def _alpha_bar(self, t: torch.Tensor) -> torch.Tensor:
        integral = self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * t ** 2
        return torch.exp(-0.5 * integral)

    def forward_diffuse(
        self, x_0: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ab = self._alpha_bar(t).view(-1, 1, 1)
        noise = torch.randn_like(x_0)
        x_t = torch.sqrt(ab) * x_0 + torch.sqrt(1.0 - ab) * noise
        return x_t, noise

    def denoise(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        B, S, D = x_t.shape
        h = self.input_proj(x_t) + self.pe[:, :S, :]
        h = h + self._time_emb(t).unsqueeze(1)
        for layer in self.layers:
            h = layer(h)
        h = self.final_norm(h)
        return self.output_proj(h)

    def train_loss(self, token_ids: torch.Tensor) -> torch.Tensor:
        x_0 = self.embed(token_ids)
        t = torch.rand(x_0.shape[0], device=x_0.device) * 0.999 + 0.001
        x_t, _ = self.forward_diffuse(x_0, t)
        x_0_pred = self.denoise(x_t, t)
        return F.mse_loss(x_0_pred, x_0)

    def round_to_tokens(self, continuous: torch.Tensor) -> torch.Tensor:
        logits = torch.einsum("bsd,vd->bsv", continuous, self.token_embedding.weight)
        return logits.argmax(dim=-1)


# ---------------------------------------------------------------------------
# Part 2: Noisy Embedding Classifier
# ---------------------------------------------------------------------------
class NoisyEmbeddingClassifier(nn.Module):
    """Classifier that operates on noisy embeddings at any noise level.

    Takes noisy token embeddings + timestep and predicts a binary label
    (e.g., positive vs. negative sentiment). The classifier must learn
    to be robust to varying noise levels so it can guide generation
    at every step of the reverse process.

    Args:
        embed_dim: Embedding dimension (must match Diffusion-LM).
        n_classes: Number of output classes.
        hidden_dim: Hidden dimension for the classifier MLP.
    """

    def __init__(self, embed_dim: int = 64, n_classes: int = 2, hidden_dim: int = 128):
        super().__init__()
        self.embed_dim = embed_dim

        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

        # Pool over sequence and classify
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, n_classes),
        )

    def _time_emb(self, t: torch.Tensor) -> torch.Tensor:
        half = self.embed_dim // 2
        freqs = torch.exp(
            torch.arange(half, device=t.device, dtype=torch.float)
            * (-math.log(10000.0) / half)
        )
        args = t.unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.time_mlp(emb)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Classify noisy embeddings.

        Args:
            x: Noisy embeddings (batch, seq_len, embed_dim).
            t: Timestep (batch,) in [0, 1].

        Returns:
            Logits of shape (batch, n_classes).
        """
        t_emb = self._time_emb(t).unsqueeze(1)  # (B, 1, D)
        h = x + t_emb
        h = h.mean(dim=1)  # (B, D)
        return self.classifier(h)

    def get_gradient(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        target_class: int,
    ) -> torch.Tensor:
        """Compute classifier gradient w.r.t. input embeddings.

        This gradient tells us how to modify the embeddings to increase
        the probability of the target class.

        Args:
            x: Noisy embeddings (batch, seq_len, embed_dim). Requires grad.
            t: Timestep (batch,).
            target_class: Class index to maximize.

        Returns:
            Gradient of log p(target_class | x, t) w.r.t. x.
        """
        x_in = x.detach().requires_grad_(True)
        logits = self.forward(x_in, t)
        log_probs = F.log_softmax(logits, dim=-1)
        target_log_prob = log_probs[:, target_class].sum()
        target_log_prob.backward()
        return x_in.grad.detach()


# ---------------------------------------------------------------------------
# Part 3: Classifier-Guided Sampling
# ---------------------------------------------------------------------------
def classifier_guided_sample(
    diffusion_model: DiffusionLM,
    classifier: NoisyEmbeddingClassifier,
    target_class: int,
    guidance_scale: float = 3.0,
    batch_size: int = 1,
    seq_len: int = 32,
    n_steps: int = 100,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Generate text guided by a classifier.

    At each reverse step, compute the classifier gradient and add it
    to the score estimate to steer generation toward the target class.

    Modified score: score_guided = score_unconditional + s * grad_x log p(y|x,t)

    Args:
        diffusion_model: Trained DiffusionLM.
        classifier: Trained NoisyEmbeddingClassifier.
        target_class: Class to guide toward (e.g., 1 for positive).
        guidance_scale: Strength of classifier guidance.
        batch_size: Number of samples.
        seq_len: Sequence length.
        n_steps: Number of reverse steps.
        device: Device.

    Returns:
        Token IDs (batch_size, seq_len).
    """
    if device is None:
        device = next(diffusion_model.parameters()).device

    embed_dim = diffusion_model.embed_dim
    x = torch.randn(batch_size, seq_len, embed_dim, device=device)
    dt = 1.0 / n_steps

    classifier.requires_grad_(False)
    # We need gradients only w.r.t. x inside get_gradient
    diffusion_model.requires_grad_(False)

    for i in range(n_steps):
        t_val = 1.0 - i * dt
        t = torch.full((batch_size,), t_val, device=device)

        # 1. Get unconditional score from diffusion model
        x_0_pred = diffusion_model.denoise(x, t)
        ab = diffusion_model._alpha_bar(t).view(-1, 1, 1)
        score = (torch.sqrt(ab) * x_0_pred - x) / (1.0 - ab + 1e-8)

        # 2. Get classifier gradient
        cls_grad = classifier.get_gradient(x, t, target_class)

        # 3. Guided score
        guided_score = score + guidance_scale * cls_grad

        # 4. Reverse SDE step
        beta_t = diffusion_model.beta_min + t_val * (
            diffusion_model.beta_max - diffusion_model.beta_min
        )
        drift = -0.5 * beta_t * x - beta_t * guided_score
        x = x + drift * (-dt)

        if i < n_steps - 1:
            noise = torch.randn_like(x)
            x = x + math.sqrt(beta_t * dt) * noise

    # Final denoising
    t_final = torch.full((batch_size,), 0.001, device=device)
    x = diffusion_model.denoise(x, t_final)

    diffusion_model.requires_grad_(True)
    classifier.requires_grad_(True)

    return diffusion_model.round_to_tokens(x)


# ---------------------------------------------------------------------------
# Part 4: Labeled Dataset
# ---------------------------------------------------------------------------
class LabeledTextDataset(Dataset):
    """Dataset of (token_ids, label) pairs for classifier training."""

    def __init__(self, sequences: list[torch.Tensor], labels: list[int]):
        assert len(sequences) == len(labels)
        self.sequences = sequences
        self.labels = labels

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        return self.sequences[idx], self.labels[idx]


def create_synthetic_labeled_data(
    texts: list[str],
    tokenizer,
    seq_len: int = 32,
    positive_words: Optional[list[str]] = None,
) -> LabeledTextDataset:
    """Create labeled data using simple keyword-based sentiment.

    Labels texts as positive (1) if they contain any positive keyword,
    negative (0) otherwise.
    """
    if positive_words is None:
        positive_words = [
            "happy", "good", "great", "love", "wonderful", "beautiful",
            "amazing", "fantastic", "excellent", "joy", "smile", "fun",
            "nice", "best", "glad", "kind", "like", "friend",
        ]

    sequences = []
    labels = []
    positive_set = set(w.lower() for w in positive_words)

    for text in texts:
        ids = tokenizer.encode(text)
        if len(ids) < seq_len:
            ids = ids + [tokenizer.pad_id] * (seq_len - len(ids))
        else:
            ids = ids[:seq_len]

        words = text.lower().split()
        label = 1 if any(w in positive_set for w in words) else 0

        sequences.append(torch.tensor(ids, dtype=torch.long))
        labels.append(label)

    return LabeledTextDataset(sequences, labels)


def train_classifier(
    classifier: NoisyEmbeddingClassifier,
    diffusion_model: DiffusionLM,
    labeled_dataset: LabeledTextDataset,
    epochs: int = 10,
    lr: float = 1e-3,
    batch_size: int = 32,
    device: Optional[torch.device] = None,
) -> list[float]:
    """Train the noisy-embedding classifier.

    For each batch, embeds tokens, adds noise at a random timestep,
    then trains the classifier to predict the label.
    """
    if device is None:
        device = next(diffusion_model.parameters()).device

    classifier.to(device)
    classifier.train()
    diffusion_model.requires_grad_(False)

    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)
    dataloader = DataLoader(labeled_dataset, batch_size=batch_size, shuffle=True)

    epoch_losses = []
    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0

        for token_ids, labels in dataloader:
            token_ids = token_ids.to(device)
            labels = labels.to(device)

            # Embed and add noise at random timestep
            with torch.no_grad():
                x_0 = diffusion_model.embed(token_ids)
                t = torch.rand(x_0.shape[0], device=device) * 0.999 + 0.001
                x_t, _ = diffusion_model.forward_diffuse(x_0, t)

            logits = classifier(x_t, t)
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        epoch_losses.append(avg_loss)
        print(f"Classifier Epoch {epoch + 1}/{epochs} - loss: {avg_loss:.4f}")

    diffusion_model.requires_grad_(True)
    return epoch_losses


class _Block(nn.Module):
    def __init__(self, d: int, h: int, drop: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, h, dropout=drop, batch_first=True)
        self.norm2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(
            nn.Linear(d, d * 4), nn.GELU(), nn.Dropout(drop),
            nn.Linear(d * 4, d), nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        x = x + self.ff(self.norm2(x))
        return x
