"""Toy autoregressive text generation.

Demonstrates the core idea of autoregressive (left-to-right) generation:
predict one token at a time, conditioning on all previous tokens.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ToyAutoregressiveLM(nn.Module):
    """A minimal autoregressive language model for a tiny vocabulary.

    Uses a single-layer architecture: embedding -> linear -> softmax.
    This is intentionally simple to highlight the generation pattern,
    not model quality.
    """

    def __init__(self, vocab_size: int, embed_dim: int = 16):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.linear = nn.Linear(embed_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute logits for next token prediction.

        Args:
            x: Token IDs of shape (batch, seq_len).

        Returns:
            Logits of shape (batch, seq_len, vocab_size).
        """
        h = self.embedding(x)  # (batch, seq_len, embed_dim)
        return self.linear(h)  # (batch, seq_len, vocab_size)


def autoregressive_generate(
    model: ToyAutoregressiveLM,
    prompt: list[int],
    max_len: int = 8,
    temperature: float = 1.0,
) -> list[int]:
    """Generate tokens one at a time, left to right.

    This is the standard autoregressive generation loop:
    1. Start with a prompt (one or more tokens).
    2. Feed the current sequence into the model.
    3. Sample the next token from the last position's logits.
    4. Append it to the sequence.
    5. Repeat until max_len is reached.

    Args:
        model: A ToyAutoregressiveLM instance.
        prompt: List of starting token IDs.
        max_len: Total sequence length to generate (including prompt).
        temperature: Sampling temperature. Lower = more deterministic.

    Returns:
        Generated sequence as a list of token IDs.
    """
    model.eval()
    sequence = list(prompt)

    with torch.no_grad():
        while len(sequence) < max_len:
            # Feed the entire sequence so far
            x = torch.tensor([sequence], dtype=torch.long)
            logits = model(x)  # (1, seq_len, vocab_size)

            # Take logits from the LAST position only
            next_logits = logits[0, -1, :] / temperature
            probs = F.softmax(next_logits, dim=-1)

            # Sample one token
            next_token = torch.multinomial(probs, num_samples=1).item()
            sequence.append(next_token)

    return sequence


# ---------------------------------------------------------------------------
# Convenience: run a quick demo when executed directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    vocab = ["the", "cat", "sat", "mat"]
    vocab_size = len(vocab)
    id2word = {i: w for i, w in enumerate(vocab)}

    torch.manual_seed(42)
    model = ToyAutoregressiveLM(vocab_size)

    prompt = [0]  # Start with "the"
    generated = autoregressive_generate(model, prompt, max_len=6)
    print("Generated IDs:", generated)
    print("Generated text:", " ".join(id2word[i] for i in generated))
