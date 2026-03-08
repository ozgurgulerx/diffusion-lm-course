"""Conceptual diffusion loop for text generation.

Demonstrates the core idea of diffusion-based generation:
start from random (noisy) tokens and iteratively denoise them
toward a coherent sequence. All positions are updated in parallel.

NOTE: The denoiser here is a dummy -- it does not actually learn.
The point is to illustrate the generation pattern, not model quality.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DummyDenoiser(nn.Module):
    """A dummy denoiser that maps noisy token IDs to logits.

    In a real diffusion LM this would be a trained transformer.
    Here we use a simple embedding + linear layer so we can
    demonstrate the iterative refinement loop.
    """

    def __init__(self, vocab_size: int, embed_dim: int = 16):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.linear = nn.Linear(embed_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict clean-token logits from noisy token IDs.

        Args:
            x: Noisy token IDs of shape (batch, seq_len).

        Returns:
            Logits of shape (batch, seq_len, vocab_size).
        """
        h = self.embedding(x)
        return self.linear(h)


def diffusion_generate(
    model: DummyDenoiser,
    vocab_size: int,
    seq_len: int = 8,
    num_steps: int = 5,
    temperature: float = 1.0,
) -> list[list[int]]:
    """Conceptual diffusion generation loop.

    Unlike autoregressive generation, diffusion works as follows:
    1. Start with a sequence of RANDOM tokens (pure noise).
    2. Feed the entire noisy sequence to the denoiser.
    3. The denoiser predicts what the CLEAN tokens should be (in parallel).
    4. Re-sample tokens from the predicted distribution.
    5. Repeat for a fixed number of steps.

    Each step refines ALL positions simultaneously, moving from noise
    toward a coherent sequence.

    Args:
        model: A DummyDenoiser instance.
        vocab_size: Size of the token vocabulary.
        seq_len: Length of the sequence to generate.
        num_steps: Number of denoising iterations.
        temperature: Sampling temperature.

    Returns:
        List of sequences (one per step), showing the refinement trajectory.
        Each sequence is a list of token IDs.
    """
    model.eval()
    trajectory = []

    # Step 0: start from pure noise -- random token IDs
    x = torch.randint(0, vocab_size, (1, seq_len))
    trajectory.append(x[0].tolist())

    with torch.no_grad():
        for step in range(num_steps):
            # Predict clean-token logits for ALL positions at once
            logits = model(x) / temperature  # (1, seq_len, vocab_size)
            probs = F.softmax(logits, dim=-1)  # (1, seq_len, vocab_size)

            # Re-sample every position in parallel
            x = torch.multinomial(
                probs.view(-1, vocab_size), num_samples=1
            ).view(1, seq_len)

            trajectory.append(x[0].tolist())

    return trajectory


# ---------------------------------------------------------------------------
# Convenience: run a quick demo when executed directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    vocab = ["the", "cat", "sat", "mat"]
    vocab_size = len(vocab)
    id2word = {i: w for i, w in enumerate(vocab)}

    torch.manual_seed(42)
    model = DummyDenoiser(vocab_size)

    trajectory = diffusion_generate(model, vocab_size, seq_len=4, num_steps=5)

    print("Diffusion generation trajectory (noise -> refined):")
    for step, seq in enumerate(trajectory):
        words = " ".join(id2word[i] for i in seq)
        label = "noise" if step == 0 else f"step {step}"
        print(f"  [{label}]  {words}")
