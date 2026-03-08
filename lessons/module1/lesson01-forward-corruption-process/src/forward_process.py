"""Forward corruption process for discrete diffusion models.

Implements the forward (noising) Markov chain from D3PM (Austin et al., 2021).
Two corruption schedules are provided:
  - Uniform: each token may be replaced by any token with equal probability.
  - Absorbing: each token may be replaced by a special [MASK] token.
"""

import torch
import torch.nn.functional as F


class DiscreteForwardProcess:
    """Discrete forward diffusion process using transition matrices Q_t.

    At each timestep t, the transition matrix Q_t defines the probability of
    transitioning from token i to token j. The cumulative corruption from
    time 0 to time t is given by the product Q_bar_t = Q_1 * Q_2 * ... * Q_t.

    For efficiency, we parameterize Q_t so that Q_bar_t has a closed-form
    expression, avoiding the need to multiply many matrices.

    Args:
        vocab_size: Number of tokens in the vocabulary.
        num_timesteps: Total number of diffusion timesteps T.
        schedule: Corruption schedule type — "uniform" or "absorbing".
        mask_token_id: Token ID used as the absorbing state (only for "absorbing").
    """

    def __init__(
        self,
        vocab_size: int,
        num_timesteps: int = 100,
        schedule: str = "uniform",
        mask_token_id: int = 2,
    ):
        self.vocab_size = vocab_size
        self.num_timesteps = num_timesteps
        self.schedule = schedule
        self.mask_token_id = mask_token_id

        # Precompute the noise schedule: beta_t values in [0, 1]
        # beta_t is the probability of corrupting each token at step t
        self.betas = self._compute_beta_schedule()

        # Precompute cumulative products: alpha_bar_t = prod(1 - beta_s, s=1..t)
        # This is the probability that a token remains unchanged from time 0 to t
        alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(alphas, dim=0)

    def _compute_beta_schedule(self) -> torch.Tensor:
        """Compute a linear noise schedule.

        Returns:
            Tensor of shape (num_timesteps,) with beta values increasing
            linearly from near 0 to near 1.
        """
        # Linear schedule from small value to ~1
        return torch.linspace(1e-4, 0.02, self.num_timesteps)

    def get_qt_matrix(self, t: int) -> torch.Tensor:
        """Get the single-step transition matrix Q_t.

        Q_t[i, j] = P(x_t = j | x_{t-1} = i).

        Args:
            t: Timestep (1-indexed, in [1, num_timesteps]).

        Returns:
            Transition matrix of shape (vocab_size, vocab_size).
        """
        beta_t = self.betas[t - 1]

        if self.schedule == "uniform":
            # With probability (1 - beta_t), token stays the same.
            # With probability beta_t, token is replaced uniformly at random.
            Qt = (1 - beta_t) * torch.eye(self.vocab_size) + (
                beta_t / self.vocab_size
            ) * torch.ones(self.vocab_size, self.vocab_size)

        elif self.schedule == "absorbing":
            # With probability (1 - beta_t), token stays the same.
            # With probability beta_t, token transitions to mask_token_id.
            Qt = (1 - beta_t) * torch.eye(self.vocab_size)
            Qt[:, self.mask_token_id] += beta_t

        else:
            raise ValueError(f"Unknown schedule: {self.schedule}")

        return Qt

    def get_qt_bar(self, t: int) -> torch.Tensor:
        """Get the cumulative transition matrix Q_bar_t = Q_1 * Q_2 * ... * Q_t.

        Instead of multiplying t matrices, we use the closed-form expression
        based on alpha_bar_t.

        Args:
            t: Timestep (1-indexed, in [1, num_timesteps]).

        Returns:
            Cumulative transition matrix of shape (vocab_size, vocab_size).
        """
        alpha_bar_t = self.alpha_bars[t - 1]

        if self.schedule == "uniform":
            # Q_bar_t[i, j] = alpha_bar_t * I[i=j] + (1 - alpha_bar_t) / K
            Qt_bar = alpha_bar_t * torch.eye(self.vocab_size) + (
                (1 - alpha_bar_t) / self.vocab_size
            ) * torch.ones(self.vocab_size, self.vocab_size)

        elif self.schedule == "absorbing":
            # Q_bar_t[i, j] = alpha_bar_t * I[i=j] + (1 - alpha_bar_t) * I[j=mask]
            Qt_bar = alpha_bar_t * torch.eye(self.vocab_size)
            Qt_bar[:, self.mask_token_id] += 1 - alpha_bar_t

        else:
            raise ValueError(f"Unknown schedule: {self.schedule}")

        return Qt_bar

    def sample_q_t(
        self, x_0: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Sample x_t ~ q(x_t | x_0) using the cumulative transition matrix.

        This is the key function for training: given clean data x_0 and a
        timestep t, produce a corrupted version x_t.

        Args:
            x_0: Clean token IDs, shape (batch_size, seq_len).
            t: Timestep for each sample, shape (batch_size,), values in [1, T].

        Returns:
            Corrupted token IDs x_t, shape (batch_size, seq_len).
        """
        batch_size, seq_len = x_0.shape
        x_t = torch.zeros_like(x_0)

        for i in range(batch_size):
            ti = t[i].item()
            alpha_bar_t = self.alpha_bars[int(ti) - 1]

            if self.schedule == "uniform":
                # Each token: with prob alpha_bar_t stays the same,
                # with prob (1 - alpha_bar_t) replaced by uniform random token
                mask = torch.rand(seq_len) > alpha_bar_t
                random_tokens = torch.randint(0, self.vocab_size, (seq_len,))
                x_t[i] = torch.where(mask, random_tokens, x_0[i])

            elif self.schedule == "absorbing":
                # Each token: with prob alpha_bar_t stays the same,
                # with prob (1 - alpha_bar_t) replaced by mask token
                mask = torch.rand(seq_len) > alpha_bar_t
                x_t[i] = torch.where(
                    mask,
                    torch.full((seq_len,), self.mask_token_id, dtype=x_0.dtype),
                    x_0[i],
                )

        return x_t

    def sample_q_t_batched(
        self, x_0: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Vectorized version of sample_q_t (no Python loop over batch).

        Args:
            x_0: Clean token IDs, shape (batch_size, seq_len).
            t: Timestep per sample, shape (batch_size,), values in [1, T].

        Returns:
            Corrupted token IDs x_t, shape (batch_size, seq_len).
        """
        batch_size, seq_len = x_0.shape
        # alpha_bar_t for each sample: shape (batch_size, 1)
        alpha_bar_t = self.alpha_bars[(t - 1).long()].unsqueeze(1)

        # Decide which tokens get corrupted
        corrupt_mask = torch.rand(batch_size, seq_len) > alpha_bar_t

        if self.schedule == "uniform":
            random_tokens = torch.randint(0, self.vocab_size, (batch_size, seq_len))
            x_t = torch.where(corrupt_mask, random_tokens, x_0)

        elif self.schedule == "absorbing":
            mask_tokens = torch.full_like(x_0, self.mask_token_id)
            x_t = torch.where(corrupt_mask, mask_tokens, x_0)

        else:
            raise ValueError(f"Unknown schedule: {self.schedule}")

        return x_t

    def get_corruption_rate(self, t: int) -> float:
        """Return the expected fraction of corrupted tokens at timestep t.

        Args:
            t: Timestep (1-indexed).

        Returns:
            Expected corruption rate (1 - alpha_bar_t).
        """
        return 1.0 - self.alpha_bars[t - 1].item()


def visualize_corruption(
    x_0: torch.Tensor,
    forward_process: DiscreteForwardProcess,
    timesteps: list[int],
    tokenizer=None,
) -> list[str]:
    """Visualize corruption at multiple timesteps.

    Args:
        x_0: A single clean sequence, shape (seq_len,).
        forward_process: DiscreteForwardProcess instance.
        timesteps: List of timesteps to visualize.
        tokenizer: Optional tokenizer with a decode method.

    Returns:
        List of decoded strings at each timestep.
    """
    results = []
    x_0_batch = x_0.unsqueeze(0)  # (1, seq_len)

    for t_val in timesteps:
        t = torch.tensor([t_val])
        x_t = forward_process.sample_q_t(x_0_batch, t)
        if tokenizer is not None:
            text = tokenizer.decode(x_t[0].tolist())
        else:
            text = str(x_t[0].tolist())
        results.append(text)
        corruption_rate = forward_process.get_corruption_rate(t_val)
        print(f"t={t_val:4d} (corruption={corruption_rate:.2%}): {text[:80]}")

    return results
