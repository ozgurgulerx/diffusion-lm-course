"""Reverse (denoising) process for discrete diffusion models.

Implements the posterior q(x_{t-1} | x_t, x_0) and reverse sampling step
from D3PM (Austin et al., 2021) Section 3.2.

Key insight: if we can predict x_0 from x_t (using a neural network), we can
compute the exact posterior for the reverse step.
"""

import torch
import torch.nn.functional as F


def compute_posterior(
    x_t: torch.Tensor,
    x_0_probs: torch.Tensor,
    t: int,
    forward_process,
) -> torch.Tensor:
    """Compute the posterior q(x_{t-1} | x_t, x_0) for discrete diffusion.

    Using Bayes' rule:
        q(x_{t-1} = j | x_t, x_0) ∝ q(x_t | x_{t-1} = j) * q(x_{t-1} = j | x_0)

    Where:
        q(x_t | x_{t-1}) is given by Q_t (single-step transition)
        q(x_{t-1} | x_0) is given by Q_bar_{t-1} (cumulative transition)

    In practice, x_0 is unknown during generation, so we use a neural network's
    prediction p(x_0 | x_t) and marginalize:
        q(x_{t-1} | x_t) ≈ sum_{x_0} q(x_{t-1} | x_t, x_0) * p(x_0 | x_t)

    Args:
        x_t: Current noisy tokens, shape (batch_size, seq_len).
        x_0_probs: Predicted distribution over x_0, shape (batch_size, seq_len, vocab_size).
            This is the neural network's prediction of what the clean token is.
        t: Current timestep (1-indexed). Must be >= 2 for a meaningful reverse step.
        forward_process: DiscreteForwardProcess instance.

    Returns:
        Posterior distribution over x_{t-1}, shape (batch_size, seq_len, vocab_size).
    """
    vocab_size = forward_process.vocab_size
    batch_size, seq_len = x_t.shape

    # Get transition matrices
    Qt = forward_process.get_qt_matrix(t)            # (K, K): Q_t[i,j] = p(x_t=j | x_{t-1}=i)
    Qt_bar_prev = forward_process.get_qt_bar(t - 1)  # (K, K): Q_bar_{t-1}[i,j] = p(x_{t-1}=j | x_0=i)

    # For each position, compute posterior over x_{t-1}
    # We marginalize over possible x_0 values using the predicted distribution

    # Step 1: For each possible x_0 value, compute q(x_{t-1} | x_t, x_0)
    # q(x_{t-1}=j | x_t=k, x_0=i) ∝ Q_t[j, k] * Q_bar_{t-1}[i, j]

    # x_t one-hot: (batch, seq_len, K)
    x_t_onehot = F.one_hot(x_t.long(), vocab_size).float()

    # Likelihood term: Q_t[j, k] for observed x_t = k
    # For each x_{t-1} = j, probability of transitioning to observed x_t
    # Shape: (batch, seq_len, K) — one value per possible x_{t-1} value j
    likelihood = x_t_onehot @ Qt.T  # (batch, seq, K) @ (K, K) -> (batch, seq, K)
    # Actually: likelihood[b, s, j] = Qt[j, x_t[b,s]] = P(x_t[b,s] | x_{t-1}=j)
    # Correction: x_t_onehot is (batch, seq, K), Qt is (K, K)
    # x_t_onehot @ Qt.T gives (batch, seq, K) where [b,s,j] = sum_k onehot[b,s,k] * Qt[k,j]
    # We want Qt[j, x_t] = P(x_t | x_{t-1}=j), so we need Qt[:, x_t]
    # Let's recompute correctly

    # likelihood[b, s, j] = Qt[j, x_t[b,s]] for each possible x_{t-1} = j
    likelihood = Qt[:, x_t.long()]  # (K, batch, seq) — index columns by observed x_t
    likelihood = likelihood.permute(1, 2, 0)  # (batch, seq, K)

    # Prior term: for each possible x_0 = i, get q(x_{t-1} = j | x_0 = i)
    # prior[i, j] = Qt_bar_prev[i, j]
    # Weighted by x_0 prediction: sum_i x_0_probs[b,s,i] * Qt_bar_prev[i, j]
    # Shape: (batch, seq, K)
    prior = x_0_probs @ Qt_bar_prev  # (batch, seq, K) @ (K, K) -> (batch, seq, K)
    # prior[b, s, j] = sum_i p(x_0=i | x_t) * p(x_{t-1}=j | x_0=i)

    # Posterior: q(x_{t-1} = j | x_t, x_0) ∝ likelihood[j] * prior[j]
    posterior = likelihood * prior  # (batch, seq, K)

    # Normalize
    posterior = posterior / (posterior.sum(dim=-1, keepdim=True) + 1e-10)

    return posterior


def sample_reverse_step(
    x_t: torch.Tensor,
    x_0_probs: torch.Tensor,
    t: int,
    forward_process,
) -> torch.Tensor:
    """Sample x_{t-1} from the reverse posterior q(x_{t-1} | x_t, x_0_pred).

    Args:
        x_t: Current noisy tokens, shape (batch_size, seq_len).
        x_0_probs: Predicted distribution over x_0 from the denoising model,
            shape (batch_size, seq_len, vocab_size).
        t: Current timestep (1-indexed). Must be >= 2.
        forward_process: DiscreteForwardProcess instance.

    Returns:
        Sampled tokens x_{t-1}, shape (batch_size, seq_len).
    """
    if t <= 1:
        # At t=1, just return the argmax of x_0 prediction
        return x_0_probs.argmax(dim=-1)

    # Compute posterior distribution
    posterior = compute_posterior(x_t, x_0_probs, t, forward_process)

    # Sample from the categorical distribution
    batch_size, seq_len, vocab_size = posterior.shape
    # Reshape for sampling: (batch * seq, vocab)
    posterior_flat = posterior.reshape(-1, vocab_size)
    x_t_minus_1 = torch.multinomial(posterior_flat, num_samples=1).squeeze(-1)
    x_t_minus_1 = x_t_minus_1.reshape(batch_size, seq_len)

    return x_t_minus_1


def sample_reverse_step_with_temperature(
    x_t: torch.Tensor,
    x_0_probs: torch.Tensor,
    t: int,
    forward_process,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Sample reverse step with temperature scaling on the x_0 prediction.

    Lower temperature -> more deterministic (greedy).
    Higher temperature -> more random (diverse).

    Args:
        x_t: Current noisy tokens, shape (batch_size, seq_len).
        x_0_probs: Predicted logits/probs over x_0, shape (batch_size, seq_len, vocab_size).
        t: Current timestep.
        forward_process: DiscreteForwardProcess instance.
        temperature: Sampling temperature (default 1.0).

    Returns:
        Sampled tokens x_{t-1}, shape (batch_size, seq_len).
    """
    if temperature != 1.0:
        # Apply temperature to the log-probabilities
        log_probs = torch.log(x_0_probs + 1e-10) / temperature
        x_0_probs = F.softmax(log_probs, dim=-1)

    return sample_reverse_step(x_t, x_0_probs, t, forward_process)


def demo_reverse_with_oracle(
    x_0: torch.Tensor,
    forward_process,
    t_start: int = 10,
) -> torch.Tensor:
    """Demonstrate the reverse process using an oracle that knows x_0.

    This is a sanity check: if we have perfect knowledge of x_0, the reverse
    process should recover the original sequence. In practice, a neural network
    approximates p(x_0 | x_t).

    Args:
        x_0: Clean token IDs, shape (batch_size, seq_len).
        forward_process: DiscreteForwardProcess instance.
        t_start: Starting timestep for the reverse process.

    Returns:
        Recovered tokens after running the full reverse chain.
    """
    vocab_size = forward_process.vocab_size

    # Step 1: Corrupt x_0 to x_t_start
    t = torch.tensor([t_start] * x_0.shape[0])
    x_t = forward_process.sample_q_t(x_0, t)

    print(f"Starting reverse process from t={t_start}")
    print(f"x_0:          {x_0[0].tolist()}")
    print(f"x_{t_start} (corrupted): {x_t[0].tolist()}")

    # Step 2: Run reverse process using oracle (perfect x_0 knowledge)
    x_current = x_t
    for t_val in range(t_start, 0, -1):
        # Oracle: we know x_0, so we give the true one-hot distribution
        x_0_probs = F.one_hot(x_0.long(), vocab_size).float()
        x_current = sample_reverse_step(x_current, x_0_probs, t_val, forward_process)

        if t_val % max(1, t_start // 5) == 0 or t_val <= 3:
            print(f"x_{t_val - 1}:         {x_current[0].tolist()}")

    return x_current
