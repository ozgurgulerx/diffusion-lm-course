"""Tests for Lab 01: MDM implementation and comparison."""

import math
import pytest
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# We import from the solutions directory to test the reference implementation.
# Students would import from their own code instead.
# ---------------------------------------------------------------------------
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "solutions"))
from solution import MDM, cosine_masking_schedule, distinct_ngrams, TransformerDenoiser


# ---------------------------------------------------------------------------
# Test masking schedule
# ---------------------------------------------------------------------------

class TestMaskingSchedule:
    def test_gamma_at_zero(self):
        """gamma(0) should be close to 0 (almost no masking)."""
        t = torch.tensor([0.0])
        gamma = cosine_masking_schedule(t)
        assert gamma.item() < 0.01, f"gamma(0) = {gamma.item()}, expected ~0"

    def test_gamma_at_one(self):
        """gamma(1) should be close to 1 (fully masked)."""
        t = torch.tensor([1.0])
        gamma = cosine_masking_schedule(t)
        assert gamma.item() > 0.99, f"gamma(1) = {gamma.item()}, expected ~1"

    def test_gamma_monotonic(self):
        """gamma(t) should be monotonically increasing."""
        t = torch.linspace(0, 1, 100)
        gamma = cosine_masking_schedule(t)
        diffs = gamma[1:] - gamma[:-1]
        assert (diffs >= -1e-6).all(), "gamma(t) is not monotonically increasing"

    def test_gamma_batch(self):
        """Schedule should work on batched inputs."""
        t = torch.rand(16)
        gamma = cosine_masking_schedule(t)
        assert gamma.shape == (16,)
        assert (gamma >= 0).all() and (gamma <= 1).all()


# ---------------------------------------------------------------------------
# Test TransformerDenoiser
# ---------------------------------------------------------------------------

class TestTransformerDenoiser:
    def test_output_shape(self):
        """Denoiser should output (B, L, V) logits."""
        V, B, L = 100, 4, 16
        model = TransformerDenoiser(vocab_size=V, d_model=64, n_heads=2,
                                     n_layers=2, max_seq_len=L)
        x = torch.randint(0, V, (B, L))
        t = torch.rand(B)
        logits = model(x, t)
        assert logits.shape == (B, L, V)

    def test_different_timesteps(self):
        """Different timesteps should produce different outputs."""
        V = 50
        model = TransformerDenoiser(vocab_size=V, d_model=64, n_heads=2,
                                     n_layers=2, max_seq_len=16)
        x = torch.randint(0, V, (1, 16))
        logits_0 = model(x, torch.tensor([0.0]))
        logits_1 = model(x, torch.tensor([1.0]))
        assert not torch.allclose(logits_0, logits_1, atol=1e-3)


# ---------------------------------------------------------------------------
# Test MDM
# ---------------------------------------------------------------------------

class TestMDM:
    @pytest.fixture
    def small_mdm(self):
        return MDM(vocab_size=50, mask_token_id=49, d_model=64, n_heads=2,
                   n_layers=2, max_seq_len=32)

    def test_forward_corrupt_shape(self, small_mdm):
        """forward_corrupt should return x_t and mask of correct shapes."""
        x_0 = torch.randint(0, 49, (4, 16))
        t = torch.rand(4)
        x_t, mask = small_mdm.forward_corrupt(x_0, t)
        assert x_t.shape == x_0.shape
        assert mask.shape == x_0.shape
        assert mask.dtype == torch.bool

    def test_forward_corrupt_masks_correctly(self, small_mdm):
        """Masked positions should have mask_token_id."""
        x_0 = torch.randint(0, 49, (4, 16))
        t = torch.tensor([0.5, 0.5, 0.5, 0.5])
        x_t, mask = small_mdm.forward_corrupt(x_0, t)
        # Where mask is True, x_t should equal mask_token_id
        assert (x_t[mask] == 49).all()
        # Where mask is False, x_t should equal x_0
        assert (x_t[~mask] == x_0[~mask]).all()

    def test_forward_corrupt_more_masking_at_higher_t(self, small_mdm):
        """Higher t should result in more masking on average."""
        torch.manual_seed(0)
        x_0 = torch.randint(0, 49, (100, 32))
        t_low = torch.full((100,), 0.1)
        t_high = torch.full((100,), 0.9)
        _, mask_low = small_mdm.forward_corrupt(x_0, t_low)
        _, mask_high = small_mdm.forward_corrupt(x_0, t_high)
        assert mask_high.float().mean() > mask_low.float().mean()

    def test_train_loss_scalar(self, small_mdm):
        """train_loss should return a scalar."""
        x_0 = torch.randint(0, 49, (4, 16))
        loss = small_mdm.train_loss(x_0)
        assert loss.dim() == 0
        assert loss.item() > 0

    def test_train_loss_backward(self, small_mdm):
        """Loss should be differentiable."""
        x_0 = torch.randint(0, 49, (4, 16))
        loss = small_mdm.train_loss(x_0)
        loss.backward()
        # Check at least one param has gradients
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in small_mdm.parameters())
        assert has_grad

    def test_sample_shape(self, small_mdm):
        """sample should return (B, L) tensor of token IDs."""
        samples = small_mdm.sample(batch_size=2, seq_len=16, num_steps=5)
        assert samples.shape == (2, 16)
        assert samples.dtype == torch.long

    def test_sample_no_mask_tokens(self, small_mdm):
        """Generated samples should not contain [MASK] tokens."""
        samples = small_mdm.sample(batch_size=4, seq_len=16, num_steps=10)
        assert (samples != 49).all(), "Samples still contain [MASK] tokens"

    def test_sample_valid_range(self, small_mdm):
        """All sampled token IDs should be in valid range."""
        samples = small_mdm.sample(batch_size=4, seq_len=16, num_steps=10)
        assert (samples >= 0).all()
        assert (samples < 50).all()


# ---------------------------------------------------------------------------
# Test metrics
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_distinct_ngrams_all_same(self):
        """All-same sequences should have low diversity."""
        seqs = torch.zeros(10, 20, dtype=torch.long)
        div = distinct_ngrams(seqs, n=2)
        # Only 1 distinct bigram out of 10*19 = 190 total
        assert div < 0.1

    def test_distinct_ngrams_all_different(self):
        """All-different sequences should have high diversity."""
        seqs = torch.arange(200).view(10, 20)
        div = distinct_ngrams(seqs, n=2)
        assert div > 0.9

    def test_distinct_ngrams_range(self):
        """Diversity should be in [0, 1]."""
        seqs = torch.randint(0, 50, (10, 20))
        div = distinct_ngrams(seqs, n=2)
        assert 0.0 <= div <= 1.0


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_train_and_sample(self):
        """End-to-end: train for a few steps, then sample."""
        torch.manual_seed(42)
        V, M = 30, 29
        model = MDM(vocab_size=V, mask_token_id=M, d_model=32, n_heads=2,
                     n_layers=1, max_seq_len=16)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        data = torch.randint(0, V - 1, (16, 16))
        for _ in range(5):
            optimizer.zero_grad()
            loss = model.train_loss(data)
            loss.backward()
            optimizer.step()

        samples = model.sample(batch_size=4, seq_len=16, num_steps=5)
        assert samples.shape == (4, 16)
        assert (samples != M).all()
