"""Tests for Lab 01 solution: Discrete diffusion model on TinyStories.

These tests verify the key components work correctly without running
the full training pipeline (which would take too long for CI).
"""

import sys

sys.path.insert(0, "../../../..")
sys.path.insert(0, "../../lesson01-forward-corruption-process")
sys.path.insert(0, "../../lesson03-d3pm-from-scratch")
sys.path.insert(0, "../../lesson04-mdlm")
sys.path.insert(0, "../../lesson05-training-and-sampling")

import pytest
import torch

from shared.datasets.text import SimpleTokenizer, TextDataset
from shared.utils.seed import set_seed


@pytest.fixture(autouse=True)
def seed():
    set_seed(42)


@pytest.fixture
def tokenizer():
    texts = ["the cat sat on the mat", "a dog ran in the park"] * 10
    return SimpleTokenizer(texts, level="char")


@pytest.fixture
def dataset(tokenizer):
    texts = ["the cat sat on the mat", "a dog ran in the park"] * 10
    return TextDataset(texts, tokenizer, seq_len=32)


class TestForwardProcess:
    """Test the forward corruption process."""

    def test_absorbing_corruption(self, tokenizer):
        from src.forward_process import DiscreteForwardProcess

        fp = DiscreteForwardProcess(
            vocab_size=tokenizer.vocab_size,
            num_timesteps=100,
            schedule="absorbing",
            mask_token_id=tokenizer.mask_id,
        )
        x_0 = torch.tensor([[3, 4, 5, 6, 7, 8, 9, 10]])
        t = torch.tensor([100])
        x_t = fp.sample_q_t(x_0, t)

        # At t=T, most tokens should be masked
        mask_count = (x_t == tokenizer.mask_id).sum().item()
        assert mask_count > 0, "No tokens were masked at t=T"

    def test_uniform_corruption(self, tokenizer):
        from src.forward_process import DiscreteForwardProcess

        fp = DiscreteForwardProcess(
            vocab_size=tokenizer.vocab_size,
            num_timesteps=100,
            schedule="uniform",
        )
        x_0 = torch.tensor([[3, 4, 5, 6, 7, 8, 9, 10]])
        t = torch.tensor([50])
        x_t = fp.sample_q_t(x_0, t)

        assert x_t.shape == x_0.shape

    def test_transition_matrix_rows_sum_to_one(self):
        from src.forward_process import DiscreteForwardProcess

        for schedule in ["uniform", "absorbing"]:
            fp = DiscreteForwardProcess(
                vocab_size=10, num_timesteps=100, schedule=schedule
            )
            Qt = fp.get_qt_matrix(50)
            row_sums = Qt.sum(dim=1)
            assert torch.allclose(row_sums, torch.ones(10), atol=1e-5), (
                f"Rows don't sum to 1 for {schedule} schedule"
            )


class TestD3PM:
    """Test the D3PM model."""

    def test_denoiser_forward(self, tokenizer):
        from src.d3pm_model import D3PMDenoiser

        denoiser = D3PMDenoiser(
            vocab_size=tokenizer.vocab_size,
            d_model=32,
            n_heads=2,
            n_layers=1,
            max_seq_len=32,
        )
        x_t = torch.randint(0, tokenizer.vocab_size, (2, 16))
        t = torch.tensor([10, 50])
        logits = denoiser(x_t, t)

        assert logits.shape == (2, 16, tokenizer.vocab_size)

    def test_d3pm_train_loss(self, tokenizer):
        from src.d3pm_model import D3PM, D3PMDenoiser

        denoiser = D3PMDenoiser(
            vocab_size=tokenizer.vocab_size,
            d_model=32,
            n_heads=2,
            n_layers=1,
            max_seq_len=32,
        )
        d3pm = D3PM(
            denoiser=denoiser,
            vocab_size=tokenizer.vocab_size,
            num_timesteps=10,
            schedule="absorbing",
            mask_token_id=tokenizer.mask_id,
        )
        x_0 = torch.randint(3, tokenizer.vocab_size, (4, 16))
        loss = d3pm.train_loss(x_0)

        assert loss.dim() == 0, "Loss should be scalar"
        assert loss.item() > 0, "Loss should be positive"
        assert not torch.isnan(loss), "Loss should not be NaN"

    def test_d3pm_sample(self, tokenizer):
        from src.d3pm_model import D3PM, D3PMDenoiser

        denoiser = D3PMDenoiser(
            vocab_size=tokenizer.vocab_size,
            d_model=32,
            n_heads=2,
            n_layers=1,
            max_seq_len=32,
        )
        d3pm = D3PM(
            denoiser=denoiser,
            vocab_size=tokenizer.vocab_size,
            num_timesteps=5,
            schedule="absorbing",
            mask_token_id=tokenizer.mask_id,
        )
        samples = d3pm.sample(batch_size=2, seq_len=16)

        assert samples.shape == (2, 16)
        assert (samples >= 0).all() and (samples < tokenizer.vocab_size).all()


class TestMDLM:
    """Test the MDLM model."""

    def test_mdlm_denoiser_forward(self, tokenizer):
        from src.mdlm import MDLMDenoiser

        denoiser = MDLMDenoiser(
            vocab_size=tokenizer.vocab_size,
            d_model=32,
            n_heads=2,
            n_layers=1,
            max_seq_len=32,
        )
        x_t = torch.randint(0, tokenizer.vocab_size, (2, 16))
        t = torch.tensor([0.3, 0.7])
        logits = denoiser(x_t, t)

        assert logits.shape == (2, 16, tokenizer.vocab_size)

    def test_mdlm_masking(self, tokenizer):
        from src.mdlm import MDLM, MDLMDenoiser

        denoiser = MDLMDenoiser(
            vocab_size=tokenizer.vocab_size,
            d_model=32,
            n_heads=2,
            n_layers=1,
            max_seq_len=32,
        )
        mdlm = MDLM(
            denoiser=denoiser,
            vocab_size=tokenizer.vocab_size,
            mask_token_id=tokenizer.mask_id,
        )
        x_0 = torch.randint(3, tokenizer.vocab_size, (4, 16))

        # At t=0, very few tokens should be masked
        x_t = mdlm.mask_tokens(x_0, torch.tensor([0.01] * 4))
        mask_rate_low = (x_t == tokenizer.mask_id).float().mean().item()

        # At t=1, most tokens should be masked
        x_t = mdlm.mask_tokens(x_0, torch.tensor([0.99] * 4))
        mask_rate_high = (x_t == tokenizer.mask_id).float().mean().item()

        assert mask_rate_high > mask_rate_low

    def test_mdlm_train_loss(self, tokenizer):
        from src.mdlm import MDLM, MDLMDenoiser

        denoiser = MDLMDenoiser(
            vocab_size=tokenizer.vocab_size,
            d_model=32,
            n_heads=2,
            n_layers=1,
            max_seq_len=32,
        )
        mdlm = MDLM(
            denoiser=denoiser,
            vocab_size=tokenizer.vocab_size,
            mask_token_id=tokenizer.mask_id,
        )
        x_0 = torch.randint(3, tokenizer.vocab_size, (4, 16))
        loss = mdlm.train_loss(x_0)

        assert loss.dim() == 0
        assert loss.item() > 0
        assert not torch.isnan(loss)

    def test_mdlm_sample(self, tokenizer):
        from src.mdlm import MDLM, MDLMDenoiser

        denoiser = MDLMDenoiser(
            vocab_size=tokenizer.vocab_size,
            d_model=32,
            n_heads=2,
            n_layers=1,
            max_seq_len=32,
        )
        mdlm = MDLM(
            denoiser=denoiser,
            vocab_size=tokenizer.vocab_size,
            mask_token_id=tokenizer.mask_id,
            num_timesteps=5,
        )
        samples = mdlm.sample(batch_size=2, seq_len=16)

        assert samples.shape == (2, 16)
        # No mask tokens should remain in final samples
        assert (samples != tokenizer.mask_id).all(), "Mask tokens remain in samples"


class TestTrainingUtils:
    """Test training utility functions."""

    def test_importance_weights(self):
        from src.training_utils import importance_weight_timesteps

        t = torch.arange(1, 101)
        for strategy in ["uniform", "snr", "truncated"]:
            weights = importance_weight_timesteps(t, 100, strategy)
            assert weights.shape == (100,)
            assert (weights >= 0).all()

    def test_cosine_schedule(self):
        from src.training_utils import get_cosine_schedule_with_warmup

        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = get_cosine_schedule_with_warmup(optimizer, 10, 100)

        # LR should increase during warmup
        lrs = []
        for _ in range(100):
            lrs.append(optimizer.param_groups[0]["lr"])
            optimizer.step()
            scheduler.step()

        assert lrs[10] > lrs[0], "LR should increase during warmup"
        assert lrs[-1] < lrs[10], "LR should decrease after warmup"

    def test_sample_with_temperature(self):
        from src.training_utils import sample_with_temperature

        logits = torch.randn(4, 16, 50)
        tokens = sample_with_temperature(logits, temperature=0.8, top_k=10)
        assert tokens.shape == (4, 16)
        assert (tokens >= 0).all() and (tokens < 50).all()

    def test_perplexity_proxy(self, tokenizer):
        from src.mdlm import MDLMDenoiser
        from src.training_utils import compute_perplexity_proxy

        model = MDLMDenoiser(
            vocab_size=tokenizer.vocab_size,
            d_model=32,
            n_heads=2,
            n_layers=1,
            max_seq_len=32,
        )
        x_0 = torch.randint(3, tokenizer.vocab_size, (4, 16))
        ppl = compute_perplexity_proxy(model, x_0, tokenizer.mask_id)

        assert ppl > 0, "Perplexity should be positive"
        assert not torch.isnan(torch.tensor(ppl)), "Perplexity should not be NaN"
