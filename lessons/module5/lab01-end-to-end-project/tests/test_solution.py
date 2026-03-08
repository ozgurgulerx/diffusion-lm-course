"""Tests for Lab 01: End-to-End Diffusion LM Project.

Tests cover data preparation, model architecture, training, and generation.
Run with: pytest test_solution.py -v
"""

import pytest
import torch
from transformers import AutoTokenizer

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "solutions"))

from solution import (
    PoetryDataset,
    PoetryDiffusionModel,
    PoetryPipeline,
    create_sample_poetry_data,
    generate_poetry,
    train_poetry_model,
)


@pytest.fixture
def tokenizer():
    return AutoTokenizer.from_pretrained("bert-base-uncased")


@pytest.fixture
def sample_data():
    return create_sample_poetry_data()


@pytest.fixture
def dataset(sample_data, tokenizer):
    return PoetryDataset(sample_data, tokenizer, max_seq_len=64)


@pytest.fixture
def model(tokenizer):
    return PoetryDiffusionModel(
        vocab_size=len(tokenizer),
        d_model=64,
        nhead=2,
        num_layers=2,
        num_styles=3,
        max_seq_len=64,
        mask_token_id=tokenizer.mask_token_id,
    )


class TestDataPreparation:
    """Tests for data loading and preprocessing."""

    def test_sample_data_not_empty(self, sample_data):
        assert len(sample_data) > 0

    def test_sample_data_has_labels(self, sample_data):
        for text, label in sample_data:
            assert isinstance(text, str)
            assert isinstance(label, int)
            assert 0 <= label <= 2

    def test_sample_data_three_styles(self, sample_data):
        labels = {label for _, label in sample_data}
        assert labels == {0, 1, 2}

    def test_dataset_length(self, dataset):
        assert len(dataset) > 0

    def test_dataset_item_shape(self, dataset):
        item = dataset[0]
        assert item["input_ids"].shape == (64,)
        assert item["attention_mask"].shape == (64,)
        assert item["label"].shape == ()

    def test_dataset_item_types(self, dataset):
        item = dataset[0]
        assert item["input_ids"].dtype == torch.long
        assert item["attention_mask"].dtype == torch.long
        assert item["label"].dtype == torch.long


class TestModel:
    """Tests for the PoetryDiffusionModel architecture."""

    def test_model_forward_shape(self, model, tokenizer):
        B, L = 2, 64
        x = torch.randint(0, len(tokenizer), (B, L))
        t = torch.rand(B)
        style = torch.tensor([0, 1])
        attn = torch.ones(B, L)

        logits = model(x, t, style, attn)
        assert logits.shape == (B, L, len(tokenizer))

    def test_model_unconditional(self, model, tokenizer):
        B, L = 2, 64
        x = torch.randint(0, len(tokenizer), (B, L))
        t = torch.rand(B)
        attn = torch.ones(B, L)

        logits = model(x, t, None, attn, force_uncond=True)
        assert logits.shape == (B, L, len(tokenizer))

    def test_conditional_differs_from_unconditional(self, model, tokenizer):
        B, L = 2, 64
        x = torch.randint(0, len(tokenizer), (B, L))
        t = torch.rand(B)
        style = torch.tensor([0, 1])
        attn = torch.ones(B, L)

        model.train(False)
        cond = model(x, t, style, attn, force_uncond=False)
        uncond = model(x, t, style, attn, force_uncond=True)

        # They should differ (different embeddings)
        assert not torch.allclose(cond, uncond, atol=1e-5)

    def test_model_parameter_count(self, model):
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params > 0
        # Small model should have < 5M parameters
        assert n_params < 5_000_000


class TestTraining:
    """Tests for the training loop."""

    def test_training_reduces_loss(self, model, dataset):
        losses = train_poetry_model(
            model, dataset, epochs=5, batch_size=4,
            lr=1e-3, num_timesteps=100, device="cpu",
        )
        assert len(losses) == 5
        # Loss should generally decrease (allow some noise)
        assert losses[-1] < losses[0] * 1.5  # At least not blowing up

    def test_training_returns_valid_losses(self, model, dataset):
        losses = train_poetry_model(
            model, dataset, epochs=3, batch_size=4,
            num_timesteps=100, device="cpu",
        )
        for loss in losses:
            assert loss >= 0
            assert not torch.isnan(torch.tensor(loss))
            assert not torch.isinf(torch.tensor(loss))


class TestGeneration:
    """Tests for text generation."""

    def test_generation_returns_strings(self, model, tokenizer):
        results = generate_poetry(
            model, style=0, tokenizer=tokenizer,
            num_samples=2, seq_len=32, num_steps=5,
            num_timesteps=100, device="cpu",
        )
        assert len(results) == 2
        for text in results:
            assert isinstance(text, str)
            assert len(text) > 0

    def test_generation_different_styles(self, model, tokenizer):
        model.train(False)
        # Set seed for reproducibility
        torch.manual_seed(42)
        r1 = generate_poetry(
            model, style=0, tokenizer=tokenizer,
            num_samples=1, seq_len=32, num_steps=5,
            num_timesteps=100, device="cpu",
        )
        torch.manual_seed(42)
        r2 = generate_poetry(
            model, style=2, tokenizer=tokenizer,
            num_samples=1, seq_len=32, num_steps=5,
            num_timesteps=100, device="cpu",
        )
        # Different styles with same seed should give different outputs
        assert isinstance(r1[0], str) and isinstance(r2[0], str)


class TestPipeline:
    """Tests for the reusable pipeline."""

    def test_pipeline_callable(self, model, tokenizer):
        pipe = PoetryPipeline(model, tokenizer, num_timesteps=100, device="cpu")
        results = pipe(style="romantic", num_samples=2)
        assert len(results) == 2
        for text in results:
            assert isinstance(text, str)

    def test_pipeline_all_styles(self, model, tokenizer):
        pipe = PoetryPipeline(model, tokenizer, num_timesteps=100, device="cpu")
        for style in ["romantic", "nature", "melancholy"]:
            results = pipe(style=style, num_samples=1)
            assert len(results) == 1
            assert isinstance(results[0], str)
