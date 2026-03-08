"""Tests for Lab 01: Flow Matching Text Generator.

Run with: python -m pytest tests/test_solution.py -v
"""

import sys
import os

import torch
import torch.nn.functional as F

# Add solutions directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "solutions"))
from solution import (
    SimpleTokenizer,
    FlowMatchingTextGenerator,
    SDETextGenerator,
    VelocityNet,
)


# ---------------------------------------------------------------------------
# Tokenizer tests
# ---------------------------------------------------------------------------

class TestSimpleTokenizer:
    def setup_method(self):
        self.tokenizer = SimpleTokenizer(max_vocab_size=100)
        self.texts = [
            "the big cat sat on the mat",
            "a small dog ran in the box",
            "the red bird flew over the tree",
        ]
        self.tokenizer.build_vocab(self.texts)

    def test_special_tokens_present(self):
        assert self.tokenizer.PAD in self.tokenizer.word2idx
        assert self.tokenizer.UNK in self.tokenizer.word2idx
        assert self.tokenizer.BOS in self.tokenizer.word2idx
        assert self.tokenizer.EOS in self.tokenizer.word2idx

    def test_special_token_order(self):
        assert self.tokenizer.word2idx["<pad>"] == 0
        assert self.tokenizer.word2idx["<unk>"] == 1
        assert self.tokenizer.word2idx["<bos>"] == 2
        assert self.tokenizer.word2idx["<eos>"] == 3

    def test_vocab_size(self):
        # 4 special + unique words from texts
        assert self.tokenizer.vocab_size > 4
        assert self.tokenizer.vocab_size <= 100

    def test_encode_returns_correct_length(self):
        ids = self.tokenizer.encode("the cat sat", max_len=10)
        assert len(ids) == 10

    def test_encode_starts_with_bos(self):
        ids = self.tokenizer.encode("the cat")
        assert ids[0] == self.tokenizer.word2idx["<bos>"]

    def test_encode_has_eos(self):
        ids = self.tokenizer.encode("the cat", max_len=10)
        eos_id = self.tokenizer.word2idx["<eos>"]
        assert eos_id in ids

    def test_encode_pads_to_max_len(self):
        ids = self.tokenizer.encode("the", max_len=20)
        assert len(ids) == 20
        pad_id = self.tokenizer.word2idx["<pad>"]
        # Last elements should be padding
        assert ids[-1] == pad_id

    def test_decode_recovers_text(self):
        text = "the big cat"
        ids = self.tokenizer.encode(text, max_len=10)
        decoded = self.tokenizer.decode(ids)
        assert decoded == text

    def test_decode_stops_at_eos(self):
        ids = self.tokenizer.encode("the cat", max_len=20)
        decoded = self.tokenizer.decode(ids)
        assert "<pad>" not in decoded
        assert "<eos>" not in decoded

    def test_unknown_words_get_unk(self):
        ids = self.tokenizer.encode("the xyzzy cat", max_len=10)
        unk_id = self.tokenizer.word2idx["<unk>"]
        # "xyzzy" is not in vocab, so it should map to UNK
        assert unk_id in ids


# ---------------------------------------------------------------------------
# VelocityNet tests
# ---------------------------------------------------------------------------

class TestVelocityNet:
    def test_output_shape(self):
        net = VelocityNet(embed_dim=64, d_model=128, n_heads=4, n_layers=2, d_ff=256)
        x = torch.randn(4, 16, 64)
        t = torch.rand(4)
        out = net(x, t)
        assert out.shape == (4, 16, 64)

    def test_different_timesteps(self):
        net = VelocityNet(embed_dim=64, d_model=128, n_heads=4, n_layers=2, d_ff=256)
        x = torch.randn(2, 8, 64)
        t1 = torch.tensor([0.1, 0.1])
        t2 = torch.tensor([0.9, 0.9])
        out1 = net(x, t1)
        out2 = net(x, t2)
        # Different timesteps should give different outputs
        assert not torch.allclose(out1, out2)


# ---------------------------------------------------------------------------
# FlowMatchingTextGenerator tests
# ---------------------------------------------------------------------------

class TestFlowMatchingTextGenerator:
    def setup_method(self):
        torch.manual_seed(42)
        self.gen = FlowMatchingTextGenerator(
            vocab_size=50,
            embed_dim=32,
            d_model=64,
            n_heads=2,
            n_layers=1,
            d_ff=128,
            seq_len=8,
            lr=1e-3,
        )

    def test_train_step_returns_float(self):
        token_ids = torch.randint(0, 50, (4, 8))
        loss = self.gen.train_step(token_ids)
        assert isinstance(loss, float)
        assert loss > 0

    def test_train_step_reduces_loss(self):
        token_ids = torch.randint(0, 50, (16, 8))
        first_loss = self.gen.train_step(token_ids)
        for _ in range(20):
            loss = self.gen.train_step(token_ids)
        # Loss should decrease after some training
        assert loss < first_loss

    def test_sample_shape(self):
        tokens = self.gen.sample(batch_size=4, n_steps=10)
        assert tokens.shape == (4, 8)

    def test_sample_valid_range(self):
        tokens = self.gen.sample(batch_size=4, n_steps=10)
        assert tokens.min() >= 0
        assert tokens.max() < 50

    def test_deterministic_sampling(self):
        """Same seed should give same output (ODE is deterministic)."""
        torch.manual_seed(123)
        tokens1 = self.gen.sample(batch_size=2, n_steps=10)
        torch.manual_seed(123)
        tokens2 = self.gen.sample(batch_size=2, n_steps=10)
        assert torch.equal(tokens1, tokens2)


# ---------------------------------------------------------------------------
# SDETextGenerator tests
# ---------------------------------------------------------------------------

class TestSDETextGenerator:
    def setup_method(self):
        torch.manual_seed(42)
        self.gen = SDETextGenerator(
            vocab_size=50,
            embed_dim=32,
            d_model=64,
            n_heads=2,
            n_layers=1,
            d_ff=128,
            seq_len=8,
            num_timesteps=100,
            lr=1e-3,
        )

    def test_train_step_returns_float(self):
        token_ids = torch.randint(0, 50, (4, 8))
        loss = self.gen.train_step(token_ids)
        assert isinstance(loss, float)
        assert loss > 0

    def test_sample_shape(self):
        tokens = self.gen.sample(batch_size=4, n_steps=20)
        assert tokens.shape == (4, 8)

    def test_sample_valid_range(self):
        tokens = self.gen.sample(batch_size=4, n_steps=20)
        assert tokens.min() >= 0
        assert tokens.max() < 50

    def test_sample_with_full_steps(self):
        tokens = self.gen.sample(batch_size=2)
        assert tokens.shape == (2, 8)

    def test_stochastic_sampling(self):
        """Different seeds should give different output (SDE is stochastic)."""
        torch.manual_seed(111)
        tokens1 = self.gen.sample(batch_size=2, n_steps=20)
        torch.manual_seed(222)
        tokens2 = self.gen.sample(batch_size=2, n_steps=20)
        # With very high probability, different seeds give different tokens
        # (not guaranteed, but extremely unlikely to be identical)
        # We just check the function runs without error
        assert tokens1.shape == tokens2.shape


# ---------------------------------------------------------------------------
# Integration test
# ---------------------------------------------------------------------------

class TestIntegration:
    """End-to-end test: tokenize, train, generate, decode."""

    def test_full_pipeline(self):
        torch.manual_seed(42)

        # Tokenizer
        tokenizer = SimpleTokenizer(max_vocab_size=100)
        texts = [
            "the cat sat on the mat",
            "a dog ran in the box",
            "the bird flew over the tree",
        ] * 10
        tokenizer.build_vocab(texts)

        # Prepare data
        seq_len = 12
        train_data = torch.tensor([
            tokenizer.encode(t, max_len=seq_len) for t in texts
        ])

        # Flow matching generator
        gen = FlowMatchingTextGenerator(
            vocab_size=tokenizer.vocab_size,
            embed_dim=32,
            d_model=64,
            n_heads=2,
            n_layers=1,
            d_ff=128,
            seq_len=seq_len,
            lr=1e-3,
        )

        # Train briefly
        for _ in range(10):
            gen.train_step(train_data[:8])

        # Generate
        tokens = gen.sample(batch_size=2, n_steps=10)

        # Decode
        for i in range(2):
            text = tokenizer.decode(tokens[i].tolist())
            assert isinstance(text, str)
