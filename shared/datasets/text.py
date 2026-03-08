"""Text dataset utilities for the course."""

from collections import Counter
from typing import Optional

import torch
from torch.utils.data import Dataset


class SimpleTokenizer:
    """Character-level or word-level tokenizer for toy experiments.

    This is intentionally simple — no BPE, no sentencepiece. It exists so early
    lessons can focus on diffusion mechanics without tokenizer complexity.
    """

    def __init__(self, texts: list[str], level: str = "char", max_vocab: int = 5000):
        """Build vocabulary from texts.

        Args:
            texts: List of text strings to build vocab from.
            level: "char" for character-level, "word" for word-level.
            max_vocab: Maximum vocabulary size (word-level only).
        """
        self.level = level
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.mask_token = "<mask>"

        special = [self.pad_token, self.unk_token, self.mask_token]

        if level == "char":
            chars = sorted(set("".join(texts)))
            self.token2id = {t: i for i, t in enumerate(special + chars)}
        else:
            words = " ".join(texts).split()
            counts = Counter(words).most_common(max_vocab - len(special))
            vocab = [w for w, _ in counts]
            self.token2id = {t: i for i, t in enumerate(special + vocab)}

        self.id2token = {i: t for t, i in self.token2id.items()}
        self.vocab_size = len(self.token2id)
        self.pad_id = self.token2id[self.pad_token]
        self.unk_id = self.token2id[self.unk_token]
        self.mask_id = self.token2id[self.mask_token]

    def encode(self, text: str) -> list[int]:
        """Convert text to list of token IDs."""
        if self.level == "char":
            return [self.token2id.get(c, self.unk_id) for c in text]
        return [self.token2id.get(w, self.unk_id) for w in text.split()]

    def decode(self, ids: list[int]) -> str:
        """Convert list of token IDs back to text."""
        tokens = [self.id2token.get(i, self.unk_token) for i in ids]
        if self.level == "char":
            return "".join(tokens)
        return " ".join(tokens)


class TextDataset(Dataset):
    """Fixed-length tokenized text dataset for training diffusion models."""

    def __init__(self, texts: list[str], tokenizer: SimpleTokenizer, seq_len: int = 64):
        """Tokenize and chunk texts into fixed-length sequences.

        Args:
            texts: Raw text strings.
            tokenizer: SimpleTokenizer instance.
            seq_len: Fixed sequence length. Sequences are padded or truncated.
        """
        self.seq_len = seq_len
        self.tokenizer = tokenizer
        self.sequences: list[torch.Tensor] = []

        all_ids: list[int] = []
        for text in texts:
            all_ids.extend(tokenizer.encode(text))

        # Chunk into fixed-length sequences
        for i in range(0, len(all_ids) - seq_len + 1, seq_len):
            chunk = all_ids[i : i + seq_len]
            self.sequences.append(torch.tensor(chunk, dtype=torch.long))

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.sequences[idx]


def load_text_dataset(
    name: str = "tinystories",
    split: str = "train",
    max_samples: Optional[int] = 10000,
) -> list[str]:
    """Load a text dataset by name.

    Args:
        name: Dataset name. Options: "tinystories", "wikitext".
        split: Dataset split ("train", "validation", "test").
        max_samples: Maximum number of samples to load. None for all.

    Returns:
        List of text strings.
    """
    from datasets import load_dataset

    if name == "tinystories":
        ds = load_dataset("roneneldan/TinyStories", split=split)
        texts = ds["text"]
    elif name == "wikitext":
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        texts = [t for t in ds["text"] if t.strip()]
    else:
        raise ValueError(f"Unknown dataset: {name}. Use 'tinystories' or 'wikitext'.")

    if max_samples is not None:
        texts = texts[:max_samples]

    return texts
