"""Data preprocessing pipeline for training diffusion language models on custom data.

Handles the full pipeline: raw text -> tokenization -> chunking -> Dataset.
Supports loading from plain text files, JSONL, CSV, or HuggingFace datasets.

References:
    - MDLM (Sahoo et al., 2024): https://arxiv.org/abs/2406.07524
    - D3PM (Austin et al., 2021): https://arxiv.org/abs/2107.03006
"""

import os
import json
from pathlib import Path
from typing import Optional, Union

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


class CustomDataPipeline:
    """End-to-end pipeline for preparing text data for diffusion LM training.

    Takes raw text from various sources and produces tokenized, chunked
    sequences ready for training. Handles padding, truncation, and the
    creation of attention masks.

    Args:
        tokenizer_name: HuggingFace tokenizer name or path.
        max_seq_len: Maximum sequence length after tokenization.
        mask_token: Token to use as the absorbing state for masked diffusion.
            If None, uses the tokenizer's default mask token.
        chunk_overlap: Number of overlapping tokens between consecutive chunks.
            Helps preserve context at chunk boundaries.
    """

    def __init__(
        self,
        tokenizer_name: str = "bert-base-uncased",
        max_seq_len: int = 128,
        mask_token: Optional[str] = None,
        chunk_overlap: int = 0,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_seq_len = max_seq_len
        self.chunk_overlap = chunk_overlap

        # Set up mask token
        if mask_token is not None:
            if mask_token not in self.tokenizer.get_vocab():
                self.tokenizer.add_special_tokens({"mask_token": mask_token})
            self.mask_token_id = self.tokenizer.convert_tokens_to_ids(mask_token)
        elif self.tokenizer.mask_token is not None:
            self.mask_token_id = self.tokenizer.mask_token_id
        else:
            # Add [MASK] if tokenizer doesn't have one
            self.tokenizer.add_special_tokens({"mask_token": "[MASK]"})
            self.mask_token_id = self.tokenizer.mask_token_id

        self.vocab_size = len(self.tokenizer)
        self.pad_token_id = self.tokenizer.pad_token_id or 0

    def load_text_file(self, path: str) -> list[str]:
        """Load raw text from a plain text file.

        Splits on double newlines to create separate documents/paragraphs.

        Args:
            path: Path to a .txt file.

        Returns:
            List of text strings (one per document/paragraph).
        """
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        # Split on double newlines to get documents
        documents = [doc.strip() for doc in text.split("\n\n") if doc.strip()]
        return documents

    def load_jsonl(self, path: str, text_field: str = "text") -> list[str]:
        """Load text from a JSONL file.

        Args:
            path: Path to a .jsonl file.
            text_field: JSON key containing the text.

        Returns:
            List of text strings.
        """
        documents = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    if text_field in record:
                        documents.append(record[text_field])
        return documents

    def load_csv(
        self, path: str, text_column: str = "text", delimiter: str = ","
    ) -> list[str]:
        """Load text from a CSV file.

        Args:
            path: Path to a .csv file.
            text_column: Column name containing the text.
            delimiter: CSV delimiter character.

        Returns:
            List of text strings.
        """
        import csv

        documents = []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            for row in reader:
                if text_column in row and row[text_column].strip():
                    documents.append(row[text_column])
        return documents

    def load_auto(self, path: str, text_field: str = "text") -> list[str]:
        """Auto-detect file format and load text.

        Args:
            path: Path to a text, JSONL, or CSV file.
            text_field: Field/column name for the text.

        Returns:
            List of text strings.
        """
        ext = Path(path).suffix.lower()
        if ext == ".jsonl":
            return self.load_jsonl(path, text_field)
        elif ext == ".csv":
            return self.load_csv(path, text_field)
        else:
            return self.load_text_file(path)

    def tokenize_and_chunk(self, documents: list[str]) -> list[list[int]]:
        """Tokenize documents and split into fixed-length chunks.

        Long documents are split into chunks of max_seq_len tokens.
        Short documents are kept as-is (padding happens later in the Dataset).

        Args:
            documents: List of raw text strings.

        Returns:
            List of token ID lists, each of length <= max_seq_len.
        """
        all_chunks = []

        for doc in documents:
            token_ids = self.tokenizer.encode(
                doc, add_special_tokens=False, truncation=False
            )

            if len(token_ids) == 0:
                continue

            # If short enough, keep as a single chunk
            if len(token_ids) <= self.max_seq_len:
                all_chunks.append(token_ids)
            else:
                # Split into overlapping chunks
                stride = self.max_seq_len - self.chunk_overlap
                for start in range(0, len(token_ids), stride):
                    chunk = token_ids[start : start + self.max_seq_len]
                    if len(chunk) > self.chunk_overlap or start == 0:
                        all_chunks.append(chunk)

        return all_chunks

    def build_dataset(
        self, documents: list[str], min_length: int = 5
    ) -> "DiffusionTextDataset":
        """Build a PyTorch Dataset from raw text documents.

        Args:
            documents: List of raw text strings.
            min_length: Minimum number of tokens to keep a chunk.

        Returns:
            DiffusionTextDataset ready for DataLoader.
        """
        chunks = self.tokenize_and_chunk(documents)
        # Filter out very short chunks
        chunks = [c for c in chunks if len(c) >= min_length]
        return DiffusionTextDataset(
            chunks=chunks,
            max_seq_len=self.max_seq_len,
            pad_token_id=self.pad_token_id,
            mask_token_id=self.mask_token_id,
            vocab_size=self.vocab_size,
        )

    def build_from_file(
        self, path: str, text_field: str = "text", min_length: int = 5
    ) -> "DiffusionTextDataset":
        """Convenience method: load a file and build a dataset in one call.

        Args:
            path: Path to data file.
            text_field: Field name for text (for JSONL/CSV).
            min_length: Minimum tokens per chunk.

        Returns:
            DiffusionTextDataset ready for DataLoader.
        """
        documents = self.load_auto(path, text_field)
        return self.build_dataset(documents, min_length)


class DiffusionTextDataset(Dataset):
    """PyTorch Dataset for diffusion language model training.

    Each sample is a tokenized, padded sequence with an attention mask
    that indicates which positions are real tokens vs padding.

    Args:
        chunks: List of token ID lists.
        max_seq_len: Maximum sequence length (used for padding).
        pad_token_id: Token ID for padding.
        mask_token_id: Token ID for the absorbing/mask state.
        vocab_size: Size of the vocabulary.
    """

    def __init__(
        self,
        chunks: list[list[int]],
        max_seq_len: int,
        pad_token_id: int,
        mask_token_id: int,
        vocab_size: int,
    ):
        self.chunks = chunks
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id
        self.vocab_size = vocab_size

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get a single training sample.

        Returns:
            Dictionary with:
                - input_ids: Token IDs, shape (max_seq_len,).
                - attention_mask: 1 for real tokens, 0 for padding, shape (max_seq_len,).
                - length: Original sequence length before padding.
        """
        token_ids = self.chunks[idx]
        length = min(len(token_ids), self.max_seq_len)

        # Pad or truncate to max_seq_len
        if len(token_ids) >= self.max_seq_len:
            padded = token_ids[: self.max_seq_len]
            attention_mask = [1] * self.max_seq_len
        else:
            pad_length = self.max_seq_len - len(token_ids)
            padded = token_ids + [self.pad_token_id] * pad_length
            attention_mask = [1] * len(token_ids) + [0] * pad_length

        return {
            "input_ids": torch.tensor(padded, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "length": torch.tensor(length, dtype=torch.long),
        }

    def get_dataloader(
        self, batch_size: int = 32, shuffle: bool = True, num_workers: int = 0
    ) -> DataLoader:
        """Create a DataLoader for this dataset.

        Args:
            batch_size: Samples per batch.
            shuffle: Whether to shuffle data each epoch.
            num_workers: Number of parallel data loading workers.

        Returns:
            PyTorch DataLoader.
        """
        return DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )


def recommend_approach(
    dataset_size: int,
    avg_doc_length: int,
    task_type: str = "unconditional",
) -> dict:
    """Recommend a diffusion approach and hyperparameters based on dataset characteristics.

    This is a heuristic guide -- actual performance depends on many factors.

    Args:
        dataset_size: Number of documents in the dataset.
        avg_doc_length: Average number of tokens per document.
        task_type: One of "unconditional", "conditional", "infilling".

    Returns:
        Dictionary with recommended approach, model size, and hyperparameters.
    """
    recommendations = {
        "approach": None,
        "model_dim": None,
        "num_layers": None,
        "num_timesteps": None,
        "batch_size": None,
        "learning_rate": None,
        "notes": [],
    }

    # Choose approach based on task type
    if task_type == "infilling":
        recommendations["approach"] = "masked (MDLM)"
        recommendations["notes"].append(
            "Masked diffusion is natural for infilling -- tokens can be "
            "unmasked in any order, allowing bidirectional context."
        )
    elif task_type == "conditional":
        recommendations["approach"] = "masked (MDLM) with classifier-free guidance"
        recommendations["notes"].append(
            "MDLM with classifier-free guidance gives strong conditional "
            "generation. Train with 10-15% condition dropout."
        )
    else:
        recommendations["approach"] = "masked (MDLM)"
        recommendations["notes"].append(
            "MDLM provides the best quality-efficiency tradeoff for "
            "unconditional text generation."
        )

    # Model size based on dataset size
    if dataset_size < 1_000:
        recommendations["model_dim"] = 256
        recommendations["num_layers"] = 4
        recommendations["batch_size"] = 16
        recommendations["learning_rate"] = 3e-4
        recommendations["notes"].append(
            "Small dataset: use a small model to avoid overfitting. "
            "Consider data augmentation."
        )
    elif dataset_size < 50_000:
        recommendations["model_dim"] = 512
        recommendations["num_layers"] = 6
        recommendations["batch_size"] = 32
        recommendations["learning_rate"] = 1e-4
        recommendations["notes"].append(
            "Medium dataset: moderate model size. Train for 50-100 epochs."
        )
    else:
        recommendations["model_dim"] = 768
        recommendations["num_layers"] = 12
        recommendations["batch_size"] = 64
        recommendations["learning_rate"] = 5e-5
        recommendations["notes"].append(
            "Large dataset: can support a larger model. Train for 20-50 epochs."
        )

    # Timesteps based on sequence length
    if avg_doc_length < 64:
        recommendations["num_timesteps"] = 100
    elif avg_doc_length < 256:
        recommendations["num_timesteps"] = 500
    else:
        recommendations["num_timesteps"] = 1000
        recommendations["notes"].append(
            "Long sequences benefit from more diffusion timesteps."
        )

    return recommendations
