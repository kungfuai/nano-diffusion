"""
Text dataset for MDLM training.

Supports HuggingFace text datasets (e.g., wikitext, openwebtext).
Tokenizes text into fixed-length sequences of token IDs.
"""

from typing import Optional
import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    """
    Loads a HuggingFace text dataset, tokenizes it, and chunks into
    fixed-length sequences.

    Args:
        dataset_name: HuggingFace dataset name (e.g., 'wikitext', 'openwebtext')
        dataset_config: Dataset configuration name (e.g., 'wikitext-2-raw-v1')
        tokenizer_name: HuggingFace tokenizer name
        seq_length: Fixed sequence length for each example
        split: Dataset split ('train' or 'test')
        max_examples: Maximum number of sequences to create (None = use all)
        cache_dir: Cache directory for downloaded datasets
    """

    def __init__(
        self,
        dataset_name: str = "wikitext",
        dataset_config: str = "wikitext-2-raw-v1",
        tokenizer_name: str = "gpt2",
        seq_length: int = 128,
        split: str = "train",
        max_examples: Optional[int] = None,
        cache_dir: Optional[str] = None,
    ):
        from datasets import load_dataset
        from transformers import AutoTokenizer

        self.seq_length = seq_length

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, cache_dir=cache_dir
        )

        # Add mask token if not present
        if self.tokenizer.mask_token is None:
            self.tokenizer.add_special_tokens({"mask_token": "<mask>"})
        self.mask_token_id = self.tokenizer.mask_token_id

        # Ensure pad token exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.vocab_size = len(self.tokenizer)

        # Load dataset
        if dataset_config:
            raw_dataset = load_dataset(
                dataset_name, dataset_config, split=split, cache_dir=cache_dir
            )
        else:
            raw_dataset = load_dataset(
                dataset_name, split=split, cache_dir=cache_dir
            )

        # Determine text column
        text_column = "text"
        if text_column not in raw_dataset.column_names:
            # Try common alternatives
            for col in ["content", "sentence", "document"]:
                if col in raw_dataset.column_names:
                    text_column = col
                    break

        # Tokenize all text and concatenate into one long sequence
        all_token_ids = []
        for example in raw_dataset:
            text = example[text_column]
            if not text or not text.strip():
                continue
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            all_token_ids.extend(tokens)

        # Chunk into fixed-length sequences
        num_sequences = len(all_token_ids) // seq_length
        if max_examples is not None:
            num_sequences = min(num_sequences, max_examples)

        self.sequences = torch.tensor(
            all_token_ids[: num_sequences * seq_length], dtype=torch.long
        ).reshape(num_sequences, seq_length)

        print(
            f"TextDataset: {num_sequences} sequences of length {seq_length} "
            f"(vocab_size={self.vocab_size}, mask_token_id={self.mask_token_id})"
        )

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return {
            "input_ids": self.sequences[idx],
            "attention_mask": torch.ones(self.seq_length, dtype=torch.float),
        }
