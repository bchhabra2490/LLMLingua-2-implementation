"""
Step 3: Tokenize word-labeled examples for a subword-based encoder and
align labels to subword tokens. Standard practice for token classification
with subword tokenizers: a word's label is copied onto only its FIRST
subword; the rest of that word's subwords get label -100 so they're
ignored by the loss (otherwise the model would be rewarded/penalized
multiple times for a single word's decision).
"""

import json
import torch
from torch.utils.data import Dataset


class TokenClassificationDataset(Dataset):
    def __init__(self, jsonl_path: str, tokenizer, max_length: int = 256):
        self.examples = [json.loads(l) for l in open(jsonl_path)]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        words, word_labels = ex["words"], ex["labels"]

        enc = self.tokenizer(
            words,
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length,
        )
        word_ids = enc.word_ids(batch_index=0) if hasattr(enc, "word_ids") else enc.word_ids()

        labels = []
        prev_word_id = None
        for wid in word_ids:
            if wid is None:
                labels.append(-100)
            elif wid != prev_word_id:
                labels.append(word_labels[wid])
            else:
                labels.append(-100)
            prev_word_id = wid

        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": labels,
        }
