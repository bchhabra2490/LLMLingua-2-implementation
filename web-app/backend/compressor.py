"""Load the latest checkpoint and compress text."""

from __future__ import annotations

import os

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from model_utils import ensure_model_weights, resolve_latest_checkpoint


def compress(text: str, tokenizer, model, ratio: float = 0.5) -> dict:
    words = text.split()
    if not words:
        return {
            "compressed": "",
            "original_word_count": 0,
            "compressed_word_count": 0,
            "actual_ratio": 0.0,
        }

    enc = tokenizer(words, is_split_into_words=True, truncation=True, return_tensors="pt")
    word_ids = enc.word_ids(batch_index=0)
    device = next(model.parameters()).device
    model_input = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**model_input).logits[0]

    probs = torch.softmax(logits, dim=-1)[:, 1]
    word_scores: dict[int, float] = {}
    for tok_idx, wid in enumerate(word_ids):
        if wid is None or wid in word_scores:
            continue
        word_scores[wid] = probs[tok_idx].item()

    target_n = max(1, round(len(words) * ratio))
    ranked = sorted(word_scores.items(), key=lambda kv: kv[1], reverse=True)
    keep_indices = {idx for idx, _ in ranked[:target_n]}
    kept_words = [w for i, w in enumerate(words) if i in keep_indices]
    compressed = " ".join(kept_words)

    return {
        "compressed": compressed,
        "original_word_count": len(words),
        "compressed_word_count": len(kept_words),
        "actual_ratio": len(kept_words) / len(words),
    }


class CompressorService:
    def __init__(self, checkpoint_path=None):
        ensure_model_weights()
        self.checkpoint_path = checkpoint_path or resolve_latest_checkpoint()
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.checkpoint_path))
        self.model = AutoModelForTokenClassification.from_pretrained(str(self.checkpoint_path))
        self.model.eval()

        if torch.backends.mps.is_available() and os.environ.get("FORCE_CPU") != "1":
            self.device = torch.device("mps")
        elif torch.cuda.is_available() and os.environ.get("FORCE_CPU") != "1":
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.model.to(self.device)

    def run(self, text: str, ratio: float) -> dict:
        ratio = max(0.05, min(1.0, ratio))
        result = compress(text, self.tokenizer, self.model, ratio)
        result["checkpoint"] = self.checkpoint_path.name
        result["target_ratio"] = ratio
        return result
