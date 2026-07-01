"""
Step 5: inference. Given the trained token classifier and a target
compression ratio, score every word's "preserve" probability and keep
the top N words needed to hit that ratio, maintaining original order --
this is the "budget controller" step from the paper.

Usage:
    python compress.py --checkpoint checkpoints/token-compressor \
        --text "your long prompt here" --ratio 0.5
"""

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification


def compress(text: str, tokenizer, model, ratio: float = 0.5) -> str:
    words = text.split()
    enc = tokenizer(words, is_split_into_words=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        logits = model(**enc).logits[0]  # (seq_len, 2)
    probs = torch.softmax(logits, dim=-1)[:, 1]  # P(preserve) per subword token

    word_ids = enc.word_ids(batch_index=0)
    word_scores = {}
    for tok_idx, wid in enumerate(word_ids):
        if wid is None or wid in word_scores:
            continue
        word_scores[wid] = probs[tok_idx].item()

    target_n = max(1, round(len(words) * ratio))
    ranked = sorted(word_scores.items(), key=lambda kv: kv[1], reverse=True)
    keep_indices = set(idx for idx, _ in ranked[:target_n])

    kept_words = [w for i, w in enumerate(words) if i in keep_indices]
    return " ".join(kept_words)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/token-compressor")
    p.add_argument("--text", required=True)
    p.add_argument("--ratio", type=float, default=0.5)
    args = p.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForTokenClassification.from_pretrained(args.checkpoint)
    model.eval()

    result = compress(args.text, tokenizer, model, args.ratio)
    print(result)


if __name__ == "__main__":
    main()
