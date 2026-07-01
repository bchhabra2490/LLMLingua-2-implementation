"""
Step 4: fine-tune a small Transformer encoder for binary token
classification (preserve / discard) -- this is the actual "small model
that scores each token" you asked about.

Usage:
    python train.py --base_model distilbert-base-uncased \
        --data labeled_dataset.jsonl --epochs 5

Model choice: the LLMLingua-2 paper uses xlm-roberta-large (~550M params,
multilingual) or multilingual-BERT. If you only need English and want
something that trains fast on a single GPU (or even CPU for small
datasets), distilbert-base-uncased or bert-base-uncased are reasonable
substitutes -- less accurate, much faster to iterate on while learning.
"""

import argparse
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)

from dataset import TokenClassificationDataset


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    mask = labels != -100
    acc = (preds[mask] == labels[mask]).mean()
    return {"token_accuracy": float(acc)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", default="distilbert-base-uncased")
    p.add_argument("--data", default="labeled_dataset.jsonl")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--out_dir", default="checkpoints/token-compressor")
    args = p.parse_args()

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Training on device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForTokenClassification.from_pretrained(args.base_model, num_labels=2)
    model.to(device)

    full_ds = TokenClassificationDataset(args.data, tokenizer)
    idx_train, idx_val = train_test_split(range(len(full_ds)), test_size=0.1, random_state=42)
    train_ds = torch.utils.data.Subset(full_ds, idx_train)
    val_ds = torch.utils.data.Subset(full_ds, idx_val)

    collator = DataCollatorForTokenClassification(tokenizer)

    training_args = TrainingArguments(
        output_dir=args.out_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_steps=20,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    print(f"Saved to {args.out_dir}")


if __name__ == "__main__":
    main()
