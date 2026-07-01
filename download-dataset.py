"""
Option A: pull the LLMLingua-2 authors' own released training data instead
of generating your own. Skips data_generation.py entirely -- this writes
directly to the format label_alignment.py expects.

Dataset: microsoft/MeetingBank-LLMCompressed (Pan et al., 2024) -- 5,169
meeting transcripts with GPT-4 extractive compressions, already split into
chunks (prompt_list / compressed_prompt_list), which is exactly the
chunk-wise granularity the paper found worked best.

Usage:
    pip install datasets
    python download_meetingbank.py --max_chunks 1000

Note on scale: the full dataset, once unpacked into chunks, is large enough
that the paper's authors reported 16-23 hours of training time on their
hardware. For a learning run on an M1 Mac, --max_chunks 500-2000 with
distilbert-base-uncased is a much more tractable starting point -- you can
always scale up once the pipeline is validated end to end.
"""

import argparse
import json

from datasets import load_dataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max_chunks", type=int, default=1000)
    p.add_argument("--output", default="compressed_pairs.jsonl")
    args = p.parse_args()

    ds = load_dataset("microsoft/MeetingBank-LLMCompressed", split="train")

    n = 0
    with open(args.output, "w") as fout:
        for sample in ds:
            for orig_chunk, comp_chunk in zip(sample["prompt_list"], sample["compressed_prompt_list"]):
                if len(orig_chunk.split()) < 5:
                    continue
                fout.write(json.dumps({"original": orig_chunk, "compressed": comp_chunk}) + "\n")
                n += 1
                if n >= args.max_chunks:
                    print(f"wrote {n} chunk pairs to {args.output}")
                    return
    print(f"wrote {n} chunk pairs to {args.output}")


if __name__ == "__main__":
    main()
