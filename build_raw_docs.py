"""
Option B: build raw_docs.jsonl from your own text -- e.g. real system
prompts from your products, so the compressor specializes toward what
you're actually going to compress in production.

Two input modes:

1. A directory of .txt/.md files -- each FILE becomes one {"text": ...} line.
   python build_raw_docs.py --input_dir ./my_prompts --output raw_docs.jsonl

2. A single file where entries are separated by a delimiter line (default
   "---") -- useful if you've got many short system prompts in one file.
   python build_raw_docs.py --input_file ./all_prompts.txt --delimiter "---" --output raw_docs.jsonl

Either way, output is one JSON object per line: {"text": "<document>"}
which is what data_generation.py expects as input.
"""

import argparse
import json
from pathlib import Path


def from_directory(input_dir: str, output_path: str):
    n = 0
    with open(output_path, "w") as fout:
        for path in sorted(Path(input_dir).glob("*")):
            if path.suffix.lower() not in (".txt", ".md"):
                continue
            text = path.read_text().strip()
            if len(text.split()) < 5:
                continue
            fout.write(json.dumps({"text": text}) + "\n")
            n += 1
    print(f"wrote {n} documents to {output_path}")


def from_delimited_file(input_file: str, delimiter: str, output_path: str):
    raw = Path(input_file).read_text()
    chunks = [c.strip() for c in raw.split(delimiter) if c.strip()]
    n = 0
    with open(output_path, "w") as fout:
        for text in chunks:
            if len(text.split()) < 5:
                continue
            fout.write(json.dumps({"text": text}) + "\n")
            n += 1
    print(f"wrote {n} documents to {output_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", default=None)
    p.add_argument("--input_file", default=None)
    p.add_argument("--delimiter", default="---")
    p.add_argument("--output", default="raw_docs.jsonl")
    args = p.parse_args()

    if args.input_dir:
        from_directory(args.input_dir, args.output)
    elif args.input_file:
        from_delimited_file(args.input_file, args.delimiter, args.output)
    else:
        raise SystemExit("pass either --input_dir or --input_file")


if __name__ == "__main__":
    main()
