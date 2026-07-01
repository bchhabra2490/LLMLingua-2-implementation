"""
Step 1: Data generation for LLMLingua-2-style training.

Uses a strong LLM (here: Claude) to produce extractive compressions of
documents -- the LLM is instructed to ONLY delete words, never rewrite,
paraphrase, or reorder. This constraint is what makes it possible to
later derive word-level keep/discard labels automatically by aligning
the original text to the compressed text (see label_alignment.py). The
original LLMLingua-2 paper used GPT-4 for this step; any strong
instruction-following model works, including Claude.

The paper also found that compressing document CHUNKS independently
(rather than one huge document at once) mattered a lot for annotation
quality -- large one-shot compressions had higher hallucination rates.
We follow the same chunk-wise strategy here.

Use text similar to what you actually want to compress in production --
LLMLingua-2 trained on meeting transcripts because that was their target
domain. If you're compressing long system prompts, train on system
prompts / instructions, not generic web text.
"""

import json
import re

import anthropic

COMPRESSION_SYSTEM_PROMPT = """You compress text by deleting low-information words. Rules, no exceptions:
1. You may ONLY delete words. Never add, replace, paraphrase, reorder, or summarize.
2. The words you keep must appear in the same order as in the original.
3. Keep names, numbers, dates, negations, and anything that changes literal meaning if removed.
4. Remove filler words, redundant phrases, and words that don't affect meaning: articles, hedges, repeated ideas, verbose connectives.
5. Output ONLY the compressed text. No preamble, no explanation, no quotes."""


def chunk_text(text: str, sentences_per_chunk: int = 3) -> list[str]:
    """Naive sentence-boundary chunking. Swap in a real sentence tokenizer
    (e.g. nltk.sent_tokenize) for anything beyond quick experiments."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [
        " ".join(sentences[i : i + sentences_per_chunk])
        for i in range(0, len(sentences), sentences_per_chunk)
        if sentences[i : i + sentences_per_chunk]
    ]


def compress_chunk(client: anthropic.Anthropic, chunk: str, target_ratio: float = 0.5) -> str:
    resp = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=len(chunk.split()) + 50,
        system=COMPRESSION_SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"Compress this to roughly {int(target_ratio * 100)}% of its original word count:\n\n{chunk}",
            }
        ],
    )
    return resp.content[0].text.strip()


def build_dataset(input_docs_path: str, output_path: str, target_ratio: float = 0.5):
    """
    input_docs_path: a .jsonl file with one {"text": "..."} object per line.
    output_path: writes one {"original": ..., "compressed": ...} pair per chunk.
    """
    client = anthropic.Anthropic()
    with open(input_docs_path) as fin, open(output_path, "w") as fout:
        for line in fin:
            doc = json.loads(line)["text"]
            for chunk in chunk_text(doc):
                if len(chunk.split()) < 5:
                    continue
                compressed = compress_chunk(client, chunk, target_ratio)
                fout.write(json.dumps({"original": chunk, "compressed": compressed}) + "\n")


if __name__ == "__main__":
    build_dataset("raw_docs.jsonl", "compressed_pairs.jsonl", target_ratio=0.5)
