"""
Step 2: Turn (original, compressed) pairs into word-level preserve/discard
labels via greedy sequential alignment, then compute the two quality
control metrics from the LLMLingua-2 paper (arXiv:2403.12968, Sec 3) so
low-quality samples can be filtered out before training.

Formulas (S_ori = original words, S_comp = compressed words, l(w) = 1 if
word w was labeled "preserve" by the alignment below):

    VR (Variation Rate) = (1/|S_comp|) * sum_{w in S_comp} 1[w not in S_ori]
    MR (Matching Rate)  = (1/|S_ori|)  * sum_{w in S_ori}  1[l(w) = 1]
    HR (Hitting Rate)   = (1/|S_comp|) * sum_{w in S_comp} 1[w in S_ori]
    AG (Alignment Gap)  = HR - MR

The paper discards the worst 5% of samples by VR and the worst 10% by AG.
High VR means the "compression" hallucinated words not in the source
(the extractive-only prompt in data_generation.py should keep this rare).
High AG means the automatic alignment is unreliable for that sample.

Alignment method: a two-pointer greedy match. This assumes strictly
extractive compression (kept words appear in the same relative order in
both texts) -- true as long as the generation prompt enforces it. If your
upstream compressor can reorder or paraphrase, replace this with a proper
sequence aligner (e.g. difflib.SequenceMatcher) instead.
"""

import json
import re


def normalize(word: str) -> str:
    return re.sub(r"[^\w]", "", word).lower()


def align(original_words: list[str], compressed_words: list[str]) -> list[int]:
    """Return a 0/1 label per original word: 1 = preserve, 0 = discard."""
    labels = [0] * len(original_words)
    norm_orig = [normalize(w) for w in original_words]
    norm_comp = [normalize(w) for w in compressed_words]

    j = 0
    for i, w in enumerate(norm_orig):
        if j < len(norm_comp) and w != "" and w == norm_comp[j]:
            labels[i] = 1
            j += 1
    return labels


def variation_rate(original_words: list[str], compressed_words: list[str]) -> float:
    orig_set = set(normalize(w) for w in original_words)
    if not compressed_words:
        return 0.0
    bad = sum(1 for w in compressed_words if normalize(w) not in orig_set)
    return bad / len(compressed_words)


def alignment_gap(original_words: list[str], compressed_words: list[str], labels: list[int]) -> float:
    mr = sum(labels) / max(len(original_words), 1)
    orig_set = set(normalize(w) for w in original_words)
    hits = sum(1 for w in compressed_words if normalize(w) in orig_set)
    hr = hits / max(len(compressed_words), 1)
    return hr - mr


def build_labeled_dataset(
    pairs_path: str,
    output_path: str,
    vr_percentile_cutoff: float = 0.95,
    ag_percentile_cutoff: float = 0.90,
):
    examples = []
    for line in open(pairs_path):
        rec = json.loads(line)
        orig_words = rec["original"].split()
        comp_words = rec["compressed"].split()
        labels = align(orig_words, comp_words)
        vr = variation_rate(orig_words, comp_words)
        ag = alignment_gap(orig_words, comp_words, labels)
        examples.append({"words": orig_words, "labels": labels, "vr": vr, "ag": ag})

    if not examples:
        print("no examples found")
        return

    vr_sorted = sorted(e["vr"] for e in examples)
    ag_sorted = sorted(e["ag"] for e in examples)
    vr_cut = vr_sorted[int(len(vr_sorted) * vr_percentile_cutoff)]
    ag_cut = ag_sorted[int(len(ag_sorted) * ag_percentile_cutoff)]

    kept = [e for e in examples if e["vr"] <= vr_cut and e["ag"] <= ag_cut]
    print(
        f"kept {len(kept)}/{len(examples)} examples after quality filtering "
        f"(VR cutoff={vr_cut:.3f}, AG cutoff={ag_cut:.3f})"
    )

    with open(output_path, "w") as fout:
        for e in kept:
            fout.write(json.dumps({"words": e["words"], "labels": e["labels"]}) + "\n")


if __name__ == "__main__":
    build_labeled_dataset("compressed_pairs.jsonl", "labeled_dataset.jsonl")
