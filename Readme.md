# Token-Classification Prompt Compressor (LLMLingua-2 style)

A small BERT-sized model that scores every word in a prompt for
"keep or discard," trained via distillation from a stronger LLM. Unlike
the gist-token encoder from before, this one outputs plain text — you can
drop the result straight into any API call, including Claude's.

Method: [LLMLingua-2: Data Distillation for Efficient and Faithful
Task-Agnostic Prompt Compression](https://arxiv.org/abs/2403.12968)
(Pan et al., 2024). Independent reimplementation, not the original code
(the original is at [microsoft/LLMLingua](https://github.com/microsoft/LLMLingua)
if you want to compare against a production implementation).

## Reading list, in order

1. **Selective-Context** (Li et al., 2023) — [arXiv:2310.06201](https://arxiv.org/abs/2310.06201) — the earliest approach, drops low-self-information tokens.
2. **LLMLingua** (Jiang et al., EMNLP 2023) — [arXiv:2310.05736](https://arxiv.org/abs/2310.05736) — perplexity-based compression with a budget controller across prompt sections.
3. **LongLLMLingua** (Jiang et al., ACL 2024) — [arXiv:2310.06839](https://arxiv.org/abs/2310.06839) — question-aware, fixes long-context position bias. Read this if you ever compress RAG contexts.
4. **LLMLingua-2** (Pan et al., 2024) — [arXiv:2403.12968](https://arxiv.org/abs/2403.12968) — **this repo.** Reframes compression as token classification with GPT-4 distillation.

## Getting input data

Two options, not mutually exclusive:

**A. Use the authors' released dataset (fastest way to a working pipeline)**
```bash
pip install datasets
python download_meetingbank.py --max_chunks 1000
# writes compressed_pairs.jsonl directly -- skip data_generation.py, go
# straight to label_alignment.py
```
`microsoft/MeetingBank-LLMCompressed` on Hugging Face has 5,169 meeting
transcripts with GPT-4 extractive compressions, already chunked. Real data,
zero API cost, validates the rest of your pipeline immediately.

**B. Use your own domain text (better fit for your actual use case)**
```bash
python build_raw_docs.py --input_dir ./my_system_prompts --output raw_docs.jsonl
# then run data_generation.py as normal, which calls Claude to compress it
```
Point it at a folder of your real system prompts / product docs, or a
single file with `---`-delimited entries. This is what actually specializes
the compressor to your domain (e.g. Morning Report's or usePostly's system
prompts) rather than meeting transcripts -- but you likely won't have
thousands of these on your own, so it's best mixed into (A)'s dataset
rather than used alone for a first model.

## Pipeline

```
data_generation.py  →  label_alignment.py  →  dataset.py  →  train.py  →  compress.py
   (get compressed        (derive per-word        (subword          (fine-tune       (run the
    text pairs from        keep/discard labels,     label              classifier)    trained
    a strong LLM)          filter by VR/AG)         alignment)                        model)
```

### 1. Generate training pairs

```bash
# raw_docs.jsonl: one {"text": "..."} per line -- use text similar to
# what you actually want to compress in production (e.g. your own long
# system prompts), not generic web text.
python data_generation.py
```

This calls Claude with a strict "delete-only" instruction so the
compression stays extractive — critical, because the next step derives
labels by aligning original and compressed text word-for-word. If the
model paraphrases instead of deleting, alignment breaks.

### 2. Derive labels + filter low-quality samples

```bash
python label_alignment.py
```

Implements the paper's exact quality metrics:

- **Variation Rate (VR)** — fraction of compressed words not found in the
  original. High VR = hallucination. Worst 5% discarded.
- **Alignment Gap (AG)** — hitting rate minus matching rate; large gap
  means the automatic labeling is unreliable for that sample. Worst 10%
  discarded.

### 3-4. Train the classifier

```bash
python train.py --base_model distilbert-base-uncased --data labeled_dataset.jsonl --epochs 5
```

Formulated as binary token classification (preserve=1, discard=0) with
cross-entropy loss — a completely standard HuggingFace `Trainer` fine-tune,
no custom attention masking needed (that's what makes this simpler to get
right than the gist-token approach).

Model size note: the paper uses xlm-roberta-large (~550M params) for
multilingual coverage. `distilbert-base-uncased` here trades some accuracy
for something that trains fast enough to iterate on while learning —
swap in `bert-base-uncased` or `xlm-roberta-base` once you want to push
quality.

### 5. Compress at inference time

```bash
python compress.py --checkpoint checkpoints/token-compressor \
    --text "your long prompt here" --ratio 0.5
```

This is the "budget controller": given a target ratio, keep the top-N
words by predicted preserve-probability, in original order.

## Expected results

LLMLingua-2 reports **2-5x compression** with output quality close to
uncompressed, and 3-6x faster inference than the original LLMLingua. Don't
expect the ~20x figure sometimes quoted for the original perplexity-based
LLMLingua — that method used a different, less robust mechanism and that
ratio doesn't transfer to the classification approach here.

## What I couldn't run for you

No GPU and no Hugging Face Hub access in this sandbox, so I couldn't
actually download `distilbert-base-uncased` or run a live training loop —
the code is structurally correct (this pattern — subword-label alignment,
`DataCollatorForTokenClassification`, `AutoModelForTokenClassification`) is
completely standard, but you should run a small smoke test (even 50
examples) before trusting it on your real dataset.

## Known gaps / next steps

- `label_alignment.py`'s aligner assumes strictly extractive compression
  (order-preserving, delete-only). If you ever loosen that constraint on
  the generation side, swap in `difflib.SequenceMatcher` instead of the
  greedy two-pointer match.
- The paper compresses **chunk-wise** (a few sentences at a time) rather
  than whole documents at once — `data_generation.py` does this, but if
  you increase chunk size, expect VR to climb (the paper observed this
  directly).
- No cross-lingual support here (English-only `distilbert`). Swap in
  `xlm-roberta-base`/`large` if you need multilingual, matching the paper.
- `compress.py` doesn't yet support "force-keep" spans (e.g. exact JSON
  keys, code blocks) — worth adding if you compress structured prompts,
  by simply setting those words' scores to 1.0 before ranking.