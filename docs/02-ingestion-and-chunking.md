# 02 · Ingestion & Chunking

> Code: `loaders.py`, `chunking.py`

Chunking is the most underrated lever in RAG. Your retriever can only return chunks you
created — if a chunk cuts a table header off its rows, or splits a definition from its
term, no amount of clever retrieval recovers the lost meaning.

## The failure mode

Fixed-size character splitting (the naive default) cuts across sentences and structure.
The model then sees fragments: a number with no label, a pronoun with no antecedent, a
"yes" with no question. Retrieval scores look fine; answers are subtly wrong.

## The two ideas here

### 1. Parent / child chunks (parent-document retrieval)

There's a tension: **small** chunks retrieve precisely (a focused embedding matches a
focused query), but **large** chunks give the model enough context to answer. You want
both. So we:

- split each document into large **parent** blocks (`parent_chunk_tokens`, ~1200), then
- split each parent into small **child** chunks (`chunk_tokens`, ~320, with overlap).

We **embed and search the children**, but at generation time we **swap in the parent**
block (`retriever._expand_to_parents` + `generate.py`). Best of both: precise matching,
full context. If several children of the same parent are retrieved, we collapse them to
one parent so the prompt isn't bloated with duplicates.

See `chunk_documents()` — it sentence-splits, greedily packs sentences into parents, then
packs each parent's sentences into overlapping children. Overlap (`chunk_overlap_tokens`)
prevents an answer that straddles a boundary from being lost.

### 2. Semantic chunking (optional)

For messy prose where structure is a poor guide, `SemanticChunker` embeds each sentence
and starts a new chunk wherever the topic *shifts* — measured as embedding distance
between consecutive sentences crossing a percentile threshold. Cleaner boundaries, at the
cost of embeddings up front. Turn it on with `arag ingest <path> --semantic`.

## Loaders

`loaders.py` is intentionally tiny: `.txt/.md/.rst` and `.pdf` (page-by-page, so a page
number rides along as metadata). Add formats here; keep structure-handling in chunking so
loaders stay trivially testable.

## Metadata

Every chunk carries `metadata` (e.g. `page`, `loc`, and anything your loader adds). This
powers **filtering** (doc 03) — "only EMEA", "only 2025", "only this source". Enrich
metadata at ingestion time; it's nearly free and pays off at query time.

## Tuning cheatsheet

| Symptom | Try |
|---|---|
| Answers miss context / feel fragmented | ↑ `parent_chunk_tokens`, ↑ overlap |
| Retrieval is imprecise / off-topic | ↓ `chunk_tokens` |
| Boundaries cut mid-thought on prose | `--semantic` |
| Tables/code mangled | pre-split by structure in a custom loader |

> Token counts here use a cheap ~4-chars/token estimate (`approx_tokens`). It's good
> enough for sizing; swap in a real tokenizer if you need exactness.
