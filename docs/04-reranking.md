# 04 · Reranking

> Code: `rerank.py`

If you add **one** thing to a basic RAG system, add a reranker. It's the highest
return-on-effort change available.

## The failure mode

First-pass retrieval (dense, BM25, even fused) ranks by **proximity** — how close a
passage is to the query in some space. Proximity is not the same as **usefulness**. The
top hit by cosine similarity is often topically related but doesn't actually contain the
answer; meanwhile the passage that *does* answer it sits at rank 7 and gets cut by
`final_top_k`.

## Bi-encoder vs cross-encoder (the key idea)

- **Bi-encoder** (what your vector index is): encodes the query and each document
  *separately*, then compares vectors. Fast, scalable, approximate — perfect for a first
  pass over the whole corpus.
- **Cross-encoder** (a reranker): feeds the query **and** a candidate **together** into
  one model that can read them jointly and judge relevance directly. Far more accurate,
  but too expensive to run over the whole corpus — so you only run it on the ~12
  candidates the first pass already narrowed to.

The pattern is **retrieve wide, rerank narrow**: `fused_top_k` (≈12–20) in, `final_top_k`
(≈4–6) out.

## How it works here

There's no hosted Google rerank endpoint, so we use **Gemini as an LLM reranker**
(`Reranker.rerank`). One structured call shows the query and all candidates and asks for a
relevance score in `[0,1]` per candidate index — judged on *usefulness for answering*, not
topical overlap. We sort by that score and keep the top `final_top_k`. Provenance on each
result becomes `how="rerank"`.

A single batched call (not one call per passage) keeps latency and cost down. For very
high QPS you'd swap in a dedicated cross-encoder; the function signature stays the same.

## Why this matters so much

A reranker lets you be **greedy in the first pass** (cast a wide net, high recall) without
polluting the prompt, because the reranker restores **precision** at the end. Wide recall +
sharp precision is exactly the combination basic RAG can't achieve with a single ranker.

## Measure it

Toggle `rerank_enabled = False` in `config.py`, run `arag evaluate data/eval.json`, then
turn it back on. Watch `mrr` and `groundedness`. This is usually the most dramatic single
ablation in the whole pipeline.
