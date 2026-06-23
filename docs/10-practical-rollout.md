# 10 · Practical Rollout, Tuning & Scaling

> The guide's core advice: advanced RAG is **steady, incremental improvement**, not one
> trick. Add one technique, measure it, keep what works.

## The staged plan (in the order to adopt)

Each stage maps to flags in `config.py` so you can turn the system "up" gradually and watch
the eval move at every step.

1. **Stabilize the baseline.** Good embeddings, sane chunking, clean metadata. Get
   `arag evaluate` running and record the numbers. *Everything else is measured against
   this.*
2. **Add reranking.** Highest ROI. (`rerank_enabled=True`) — doc 04.
3. **Add hybrid search.** Catch exact tokens BM25-style and fuse with RRF — doc 03.
   (Always on in this repo; it's the store's default.)
4. **Add query understanding.** Multi-query + HyDE to bridge phrasing gaps
   (`multiquery_enabled`, `hyde_enabled`) — doc 05.
5. **Optimize context supply.** Parent-document retrieval + (optional) semantic chunking so
   the model gets full, clean context — doc 02.
6. **Structure data as a graph.** Entities + relationships with provenance for multi-hop
   and global questions — doc 07.
7. **Enable agentic multi-step Q&A.** Plan → route → act → verify → stop, with per-claim
   citations and auditable paths — doc 08.
8. **Harden grounding.** Strict prompts + CRAG checks + citation tagging to cut
   hallucinations (`crag_enabled`) — doc 06.

## Tuning quick-reference

| You observe | First lever to try |
|---|---|
| Misses exact IDs / names | hybrid is doing its job — check tokenization; ↑ `lexical_top_k` |
| Right doc retrieved, wrong answer | ↑ `parent_chunk_tokens`; verify citations |
| Off-topic passages in prompt | add metadata filters; ↑ rerank strictness (lower `final_top_k`) |
| Hallucinations on edge questions | `crag_enabled=True`; tighten `GROUNDING_SYSTEM` |
| Multi-hop questions fail | `mode="agentic"`; build the graph |
| Too slow / expensive | ↓ `multiquery_n`; `hyde_enabled=False`; `mode="simple"`; cache |
| Recall low on paraphrases | `hyde_enabled=True`; ↑ `multiquery_n` |

## Cost & latency levers

- **Two-model split** — fast model for the many small calls (route/grade/expand/rerank),
  strong model only for hard synthesis. Already wired (`gen_model` vs `reasoning_model`).
- **`mode="simple"`** for latency-sensitive single lookups; reserve `agentic` for genuinely
  multi-hop questions.
- **Cache** embeddings of repeated queries and answers to common questions.
- **Batch** embeddings (the store already does; tune `embed_batch_size`).
- **Smaller `embed_dim`** (Matryoshka) shrinks the index and speeds dense search with
  modest quality cost — measure it.

## Scaling to real corpora

- **Vector search**: swap brute-force `dense_search` for an ANN index (FAISS / ScaNN /
  hosted vector DB) — same interface.
- **Graph**: move from in-memory `networkx` to **Neo4j** + Cypher for large/shared graphs
  and richer traversals (doc 07).
- **Freshness**: re-embed/re-extract on data changes so the index doesn't reflect
  yesterday's state. Stale embeddings are a silent accuracy leak.
- **Orchestration**: the agent loop here is intentionally simple. For complex routing and
  long-running flows, frameworks like LangGraph / LlamaIndex can manage the
  plan→route→act→verify→stop graph — the *concepts* in this repo transfer directly.

## When is it "ship-ready"?

When, on **your** eval set built from **your** traffic: recall@k is high, groundedness is
high, the system **abstains** instead of guessing on out-of-scope questions, latency fits
your SLA, and you can **explain** any answer from its citations (and graph paths). Then keep
the eval running as you optimize — quality is a setting you maintain, not a milestone you
pass.
