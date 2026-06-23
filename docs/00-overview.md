# 00 · Overview & Learning Path

Welcome. This repo is two things at once:

1. A working **advanced RAG pipeline** on Google Gemini.
2. A **course**: each technique is a small file you can read in one sitting, with a doc
   (this folder) explaining the *why* behind the *how*.

## The one-paragraph mental model

A basic RAG system embeds your documents, finds the nearest few to a question, and
stuffs them into the prompt. It demos well and breaks in production: it misses exact
terms, retrieves near-duplicates, cuts documents at the wrong place, forgets
conversational constraints, and hallucinates when the retrieved context is thin.
**Advanced RAG** is the collection of techniques that fix each of those failure modes —
better *evidence*, better *context*, better *reasoning*, and a *verification* step — so
answers become accurate, explainable, and repeatable at scale.

## Suggested reading order

| # | Doc | Technique | Code |
|---|-----|-----------|------|
| 01 | [Architecture](01-architecture.md) | How the pieces fit | `pipeline.py` |
| 02 | [Ingestion & Chunking](02-ingestion-and-chunking.md) | parent/child, semantic chunking | `chunking.py`, `loaders.py` |
| 03 | [Hybrid Retrieval](03-hybrid-retrieval.md) | dense + BM25 + RRF + filters | `store.py` |
| 04 | [Reranking](04-reranking.md) | cross-encoder reranking | `rerank.py` |
| 05 | [Query Understanding](05-query-understanding.md) | multi-query + HyDE | `query_understanding.py` |
| 06 | [Grounding & CRAG](06-grounding-and-crag.md) | citations + corrective loop | `generate.py`, `crag.py` |
| 07 | [GraphRAG](07-graphrag.md) | knowledge-graph retrieval | `graph.py` |
| 08 | [Agentic RAG](08-agentic-rag.md) | plan/route/act/verify/stop | `agent.py` |
| 09 | [Evaluation](09-evaluation.md) | recall@k, MRR, LLM-judge | `evaluate.py` |
| 10 | [Practical Rollout](10-practical-rollout.md) | staged plan, tuning, scaling | — |

## How to study this repo

For each layer:

1. **Read the doc** here for the concept + the failure mode it fixes.
2. **Read the file** it points to — they're short and heavily commented.
3. **Toggle it off** in `config.py` (e.g. `rerank_enabled = False`), run
   `arag evaluate data/eval.json`, and watch the metric move. That ablation loop is the
   single most important habit in RAG work.

## The golden rules (everything else is detail)

1. **Retrieval is the bottleneck.** The model can only be as good as what you show it.
2. **Hybrid beats pure-vector** because real questions contain exact tokens (IDs, names).
3. **Rerank** — first-pass ranking optimizes proximity, not usefulness.
4. **Ground hard** — answer only from sources, cite, and allow "I don't know".
5. **Verify before you generate** (CRAG) — a thin context is a hallucination waiting to happen.
6. **Measure one change at a time.** No eval set = no progress, just vibes.
