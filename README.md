# Advanced RAG on Google Gemini

A production-shaped, **advanced Retrieval-Augmented Generation** pipeline built end to
end on Google Gemini — and a teaching codebase. Every technique from the "advanced RAG"
playbook is implemented as a small, inspectable, independently-toggleable layer, so you
can read it, run it, ablate it, and measure it.

> New to the concepts? Start with [`docs/00-overview.md`](docs/00-overview.md) — it's a
> guided learning path that maps each idea to the exact file that implements it.

## What's inside

| Technique | Where | What it does |
|---|---|---|
| Parent/child + semantic chunking | `chunking.py` | Retrieve precise children, feed the model the parent block |
| Hybrid retrieval (dense + BM25) | `store.py` | Meaning **and** exact tokens, fused with Reciprocal Rank Fusion |
| Metadata filtering | `store.py` | Restrict by source/region/date before ranking |
| Query understanding | `query_understanding.py` | Multi-query expansion + HyDE |
| Reranking | `rerank.py` | Gemini cross-encoder re-scores candidates by usefulness |
| Grounding + citations | `generate.py` | Answer only from sources, cite `[n]`, else "I don't know" |
| Corrective RAG (CRAG) | `crag.py` | Grade evidence before generating; broaden if weak |
| GraphRAG | `graph.py` | Entity/relation graph, k-hop local + community global search |
| Agentic RAG | `agent.py` | plan → route → act → verify → stop for multi-hop questions |
| Conversation memory | `pipeline.py` | Rewrite follow-ups into standalone questions |
| Evaluation | `evaluate.py` | Recall@k, MRR, LLM-judge groundedness & relevance |

## Architecture

```
loaders ─► chunking ─► HybridStore ─► AdvancedRetriever ─► Generator ─► Answer
 (txt/pdf)  (parent/    (dense+BM25     (expand→HyDE→hybrid    (grounded,
            child)       +RRF+meta)      →rerank→parent→CRAG)   cited)
                                │
                                ├─► GraphIndex (entities, relations, communities)
                                │
                                └─► AgenticRAG (plan/route/act/verify/stop)
```

## Setup

```bash
uv venv && uv pip install -e .        # or: pip install -e .
cp .env.example .env                  # then add your key
# GOOGLE_API_KEY=...  (get one at https://aistudio.google.com/apikey)
```

## Quickstart (CLI)

```bash
arag ingest data/sample              # chunk → embed → build graph → persist
arag info                            # index stats
arag ask "Which enterprise customers renewed in Q2 2025 and also opened an SSO ticket?" --trace
arag chat                            # interactive, with memory
arag evaluate data/eval.json --out report.json
```

## Quickstart (Python)

```python
from advanced_rag import RAGPipeline

pipe = RAGPipeline()
pipe.ingest("data/sample")                       # one-time
res = pipe.ask("Which plans include SCIM?", mode="agentic")
print(res.render())
```

Or run the end-to-end demo: `python examples/quickstart.py`.

## Configuration

All knobs live in `advanced_rag/config.py` and can be overridden via env vars
(`ARAG_GEN_MODEL`, `ARAG_EMBED_DIM`, …). Feature flags (`rerank_enabled`,
`hyde_enabled`, `crag_enabled`, `multiquery_enabled`) let you ablate one technique at a
time and re-run `arag evaluate` to see the delta — the core "change one thing, measure"
workflow.

## Tests

Offline tests (no API key) cover chunking, hybrid search, RRF, filtering, persistence:

```bash
python -m pytest tests/ -q       # or: python tests/test_offline.py
```

## Models

- Generation / routing / grading: `gemini-2.5-flash`
- Hard multi-hop synthesis: `gemini-2.5-pro`
- Embeddings: `gemini-embedding-001` @ 1536-dim (Matryoshka-truncated, L2-normalized)

See [`docs/`](docs/) for the full guided tour.
