# 01 · Architecture

## Data flow

```
                          INGEST (one-time)
  files ──► loaders ──► chunking ──► embed ──► HybridStore (vectors + BM25)
   .md/.pdf  Document     Chunk      Gemini          │
                          (parent/                    └──► GraphIndex
                           child)                          (entities, relations,
                                                            communities)

                          QUERY (per question)
  question
     │
     ▼  RAGPipeline.ask(mode=...)
     ├── mode="simple"  ─► AdvancedRetriever ─► Generator ─► Answer
     │                     (expand→HyDE→hybrid→rerank→parent→CRAG)
     │
     └── mode="agentic" ─► AgenticRAG
                            plan → route → act → verify → stop → synthesize
                            (routes sub-questions to hybrid / graph_local / graph_global)
```

## Module map

| File | Responsibility | Key class/fn |
|------|----------------|--------------|
| `config.py` | All tunables + feature flags | `Settings` |
| `gemini.py` | The only code that calls the API | `Gemini` (embed / generate / generate_structured) |
| `schema.py` | Core data types | `Document`, `Chunk`, `Scored` |
| `loaders.py` | Read files → `Document` | `load_path` |
| `chunking.py` | Split into retrievable units | `chunk_documents`, `SemanticChunker` |
| `store.py` | Hybrid index + fusion + filters + persistence | `HybridStore` |
| `query_understanding.py` | Rewrite/expand queries | `QueryUnderstanding` |
| `rerank.py` | Re-score candidates | `Reranker` |
| `crag.py` | Grade retrieval quality | `CRAG` |
| `retriever.py` | Orchestrate the retrieval flow | `AdvancedRetriever` |
| `generate.py` | Grounded answer + citations | `Generator` |
| `graph.py` | GraphRAG | `GraphIndex` |
| `agent.py` | Multi-hop agent | `AgenticRAG` |
| `pipeline.py` | The public façade | `RAGPipeline` |
| `evaluate.py` | Metrics | `Evaluator` |
| `cli.py` | `arag` command | — |

## Design choices worth knowing

- **One API wrapper (`gemini.py`).** Retries, batching, embedding task-types, and
  normalization live in exactly one place. Every "agentic" decision (plan, route, grade,
  rerank, extract) is just `generate_structured(...)` into a Pydantic schema.
- **Plain, serializable index.** The store is a NumPy matrix + JSONL on disk — no vector
  DB to run. You can `cat .arag_index/chunks.jsonl` and see exactly what was indexed.
  For scale, swap the brute-force search for an ANN index (FAISS/ScaNN); the interface
  doesn't change.
- **Everything is a toggle.** `Settings` flags let you ablate any technique and re-run
  the evaluator. This is deliberate — the repo is meant to be *measured*, not admired.
- **Two models on purpose.** A fast model (`gemini-2.5-flash`) does the many small
  calls (routing, grading, expansion); a stronger model (`gemini-2.5-pro`) is reserved
  for hard multi-hop synthesis. Cost and latency stay sane.
