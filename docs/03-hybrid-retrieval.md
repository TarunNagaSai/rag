# 03 · Hybrid Retrieval (dense + BM25 + RRF)

> Code: `store.py`

This is where most of your quality lives.

## The failure mode

Pure vector (dense) retrieval matches **meaning**. That's great for "how do I reset my
password?" → "account recovery steps". But it quietly fails on **exact tokens**:

- IDs and codes: `SSO-4412`, `SKU AN-220`, `SOC 2`
- proper nouns and acronyms that are rare in the corpus
- anything where the *string* matters more than the *vibe*

Embeddings smear rare tokens into a fuzzy neighborhood, so the one passage that literally
contains `SSO-4412` may not be nearest. Users notice immediately.

## The fix: run both, then fuse

`HybridStore` keeps two indexes:

- **Dense** — Gemini embeddings, cosine similarity (we L2-normalize so cosine == dot
  product → a single fast matrix multiply in `dense_search`).
- **Lexical** — BM25 (`rank_bm25`) over tokenized text (`lexical_search`), which rewards
  exact term matches and rare-term specificity.

Then we merge the two ranked lists with **Reciprocal Rank Fusion (RRF)**:

```
score(doc) = Σ_lists  1 / (k + rank_in_list(doc))      # k defaults to 60
```

Why RRF and not "average the scores"? Because dense scores (cosine ∈ [-1,1]) and BM25
scores (unbounded) live on totally different scales — averaging them is meaningless. RRF
only uses **rank position**, so it's scale-free, parameter-light, and remarkably hard to
beat. A doc that ranks well in *either* list rises; a doc that ranks well in *both* wins.
See `reciprocal_rank_fusion()`.

## Embedding task types (a quietly huge detail)

`gemini-embedding-001` lets you declare how text will be used. We embed **documents** with
`task_type=RETRIEVAL_DOCUMENT` and **queries** with `RETRIEVAL_QUERY` (see `gemini.py`).
The model optimizes the two asymmetrically into a shared space — measurably better
retrieval than embedding both the same way. Easy to get wrong; easy to get right.

We also use **Matryoshka truncation** (`embed_dim=1536` of a 3072-dim model) for a smaller
footprint, and **re-normalize** after truncating so cosine still holds.

## Metadata filtering

`build_filter(sources=..., where={...})` builds a predicate applied **before** ranking, so
off-scope documents never compete. `where` matches metadata equality or list-membership
(e.g. `{"region": ["EMEA", "AMER"]}`). On the CLI: `arag ask "..." --source acme_customers`.
Filtering reduces prompt bloat and is the cleanest fix for "it answered using the wrong
region/quarter".

## Knobs (`config.py`)

| Setting | Meaning |
|---|---|
| `dense_top_k` / `lexical_top_k` | candidates pulled from each index |
| `rrf_k` | RRF constant; higher = flatter fusion |
| `fused_top_k` | kept after fusion → handed to the reranker |
| `final_top_k` | passages actually sent to the LLM |

## Scaling note

`dense_search` is brute-force (fine to tens/hundreds of thousands of chunks). Beyond that,
replace it with an ANN index (FAISS / ScaNN / a hosted vector DB). The `HybridStore`
interface stays identical — only the nearest-neighbor call changes.
