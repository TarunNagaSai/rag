# 05 · Query Understanding (Multi-Query + HyDE)

> Code: `query_understanding.py`

## The failure mode

People don't ask questions the way documents are written. A user types "how do I stop
randos logging into our tenant?"; the doc says "enforce SSO and SCIM provisioning for all
members." Same meaning, almost no shared words. Single-shot retrieval under-fetches
because it's anchored to the user's exact phrasing.

This gets worse for **multi-hop** questions, where the right passages use vocabulary the
question never mentions.

## Two complementary fixes

### Multi-query expansion (`expand`)

Ask the LLM to rewrite the question several ways — synonyms, expanded acronyms,
formal/informal framings — then retrieve for **each** variant and union the results
(fused again with RRF in the retriever). Different phrasings surface different passages, so
recall goes up. We always keep the original query in the set, and dedupe variants.

Controlled by `multiquery_enabled` and `multiquery_n`.

### HyDE — Hypothetical Document Embeddings (`hyde`)

A subtle, powerful trick: instead of embedding the *question*, ask the LLM to **draft a
hypothetical answer** and embed *that*. Why? Because an answer passage in your corpus looks
far more like *another answer* than like a *question*. The hypothetical answer lands in the
right neighborhood of the embedding space even when the question doesn't.

The draft can be partly wrong — that's fine. It's never shown to the user; it's purely a
retrieval probe. The real, grounded answer is generated later from actually-retrieved
sources (doc 06). Controlled by `hyde_enabled`.

## How they plug in

In `retriever.AdvancedRetriever._gather`, every expanded query runs a hybrid search; the
HyDE draft runs one more; all the ranked lists are fused with RRF. So query understanding
multiplies your shots on goal, and fusion keeps the result coherent.

## The trade-off

Both techniques add LLM calls (latency, cost) and can *over-fetch* — pulling in loosely
related passages. That's exactly why they sit **upstream of the reranker and CRAG**: cast a
wide, high-recall net here, then let reranking (doc 04) and corrective grading (doc 06)
restore precision. Recall first, precision last.

## Measure it

Toggle `hyde_enabled` / `multiquery_enabled` independently and watch `recall@k`. These help
most on paraphrase-heavy or multi-hop datasets and do little on keyword-exact ones — so the
*right* setting depends on your traffic. Let the eval set decide.
