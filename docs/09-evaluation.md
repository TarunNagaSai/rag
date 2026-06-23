# 09 · Evaluation

> Code: `evaluate.py`, dataset: `data/eval.json`

**Don't skimp on this.** Every other doc ends with "measure it" — this is how. Without an
eval set you're not improving a RAG system, you're rearranging it by vibes.

## What we measure

Two families, matching the guide:

### Retrieval quality (cheap, deterministic)

- **Recall@k** — did at least one *correct* source make it into the retrieved top-k?
  Measures whether the right evidence is even *available* to the model.
- **MRR** (Mean Reciprocal Rank) — `1/rank` of the first correct source. Rewards putting
  the right evidence **high**, which is what reranking is for.

These compare retrieved chunk `source` against the dataset's `expected_sources`
(substring match). No LLM needed — fast and repeatable.

### Answer quality (LLM-as-judge)

- **Groundedness / faithfulness** — is every claim in the answer supported by the
  retrieved sources? This is your hallucination meter. (An honest "I don't know" with no
  evidence scores as grounded — abstaining is correct behavior.)
- **Answer relevance** — does the answer actually address the question (optionally compared
  to an `ideal_answer`)?

A calibrated Gemini judge scores each `[0,1]` with a short justification (`_Judge`).

## Dataset format

```json
[
  {
    "question": "Which plans include SCIM provisioning?",
    "expected_sources": ["acme_product.md"],
    "ideal_answer": "Only Platinum includes SCIM."
  }
]
```

`expected_sources` drives retrieval metrics; `ideal_answer` is optional guidance for the
relevance judge.

## Run it

```bash
arag evaluate data/eval.json --mode agentic --out report.json
```

You get a summary table (recall@k, mrr, groundedness, answer_relevance) and a full
per-case report with the actual answers — read the failures, that's where the signal is.

## The workflow that actually improves systems

1. Establish a **baseline** (current settings).
2. **Change one thing** — a flag in `config.py`, a chunk size, a model.
3. Re-run `arag evaluate` and compare to baseline.
4. Keep the change if it helps; revert if it doesn't; never bundle changes.

This is the single discipline that separates a RAG system that improves over time from one
that drifts. Treat `config.py` flags as your experiment grid:

| Hypothesis | Flag to flip |
|---|---|
| "Reranking is carrying us" | `rerank_enabled` |
| "HyDE helps our paraphrased queries" | `hyde_enabled` |
| "CRAG is cutting hallucinations" | `crag_enabled` |
| "Multi-query boosts recall" | `multiquery_enabled` |
| "Bigger parents = better answers" | `parent_chunk_tokens` |

## Caveats

- LLM-judge scores are **relative**, not absolute truth — use them to compare configs, and
  keep the judge model/prompt fixed across runs so comparisons are fair.
- Build a **real** eval set from your own traffic (20–50 questions to start). The
  4-question `data/eval.json` is a demo, not a benchmark.
