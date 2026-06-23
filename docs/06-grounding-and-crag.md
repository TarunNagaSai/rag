# 06 · Grounding & Corrective RAG (CRAG)

> Code: `generate.py`, `crag.py`

Good retrieval still isn't enough. A model handed thin or off-topic context will
confidently fill the gaps. Two mechanisms keep answers tethered to evidence.

## Grounding (`generate.py`)

Three reinforcing controls:

1. **A strict system prompt** (`GROUNDING_SYSTEM`): answer *only* from the numbered
   sources; if they don't support an answer, reply exactly "I don't know based on the
   available sources"; never invent facts or citations.
2. **Numbered sources + mandatory `[n]` citations** after each claim. This makes answers
   auditable and lets the evaluator (and your UI) show which source backed which claim. We
   parse the `[n]` markers back out into `Answer.cited`.
3. **Parent-block context** — we feed the larger parent block, not the matched fragment,
   so the model has enough surrounding context to answer faithfully (doc 02).

The "I don't know" escape hatch is a feature, not a failure. A system that abstains when
evidence is missing is worth far more in production than one that always answers.

## CRAG — verify *before* you generate (`crag.py`)

Corrective RAG adds a cheap self-check between retrieval and generation. A grader reads the
question + retrieved passages and returns one of:

- **correct** — evidence clearly supports a complete answer → proceed.
- **ambiguous** — partially relevant → a corrective action will help.
- **insufficient** — evidence doesn't support an answer → don't let the model guess.

When the grade is ambiguous or insufficient, `AdvancedRetriever.retrieve` takes a
**corrective action**: broaden the search (drop filters, double `fused_top_k`, force HyDE),
re-rank, and re-grade. The agent (doc 08) goes further and can **switch tools** entirely
(e.g. graph → hybrid). This one feedback loop is one of the most effective hallucination
reducers available, because it attacks the root cause: generating from weak context.

## Why grade instead of just "retrieve more"?

Blindly raising `k` adds noise as often as signal and inflates latency/cost. CRAG makes the
extra work **conditional** — you only pay for a second retrieval pass on the queries that
actually need it, and you get a logged reason (`grade_reason`, in the trace) explaining why.

## See it

```bash
arag ask "What is Acme's refund SLA for Bronze plan?" --trace
```

If the corpus doesn't cover it, you should see a CRAG grade of `insufficient`, a broaden
attempt, and ultimately an honest "I don't know" rather than a fabricated SLA.

## Toggle

`crag_enabled` in `config.py`. With it off, you'll typically see `groundedness` dip on
out-of-scope questions because nothing stops the model from guessing.
