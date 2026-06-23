# 08 · Agentic RAG (plan → route → act → verify → stop)

> Code: `agent.py`

## The failure mode

Some questions simply can't be answered by one retrieval call, no matter how good:

> "Which enterprise customers renewed in Q2 **and** opened an SSO ticket, and **who**
> manages those accounts?"

That's three sub-goals (renewals, SSO tickets, account managers) joined together. A single
top-k pull returns a mush of partially-relevant chunks and the model stitches a plausible —
often wrong — answer. You need to **decompose**, retrieve per sub-goal with the right tool,
and only then synthesize.

## The loop

`AgenticRAG.answer` runs the classic agent loop, kept deliberately bounded:

1. **Plan** (`_plan`): Gemini decides whether the question needs decomposition and, if so,
   breaks it into the minimal set of self-contained sub-questions. Simple lookups stay
   single-hop — decomposition only when it genuinely helps.
2. **Route**: each sub-question is tagged with the best tool:
   - `hybrid` — facts, names, dates (the full retriever from docs 03–06)
   - `graph_local` — relationships/joins around entities (doc 07)
   - `graph_global` — big-picture/thematic (community summaries)
3. **Act** (`_act`): run the chosen tool, collect evidence **with provenance**.
4. **Verify**: CRAG-grade each sub-question's evidence. If a graph tool returns weak
   evidence, the agent **switches to robust hybrid retrieval** and re-grades — coverage
   over cleverness.
5. **Stop**: halt when all sub-goals are covered **or** the budget is hit
   (`agent_max_hops`, `agent_max_subquestions`). Bounded loops can't run away on cost.

Finally it **synthesizes** one grounded, cited answer over the union of all collected
evidence — using the stronger `reasoning_model` when the question was decomposed, the fast
model otherwise.

## Why budgets and verification matter

An agent without termination criteria is a runaway bill. An agent without verification is a
confident liar. The two together — *bounded* exploration plus *grade-and-switch* — are what
make this safe to ship. Every decision is logged (`AgentResult.trace`, `steps`) so you can
see the plan, the tool per hop, the grade, and how much evidence each hop produced:

```bash
arag ask "Which enterprise customers renewed in Q2 and opened an SSO ticket, and who manages them?" --trace
```

## Chain-of-thought, used responsibly

The plan/verify reasoning is the agent's private scratchpad — it organizes hops and tracks
findings. But the **final answer is still strictly grounded** (doc 06): reasoning decides
*what to retrieve*, retrieved sources decide *what to claim*. Use the agentic path for
reasoning-heavy, multi-hop questions; use `mode="simple"` for latency-sensitive single
lookups. More steps = more accuracy *and* more latency/cost; match the mode to the query.
