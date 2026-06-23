"""Query understanding — bridge the vocabulary gap between how questions are asked
and how documents are written.

Two complementary techniques:

  * Multi-query expansion: rewrite the question several ways and union the results
    (boosts recall — different phrasings surface different passages).
  * HyDE (Hypothetical Document Embeddings): ask the LLM to *draft an answer*, then
    embed that. A hypothetical answer is lexically/semantically closer to real
    answer passages than the bare question is.

Keep this layer small and inspectable — it should be easy to see what was sent.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from .config import Settings, get_settings
from .gemini import Gemini


class _Rewrites(BaseModel):
    queries: list[str] = Field(description="Diverse standalone rewrites of the question")


class QueryUnderstanding:
    def __init__(self, gemini: Gemini | None = None, settings: Settings | None = None):
        self.s = settings or get_settings()
        self.g = gemini or Gemini(self.s)

    def expand(self, question: str, n: int | None = None) -> list[str]:
        """Return [original, *rewrites]. Rewrites use synonyms / alternate framings."""
        n = n or self.s.multiquery_n
        if not self.s.multiquery_enabled or n <= 0:
            return [question]
        out = self.g.generate_structured(
            prompt=(
                f"Rewrite the user question into {n} diverse, standalone search queries. "
                "Vary terminology (synonyms, expansions of acronyms, formal/informal). "
                "Each must be self-contained and answerable independently.\n\n"
                f"Question: {question}"
            ),
            schema=_Rewrites,
            system="You generate high-recall search query variations for a retrieval system.",
            temperature=0.4,
        )
        seen, queries = set(), [question]
        for q in out.queries:
            q = q.strip()
            if q and q.lower() not in seen:
                seen.add(q.lower())
                queries.append(q)
        return queries[: n + 1]

    def hyde(self, question: str) -> str | None:
        """Generate a short hypothetical answer passage to embed instead of the query."""
        if not self.s.hyde_enabled:
            return None
        return self.g.generate(
            prompt=(
                "Write a concise, factual paragraph that would directly answer the "
                "question below, as if quoting an ideal source document. Do not hedge, "
                "do not say you are unsure — this is a retrieval aid, not the final answer.\n\n"
                f"Question: {question}"
            ),
            system="You draft hypothetical answer passages for HyDE retrieval.",
            temperature=0.3,
            max_output_tokens=256,
        ).strip() or None
