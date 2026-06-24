"""Reranking — the highest-ROI single addition to a basic RAG system.

First-pass retrieval (dense/BM25/RRF) optimizes for *proximity*, not *usefulness*.
A reranker re-scores each candidate against the query with a model that can read
both together (a cross-encoder). We don't have a hosted Google rerank endpoint, so
we use Gemini as an LLM reranker: it reads the query + every candidate and assigns
a graded relevance score. One structured call ranks the whole batch.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from advanced_rag.core.config import Settings, get_settings
from advanced_rag.llm.gemini import Gemini
from advanced_rag.schema.schema import ModelSettings, Scored


class _RankItem(BaseModel):
    index: int = Field(description="0-based index of the candidate passage")
    relevance: float = Field(description="Relevance to the query, 0.0 (irrelevant) to 1.0 (perfect)")


class _Ranking(BaseModel):
    rankings: list[_RankItem]


class Reranker:
    def __init__(self, gemini: Gemini | None = None, settings: Settings | None = None):
        self.s = settings or get_settings()
        self.g = gemini or Gemini(self.s)

    def rerank(self, query: str, candidates: list[Scored],
               top_k: int | None = None) -> list[Scored]:
        top_k = top_k or self.s.final_top_k
        if not candidates:
            return []
        if not self.s.rerank_enabled or len(candidates) == 1:
            return candidates[:top_k]

        passages = "\n\n".join(
            f"[{i}] (source: {c.citation})\n{c.chunk.text}"
            for i, c in enumerate(candidates)
        )
        ranking = self.g.generate_structured(
            prompt=(
                "Score how well each passage helps answer the QUERY. Judge usefulness "
                "for *answering*, not mere topical overlap. Return a relevance score in "
                "[0,1] for every passage index.\n\n"
                f"QUERY: {query}\n\nPASSAGES:\n{passages}"
            ),
            schema=_Ranking,
            settings=ModelSettings(
                system="You are a precise cross-encoder reranker for retrieval.",
                temperature=0.0,
            ),
        )
        scores = {r.index: r.relevance for r in ranking.rankings}
        reranked = [
            Scored(chunk=c.chunk, score=scores.get(i, 0.0), how="rerank")
            for i, c in enumerate(candidates)
        ]
        reranked.sort(key=lambda s: -s.score)
        return reranked[:top_k]
