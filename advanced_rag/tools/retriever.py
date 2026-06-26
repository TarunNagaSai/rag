"""Retriever — wraps HybridStore with query expansion and reranking stubs."""

from __future__ import annotations

from dataclasses import dataclass, field

from advanced_rag.core.config import Settings, get_settings
from advanced_rag.llm.gemini import Gemini
from advanced_rag.schema.schema import Scored
from advanced_rag.tools.store import Filter, HybridStore


@dataclass
class RetrievalResult:
    question: str
    evidence: list[Scored]
    trace: list[str] = field(default_factory=list)


class AdvancedRetriever:
    def __init__(
        self,
        store: HybridStore,
        settings: Settings | None = None,
        gemini: Gemini | None = None,
    ):
        self.store = store
        self.s = settings or get_settings()
        self.g = gemini or Gemini(self.s)

    def retrieve(
        self, question: str, filt: Filter | None = None
    ) -> RetrievalResult:
        query_vec = self.g.embed_query(question)
        results = self.store.hybrid_search(
            question, query_vec, top_k=self.s.fused_top_k, filt=filt
        )
        return RetrievalResult(
            question=question,
            evidence=results[: self.s.final_top_k],
        )
