"""AdvancedRetriever — composes every retrieval technique into one inspectable flow.

    question
      │  multi-query expansion + HyDE        (query_understanding)
      ▼
    hybrid search per variant               (store: dense + BM25 + RRF)
      │  fuse variants with RRF
      ▼
    rerank candidates                       (rerank: Gemini cross-encoder)
      │  parent-document expansion          (swap child -> parent block)
      ▼
    CRAG grade ── weak? ─► broaden & retry once
      │
      ▼  evidence (+ grade, +trace)

Every stage is optional via Settings, so you can ablate techniques and measure the
delta with the evaluation harness — exactly the "change one thing, measure" loop
the guide recommends.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from advanced_rag.core.config import Settings, get_settings
from advanced_rag.tools.crag import CRAG, Grade
from advanced_rag.llm.gemini import Gemini
from advanced_rag.tools.query_understanding import QueryUnderstanding
from advanced_rag.tools.rerank import Reranker
from advanced_rag.schema.schema import Scored
from advanced_rag.tools.store import HybridStore, Filter, reciprocal_rank_fusion


@dataclass
class RetrievalResult:
    evidence: list[Scored]
    grade: str = "n/a"
    grade_reason: str = ""
    queries_used: list[str] = field(default_factory=list)
    trace: list[str] = field(default_factory=list)


class AdvancedRetriever:
    def __init__(self, store: HybridStore, settings: Settings | None = None,
                 gemini: Gemini | None = None):
        self.s = settings or get_settings()
        self.g = gemini or store.g
        self.store = store
        self.qu = QueryUnderstanding(self.g, self.s)
        self.reranker = Reranker(self.g, self.s)
        self.crag = CRAG(self.g, self.s)

    # --------------------------------------------------------------- first pass
    def _gather(self, queries: list[str], filt: Filter | None,
                use_hyde: bool, fused_top_k: int) -> list[Scored]:
        ranked_lists: list[list[str]] = []
        pool: dict[str, Scored] = {}
        for q in queries:
            qvec = self.g.embed_query(q)
            hits = self.store.hybrid_search(q, qvec, top_k=fused_top_k, filt=filt)
            ranked_lists.append([h.chunk.id for h in hits])
            for h in hits:
                pool.setdefault(h.chunk.id, h)
        if use_hyde:
            hypo = self.qu.hyde(queries[0])
            if hypo:
                hvec = self.g.embed_query(hypo)
                hits = self.store.hybrid_search(hypo, hvec, top_k=fused_top_k, filt=filt)
                ranked_lists.append([h.chunk.id for h in hits])
                for h in hits:
                    pool.setdefault(h.chunk.id, h)
        fused = reciprocal_rank_fusion(ranked_lists, k=self.s.rrf_k)
        ranked = sorted(fused.items(), key=lambda kv: -kv[1])[:fused_top_k]
        return [Scored(chunk=pool[cid].chunk, score=score, how="rrf") for cid, score in ranked]

    @staticmethod
    def _expand_to_parents(scored: list[Scored]) -> list[Scored]:
        """Parent-document retrieval: collapse children to unique parent blocks,
        keeping the best-scoring child as the representative + citation anchor."""
        best: dict[str, Scored] = {}
        for s in scored:
            pid = s.chunk.parent_id
            if pid not in best or s.score > best[pid].score:
                best[pid] = s
        return sorted(best.values(), key=lambda s: -s.score)

    # ------------------------------------------------------------------- public
    def retrieve(self, question: str, filt: Filter | None = None) -> RetrievalResult:
        trace: list[str] = []
        queries = self.qu.expand(question)
        trace.append(f"expanded into {len(queries)} queries")

        candidates = self._gather(queries, filt, self.s.hyde_enabled, self.s.fused_top_k)
        trace.append(f"gathered {len(candidates)} fused candidates")

        reranked = self.reranker.rerank(question, candidates, top_k=self.s.final_top_k)
        evidence = self._expand_to_parents(reranked)
        trace.append(f"reranked -> {len(evidence)} parent-expanded passages")

        grade_label, grade_reason = "n/a", ""
        if self.s.crag_enabled:
            g = self.crag.grade(question, evidence)
            grade_label, grade_reason = g.grade.value, g.reasoning
            trace.append(f"CRAG grade={grade_label}: {g.reasoning}")
            if g.grade in (Grade.AMBIGUOUS, Grade.INSUFFICIENT):
                # Corrective action: broaden (drop filter, bigger k, force HyDE).
                trace.append("CRAG corrective: broadening retrieval")
                broad = self._gather(queries, None, True, self.s.fused_top_k * 2)
                reranked = self.reranker.rerank(question, broad, top_k=self.s.final_top_k)
                evidence = self._expand_to_parents(reranked)
                g2 = self.crag.grade(question, evidence)
                grade_label, grade_reason = g2.grade.value, g2.reasoning
                trace.append(f"CRAG re-grade={grade_label}")

        return RetrievalResult(
            evidence=evidence,
            grade=grade_label,
            grade_reason=grade_reason,
            queries_used=queries,
            trace=trace,
        )
