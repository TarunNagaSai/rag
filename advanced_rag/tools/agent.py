"""Agentic RAG — plan → route → act → verify → stop.

Single-shot top-k retrieval misses multi-hop questions ("who reported the bug that
delayed the release that customer X depends on?"). An agent decomposes the question
into sub-goals, routes each to the *right* retriever, gathers evidence with
provenance, verifies coverage, and stops on budget — then synthesizes one grounded
answer with per-claim citations.

Routing menu:
  * hybrid       — facts, names, dates (dense + BM25 + rerank + CRAG)
  * graph_local  — relationships / joins around specific entities (k-hop traversal)
  * graph_global — big-picture, thematic questions (community summaries)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from pydantic import BaseModel, Field

from advanced_rag.core.config import Settings, get_settings
from advanced_rag.tools.crag import CRAG, Grade
from advanced_rag.llm.gemini import Gemini
from advanced_rag.llm.generate import Answer, Generator
from advanced_rag.tools.graph import GraphIndex
from advanced_rag.tools.retriever import AdvancedRetriever
from advanced_rag.schema.schema import ModelSettings, Scored


class Tool(str, Enum):
    HYBRID = "hybrid"
    GRAPH_LOCAL = "graph_local"
    GRAPH_GLOBAL = "graph_global"


class _SubQ(BaseModel):
    question: str = Field(description="A concrete, self-contained sub-question")
    tool: Tool = Field(description="Best retriever for this sub-question")
    reason: str = Field(default="", description="Why this tool")


class _Plan(BaseModel):
    needs_decomposition: bool = Field(description="False for simple single-hop lookups")
    subquestions: list[_SubQ]


@dataclass
class Step:
    question: str
    tool: str
    grade: str
    n_evidence: int


@dataclass
class AgentResult:
    answer: Answer
    plan: list[str] = field(default_factory=list)
    steps: list[Step] = field(default_factory=list)
    trace: list[str] = field(default_factory=list)


class AgenticRAG:
    def __init__(self, retriever: AdvancedRetriever, graph: GraphIndex | None = None,
                 settings: Settings | None = None, gemini: Gemini | None = None):
        self.s = settings or get_settings()
        self.g = gemini or retriever.g
        self.retriever = retriever
        self.graph = graph
        self.generator = Generator(self.g, self.s)
        self.crag = CRAG(self.g, self.s)

    # ------------------------------------------------------------------- plan
    def _plan(self, question: str) -> _Plan:
        tools = "hybrid (facts/names/dates), graph_local (relationships between entities)"
        if self.graph and self.graph.communities:
            tools += ", graph_global (big-picture/thematic)"
        return self.g.generate_structured(
            prompt=(
                "Plan how to answer the QUESTION with a retrieval system.\n"
                f"Available tools: {tools}.\n"
                "If it is a simple single-hop lookup, set needs_decomposition=false and "
                "return one sub-question. Otherwise break it into the minimal set of "
                f"sub-questions (max {self.s.agent_max_subquestions}), each routed to the "
                "best tool.\n\n"
                f"QUESTION: {question}"
            ),
            schema=_Plan,
            settings=ModelSettings(
                system="You are a retrieval planner. Decompose only when it genuinely helps.",
                temperature=0.1,
            ),
        )

    # -------------------------------------------------------------------- act
    def _act(self, sub: _SubQ) -> list[Scored]:
        if sub.tool == Tool.GRAPH_LOCAL and self.graph:
            ev = self.graph.local_search(sub.question)
            if ev:
                return ev
            return self.retriever.retrieve(sub.question).evidence  # fallback
        if sub.tool == Tool.GRAPH_GLOBAL and self.graph:
            summaries = self.graph.global_search(sub.question)
            if summaries:
                return [
                    Scored(chunk=_summary_chunk(text, i), score=1.0, how="graph_global")
                    for i, text in enumerate(summaries)
                ]
        return self.retriever.retrieve(sub.question).evidence

    # ----------------------------------------------------------------- public
    def answer(self, question: str) -> AgentResult:
        trace: list[str] = []
        plan = self._plan(question)
        subs = plan.subquestions[: self.s.agent_max_subquestions] or [
            _SubQ(question=question, tool=Tool.HYBRID)
        ]
        trace.append(f"plan: {len(subs)} sub-question(s), decompose={plan.needs_decomposition}")

        all_evidence: list[Scored] = []
        seen_ids: set[str] = set()
        steps: list[Step] = []

        for i, sub in enumerate(subs):
            if i >= self.s.agent_max_hops:
                trace.append("hop budget reached; stopping")
                break
            evidence = self._act(sub)
            g = self.crag.grade(sub.question, evidence)
            if g.grade in (Grade.AMBIGUOUS, Grade.INSUFFICIENT) and sub.tool != Tool.HYBRID:
                # verify failed -> switch tool to robust hybrid retrieval
                trace.append(f"[{sub.tool.value}] weak; switching to hybrid for: {sub.question}")
                evidence = self.retriever.retrieve(sub.question).evidence
                g = self.crag.grade(sub.question, evidence)
            steps.append(Step(sub.question, sub.tool.value, g.grade.value, len(evidence)))
            trace.append(f"[{sub.tool.value}] '{sub.question}' -> {len(evidence)} ev, grade={g.grade.value}")
            for e in evidence:
                if e.chunk.id not in seen_ids:
                    seen_ids.add(e.chunk.id)
                    all_evidence.append(e)

        # Synthesize with the stronger model for multi-hop questions.
        model = self.s.reasoning_model if plan.needs_decomposition else self.s.gen_model
        ans = self.generator.answer(question, all_evidence, model=model)
        return AgentResult(answer=ans, plan=[s.question for s in subs], steps=steps, trace=trace)


def _summary_chunk(text: str, i: int):
    from advanced_rag.schema.schema import Chunk

    return Chunk(
        id=f"community-{i}", text=text, source="graph://community",
        parent_id=f"community-{i}", parent_text=text, metadata={"loc": f"community-{i}"},
    )
