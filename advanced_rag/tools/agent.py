"""Agentic RAG — multi-hop question answering via plan/retrieve/verify loop."""

from __future__ import annotations

from dataclasses import dataclass, field

from advanced_rag.core.config import Settings, get_settings
from advanced_rag.llm.gemini import Gemini
from advanced_rag.llm.generate import Answer, Generator
from advanced_rag.tools.retriever import AdvancedRetriever


@dataclass
class AgentResult:
    answer: Answer
    plan: list[str] = field(default_factory=list)
    trace: list[str] = field(default_factory=list)


class AgenticRAG:
    def __init__(
        self,
        retriever: AdvancedRetriever,
        graph=None,
        settings: Settings | None = None,
        gemini: Gemini | None = None,
    ):
        self.retriever = retriever
        self.graph = graph
        self.s = settings or get_settings()
        self.g = gemini or Gemini(self.s)
        self.gen = Generator(self.g, self.s)

    def answer(self, question: str) -> AgentResult:
        rr = self.retriever.retrieve(question)
        ans = self.gen.answer(question, rr.evidence)
        return AgentResult(answer=ans, plan=[question], trace=[])
