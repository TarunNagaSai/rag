"""RAGPipeline — the one object you actually use.

It wires the layers together and exposes three entry points:

  * ingest()  — load -> chunk -> embed -> (optionally) build the knowledge graph
  * ask()     — single question; 'simple' (fast path) or 'agentic' (multi-hop)
  * chat()    — multi-turn; condenses follow-ups into standalone questions using
                retrieval-friendly conversation memory (no giant transcript dumps)

Everything persists to/loads from ``settings.index_dir`` so ingestion is a one-time
cost.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .agent import AgentResult, AgenticRAG
from .chunking import SemanticChunker, chunk_documents
from .config import Settings, get_settings
from .gemini import Gemini
from .generate import Answer, Generator
from .graph import GraphIndex
from .loaders import load_path, load_text
from .retriever import AdvancedRetriever, RetrievalResult
from .schema import Chunk
from .store import HybridStore, build_filter
from schema.schema import ModelSettings


@dataclass
class AskResult:
    answer: Answer
    retrieval: RetrievalResult | None = None
    agent: AgentResult | None = None
    mode: str = "simple"

    def render(self) -> str:
        return self.answer.render()


@dataclass
class ConversationMemory:
    """Retrieval-based memory: keep compact (question, answer) turns and use them to
    rewrite follow-ups into standalone questions instead of injecting full transcripts."""

    turns: list[tuple[str, str]] = field(default_factory=list)

    def add(self, q: str, a: str) -> None:
        self.turns.append((q, a[:500]))

    def recent(self, n: int = 4) -> str:
        return "\n".join(f"Q: {q}\nA: {a}" for q, a in self.turns[-n:])


class RAGPipeline:
    def __init__(self, settings: Settings | None = None, gemini: Gemini | None = None,
                 store: HybridStore | None = None, graph: GraphIndex | None = None):
        self.s = settings or get_settings()
        self.g = gemini or Gemini(self.s)
        self.store = store or HybridStore(self.s, self.g)
        self.graph = graph
        self.retriever = AdvancedRetriever(self.store, self.s, self.g)
        self.generator = Generator(self.g, self.s)
        self.memory = ConversationMemory()

    # ----------------------------------------------------------------- ingest
    def ingest(self, source: str | Path | None = None, *, text: str | None = None,
               build_graph: bool = True, semantic: bool = False,
               save: bool = True) -> int:
        if text is not None:
            docs = load_text(text)
        elif source is not None:
            docs = load_path(source)
        else:
            raise ValueError("Provide either a source path or text=")

        chunks: list[Chunk]
        if semantic:
            chunks = SemanticChunker(self.g, self.s).chunk(docs)
        else:
            chunks = chunk_documents(docs, self.s)

        self.store.add(chunks)

        if build_graph:
            self.graph = GraphIndex(self.s, self.g).build(self.store.chunks)

        if save:
            self.save()
        return len(chunks)

    # ------------------------------------------------------------------- ask
    def _agent(self) -> AgenticRAG:
        return AgenticRAG(self.retriever, self.graph, self.s, self.g)

    def ask(self, question: str, *, mode: str = "agentic",
            sources: list[str] | None = None, where: dict | None = None) -> AskResult:
        filt = build_filter(sources, where)
        if mode == "simple":
            rr = self.retriever.retrieve(question, filt)
            ans = self.generator.answer(question, rr.evidence)
            return AskResult(answer=ans, retrieval=rr, mode="simple")
        # agentic (default): plan/route/act/verify/stop
        ar = self._agent().answer(question)
        return AskResult(answer=ar.answer, agent=ar, mode="agentic")

    # ------------------------------------------------------------------ chat
    def chat(self, question: str, *, mode: str = "agentic") -> AskResult:
        standalone = question
        if self.memory.turns:
            standalone = self.g.generate(
                "Rewrite the follow-up into a standalone question using the "
                "conversation context. Keep all constraints (filters, time windows). "
                "Return only the rewritten question.\n\n"
                f"Conversation:\n{self.memory.recent()}\n\nFollow-up: {question}",
                settings=ModelSettings(
                    system="You resolve coreference and carry constraints across turns.",
                    temperature=0.0,
                ),
            ).strip() or question
        result = self.ask(standalone, mode=mode)
        self.memory.add(question, result.answer.text)
        return result

    # ------------------------------------------------------------ persistence
    def save(self) -> Path:
        d = self.store.save()
        if self.graph is not None:
            self.graph.save()
        return d

    @classmethod
    def load(cls, settings: Settings | None = None, gemini: Gemini | None = None) -> "RAGPipeline":
        s = settings or get_settings()
        g = gemini or Gemini(s)
        if not HybridStore.exists(settings=s):
            raise FileNotFoundError(
                f"No index found at {s.index_dir}. Run ingest first."
            )
        store = HybridStore.load(settings=s, gemini=g)
        graph = None
        if GraphIndex.exists(settings=s):
            graph = GraphIndex.load(store.chunks, settings=s, gemini=g)
        return cls(s, g, store, graph)
