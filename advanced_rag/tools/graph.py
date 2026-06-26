"""GraphIndex stub — placeholder for future GraphRAG implementation."""

from __future__ import annotations

from advanced_rag.core.config import Settings, get_settings
from advanced_rag.llm.gemini import Gemini
from advanced_rag.schema.schema import Chunk


class GraphIndex:
    def __init__(self, settings: Settings | None = None, gemini: Gemini | None = None):
        self.s = settings or get_settings()
        self.g = gemini or Gemini(self.s)
        self.graph = _FakeGraph()
        self.communities: list = []

    def build(self, chunks: list[Chunk]) -> "GraphIndex":
        return self

    def save(self) -> None:
        pass

    @staticmethod
    def exists(settings: Settings | None = None) -> bool:
        return False

    @classmethod
    def load(cls, chunks, settings=None, gemini=None) -> "GraphIndex":
        return cls(settings, gemini)


class _FakeGraph:
    def number_of_nodes(self) -> int:
        return 0

    def number_of_edges(self) -> int:
        return 0
