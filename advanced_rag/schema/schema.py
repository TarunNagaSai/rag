"""Core data structures shared across the pipeline.

We deliberately keep these plain and serializable so the whole index can be
persisted to disk and inspected by hand — observability beats magic.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any


def _hash(*parts: str) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(p.encode("utf-8"))
    return h.hexdigest()[:16]


@dataclass
class ModelSettings:
    """Shared generation parameters for every Gemini call."""

    system: str | None = None
    temperature: float = 0.2
    model: str | None = None
    max_output_tokens: int | None = None


@dataclass
class Document:
    """A raw source document before chunking."""

    text: str
    source: str                       # file path / URL / id
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    """A retrievable child chunk. ``parent_text`` is the larger block we swap in
    at generation time (parent-document retrieval)."""

    id: str
    text: str
    source: str
    parent_id: str
    parent_text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def make_id(source: str, text: str, idx: int) -> str:
        return _hash(source, str(idx), text[:64])

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "source": self.source,
            "parent_id": self.parent_id,
            "parent_text": self.parent_text,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Chunk":
        return cls(**d)


@dataclass
class Scored:
    """A chunk paired with a relevance score and provenance about how it was found."""

    chunk: Chunk
    score: float
    how: str = ""  # e.g. "dense", "bm25", "rrf", "rerank"

    @property
    def citation(self) -> str:
        loc = self.chunk.metadata.get("loc")
        return f"{self.chunk.source}" + (f"#{loc}" if loc else "")
