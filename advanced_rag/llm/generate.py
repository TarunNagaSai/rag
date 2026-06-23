"""Grounded generation — turn evidence into an answer that *can't* drift from sources.

Grounding is enforced three ways:
  1. A strict system prompt: answer only from sources, else say "I don't know".
  2. Numbered sources + a requirement to cite [n] next to every claim.
  3. The evidence we feed is the parent block (full context), not a fragment.

We return the citation map alongside the text so the UI / evaluator can show
exactly which source backed which claim.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass, field

from .config import Settings, get_settings
from .gemini import Gemini
from .schema import Scored
from schema.schema import ModelSettings

GROUNDING_SYSTEM = (
    "You are a careful question-answering assistant for a retrieval system. "
    "Answer ONLY using the numbered sources provided. Cite the source id in square "
    "brackets like [1] immediately after each claim it supports. If the sources do "
    "not contain enough information to answer, reply exactly: "
    "\"I don't know based on the available sources.\" "
    "Never invent facts, numbers, names, or citations."
)


@dataclass
class Answer:
    text: str
    sources: list[tuple[int, str]] = field(default_factory=list)  # (n, citation)
    cited: list[int] = field(default_factory=list)

    def render(self) -> str:
        lines = [self.text.strip(), ""]
        if self.sources:
            lines.append("Sources:")
            for n, cite in self.sources:
                marker = "•" if n in self.cited else " "
                lines.append(f"  {marker} [{n}] {cite}")
        return "\n".join(lines)


class Generator:
    def __init__(self, gemini: Gemini | None = None, settings: Settings | None = None):
        self.s = settings or get_settings()
        self.g = gemini or Gemini(self.s)

    def answer(self, question: str, evidence: list[Scored],
               *, model: str | None = None) -> Answer:
        if not evidence:
            return Answer(text="I don't know based on the available sources.")

        blocks, sources = [], []
        for i, e in enumerate(evidence, start=1):
            # Parent-document retrieval: hand the model the larger parent block.
            text = e.chunk.parent_text or e.chunk.text
            blocks.append(f"[{i}] (source: {e.citation})\n{text}")
            sources.append((i, e.citation))

        prompt = (
            f"QUESTION: {question}\n\n"
            f"SOURCES:\n" + "\n\n".join(blocks) + "\n\n"
            "Write a clear, well-structured answer grounded in the sources above, "
            "citing [n] after each supported claim."
        )
        text = self.g.generate(
            prompt, settings=ModelSettings(system=GROUNDING_SYSTEM, temperature=0.1, model=model)
        )
        cited = sorted({int(m) for m in re.findall(r"\[(\d+)\]", text)})
        return Answer(text=text, sources=sources, cited=cited)

    def answer_stream(
        self, question: str, evidence: list[Scored], *, model: str | None = None
    ) -> tuple[list[tuple[int, str]], Iterator[str]]:
        """Return (sources_list, text_chunk_iterator) without blocking on the model."""
        if not evidence:
            return [], iter(["I don't know based on the available sources."])

        blocks, sources = [], []
        for i, e in enumerate(evidence, start=1):
            text = e.chunk.parent_text or e.chunk.text
            blocks.append(f"[{i}] (source: {e.citation})\n{text}")
            sources.append((i, e.citation))

        prompt = (
            f"QUESTION: {question}\n\n"
            f"SOURCES:\n" + "\n\n".join(blocks) + "\n\n"
            "Write a clear, well-structured answer grounded in the sources above, "
            "citing [n] after each supported claim."
        )
        stream = self.g.generate_content_stream(
            prompt,
            settings=ModelSettings(system=GROUNDING_SYSTEM, temperature=0.1, model=model),
        )
        return sources, stream
