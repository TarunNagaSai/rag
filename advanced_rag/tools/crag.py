"""Corrective RAG (CRAG) — a lightweight self-check *before* generation.

The idea: don't blindly trust the retrieved set. Grade it. If the evidence looks
strong, proceed. If it's weak or partial, take a corrective action (broaden the
search, relax filters, or fall back) instead of letting the model hallucinate to
fill the gap. This single feedback loop measurably cuts hallucinations.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from .config import Settings, get_settings
from .gemini import Gemini
from .schema import Scored


class Grade(str, Enum):
    CORRECT = "correct"          # evidence clearly supports an answer
    AMBIGUOUS = "ambiguous"      # partial / needs broadening
    INSUFFICIENT = "insufficient"  # evidence does not support an answer


class _GradeOut(BaseModel):
    grade: Grade
    reasoning: str = Field(description="One sentence justification")
    missing: str = Field(default="", description="What evidence is missing, if any")


class CRAG:
    def __init__(self, gemini: Gemini | None = None, settings: Settings | None = None):
        self.s = settings or get_settings()
        self.g = gemini or Gemini(self.s)

    def grade(self, question: str, evidence: list[Scored]) -> _GradeOut:
        if not evidence:
            return _GradeOut(grade=Grade.INSUFFICIENT, reasoning="No evidence retrieved.",
                             missing="Any relevant source.")
        passages = "\n\n".join(
            f"[{i}] {e.chunk.text}" for i, e in enumerate(evidence)
        )
        return self.g.generate_structured(
            prompt=(
                "Decide whether the retrieved passages contain enough evidence to "
                "answer the question faithfully.\n"
                "- 'correct': the passages clearly support a complete answer.\n"
                "- 'ambiguous': they are partially relevant; more retrieval would help.\n"
                "- 'insufficient': they do not support an answer.\n\n"
                f"QUESTION: {question}\n\nPASSAGES:\n{passages}"
            ),
            schema=_GradeOut,
            system="You are a strict retrieval-quality grader for a RAG system.",
            temperature=0.0,
        )
