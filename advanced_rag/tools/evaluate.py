"""Evaluation — "don't skimp on this." You can't improve what you don't measure.

Two families of metrics, matching the guide:

  Retrieval quality (cheap, deterministic):
    * Recall@k — did the right source make it into the top-k?
    * MRR      — how high did the first right source rank?

  Answer quality (LLM-as-judge):
    * Groundedness/faithfulness — is every claim supported by the evidence?
    * Answer relevance         — does it actually answer the question?

Dataset format (JSON list):
    [{"question": "...", "expected_sources": ["report.md"], "ideal_answer": "..."}]
``expected_sources`` are substrings matched against chunk sources; ``ideal_answer``
is optional context for the relevance judge.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean

from pydantic import BaseModel, Field

from advanced_rag.core.config import Settings, get_settings
from advanced_rag.llm.gemini import Gemini
from advanced_rag.agents.pipeline import RAGPipeline
from advanced_rag.schema.schema import ModelSettings, Scored


class _Judge(BaseModel):
    score: float = Field(description="Score from 0.0 to 1.0")
    reasoning: str = Field(description="Brief justification")


@dataclass
class CaseResult:
    question: str
    recall_at_k: float
    reciprocal_rank: float
    groundedness: float
    answer_relevance: float
    answer: str


@dataclass
class EvalReport:
    cases: list[CaseResult]

    def summary(self) -> dict[str, float]:
        if not self.cases:
            return {}
        return {
            "recall@k": round(mean(c.recall_at_k for c in self.cases), 3),
            "mrr": round(mean(c.reciprocal_rank for c in self.cases), 3),
            "groundedness": round(mean(c.groundedness for c in self.cases), 3),
            "answer_relevance": round(mean(c.answer_relevance for c in self.cases), 3),
        }


class Evaluator:
    def __init__(self, pipeline: RAGPipeline, settings: Settings | None = None):
        self.s = settings or get_settings()
        self.p = pipeline
        self.g = pipeline.g

    # ----------------------------------------------------- retrieval metrics
    @staticmethod
    def _retrieval_metrics(evidence: list[Scored], expected: list[str]) -> tuple[float, float]:
        if not expected:
            return (0.0, 0.0)
        hit, rr = 0.0, 0.0
        for rank, e in enumerate(evidence, start=1):
            if any(exp.lower() in e.chunk.source.lower() for exp in expected):
                hit = 1.0
                rr = 1.0 / rank
                break
        return hit, rr

    # ------------------------------------------------------- LLM-as-judge
    def _judge(self, instruction: str) -> float:
        out = self.g.generate_structured(
            prompt=instruction, schema=_Judge,
            settings=ModelSettings(
                system="You are a strict, calibrated RAG evaluator. Output only the score and reasoning.",
                temperature=0.0,
            ),
        )
        return max(0.0, min(1.0, out.score))

    def groundedness(self, answer: str, evidence: list[Scored]) -> float:
        if not answer.strip() or answer.lower().startswith("i don't know"):
            return 1.0 if not evidence else 0.0
        ctx = "\n\n".join(f"[{i+1}] {e.chunk.parent_text or e.chunk.text}"
                          for i, e in enumerate(evidence))
        return self._judge(
            "Rate how fully the ANSWER is supported by the SOURCES (faithfulness). "
            "1.0 = every claim is supported; 0.0 = mostly unsupported/hallucinated.\n\n"
            f"SOURCES:\n{ctx}\n\nANSWER:\n{answer}"
        )

    def answer_relevance(self, question: str, answer: str, ideal: str = "") -> float:
        extra = f"\n\nReference answer (for guidance):\n{ideal}" if ideal else ""
        return self._judge(
            "Rate how well the ANSWER addresses the QUESTION (relevance/completeness). "
            "1.0 = directly and completely answers; 0.0 = off-topic or empty.\n\n"
            f"QUESTION: {question}\n\nANSWER:\n{answer}{extra}"
        )

    # ------------------------------------------------------------------ run
    def run(self, dataset: list[dict], *, mode: str = "agentic") -> EvalReport:
        cases: list[CaseResult] = []
        for item in dataset:
            q = item["question"]
            expected = item.get("expected_sources", [])
            ideal = item.get("ideal_answer", "")
            # Retrieval metrics use the deterministic retriever path.
            rr_result = self.p.retriever.retrieve(q)
            recall, mrr = self._retrieval_metrics(rr_result.evidence, expected)
            # Answer + answer-quality metrics use the requested mode.
            res = self.p.ask(q, mode=mode)
            grounded = self.groundedness(res.answer.text, rr_result.evidence)
            relevance = self.answer_relevance(q, res.answer.text, ideal)
            cases.append(CaseResult(
                question=q, recall_at_k=recall, reciprocal_rank=mrr,
                groundedness=grounded, answer_relevance=relevance,
                answer=res.answer.text,
            ))
        return EvalReport(cases=cases)


def load_dataset(path: str | Path) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def report_to_dict(report: EvalReport) -> dict:
    return {"summary": report.summary(), "cases": [asdict(c) for c in report.cases]}
