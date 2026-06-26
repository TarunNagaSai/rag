"""Evaluation harness stub."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class EvalCase:
    question: str
    expected: str


@dataclass
class EvalReport:
    cases: list[dict]

    def summary(self) -> dict[str, float]:
        return {"cases": float(len(self.cases))}


def load_dataset(path: str) -> list[EvalCase]:
    data = json.loads(Path(path).read_text())
    return [EvalCase(question=d["question"], expected=d.get("expected", "")) for d in data]


def report_to_dict(report: EvalReport) -> dict:
    return {"cases": report.cases, "summary": report.summary()}


class Evaluator:
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def run(self, dataset: list[EvalCase], mode: str = "simple") -> EvalReport:
        results = []
        for case in dataset:
            res = self.pipeline.ask(case.question, mode=mode)
            results.append({"question": case.question, "answer": res.answer.text})
        return EvalReport(cases=results)
