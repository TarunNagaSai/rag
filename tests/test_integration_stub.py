"""Integration test with a STUB Gemini — exercises the full pipeline wiring
(ingest -> graph build -> retriever -> agent -> generate) with zero network.

This proves the modules compose correctly. Swapping in the real Gemini (valid key)
is the only change needed for live use.

Run: python tests/test_integration_stub.py
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from advanced_rag.config import get_settings
from advanced_rag.pipeline import RAGPipeline


class StubGemini:
    """Deterministic embeddings + schema-aware canned structured outputs."""

    def __init__(self, settings):
        self.s = settings

    # ---- embeddings ----
    def _vec(self, text: str) -> np.ndarray:
        h = hashlib.sha256(text.encode()).digest()
        rng = np.random.default_rng(int.from_bytes(h[:8], "little"))
        v = rng.standard_normal(self.s.embed_dim).astype(np.float32)
        return v / (np.linalg.norm(v) or 1.0)

    def embed(self, texts, task_type="RETRIEVAL_DOCUMENT"):
        if not texts:
            return np.zeros((0, self.s.embed_dim), dtype=np.float32)
        return np.vstack([self._vec(t) for t in texts])

    def embed_query(self, text):
        return self._vec(text)

    # ---- text generation ----
    def generate(self, prompt, system=None, temperature=0.2, model=None,
                 max_output_tokens=None):
        if "hypothetical" in (system or "").lower() or "HyDE" in prompt:
            return "Northwind Traders renewed and opened an SSO ticket."
        if "standalone question" in prompt.lower():
            return "Who manages the accounts that renewed and opened SSO tickets?"
        if "community summaries" in (system or "").lower():
            return "Cluster about customers, renewals and SSO tickets."
        # default: a grounded-looking answer that cites [1]
        return "Northwind Traders and Umbrella Health renewed and opened SSO tickets [1]."

    # ---- structured generation ----
    def generate_structured(self, prompt, schema, system=None, temperature=0.0, model=None):
        name = schema.__name__
        if name == "_Rewrites":
            return schema(queries=["renewed customers SSO ticket", "Q2 renewals support SSO"])
        if name == "_Ranking":
            # rank candidates by their index appearing in prompt
            import re
            idxs = sorted({int(i) for i in re.findall(r"\[(\d+)\]", prompt)})
            from advanced_rag.rerank import _RankItem
            return schema(rankings=[_RankItem(index=i, relevance=1.0 - 0.05 * i) for i in idxs])
        if name == "_GradeOut":
            from advanced_rag.crag import Grade
            return schema(grade=Grade.CORRECT, reasoning="evidence supports an answer", missing="")
        if name == "_Extraction":
            from advanced_rag.graph import _Entity, _Relation
            return schema(
                entities=[_Entity(name="Northwind", type="ORG"),
                          _Entity(name="Priya Raman", type="PERSON")],
                relations=[_Relation(source="Priya Raman", relation="manages", target="Northwind")],
            )
        if name == "_QueryEntities":
            return schema(entities=["Northwind", "Priya Raman"])
        if name == "_Plan":
            from advanced_rag.agent import _SubQ, Tool
            return schema(needs_decomposition=True, subquestions=[
                _SubQ(question="Which customers renewed in Q2 2025?", tool=Tool.HYBRID),
                _SubQ(question="Which of them opened an SSO ticket?", tool=Tool.HYBRID),
            ])
        if name == "_Judge":
            return schema(score=0.9, reasoning="looks supported and relevant")
        raise AssertionError(f"Unhandled schema {name}")


def main():
    s = get_settings()
    stub = StubGemini(s)
    pipe = RAGPipeline(settings=s, gemini=stub)

    n = pipe.ingest("data/sample", build_graph=True, save=False)
    assert n > 0, "no chunks indexed"
    assert pipe.graph and pipe.graph.graph.number_of_nodes() > 0, "graph not built"
    print(f"ingested {n} chunks; graph nodes={pipe.graph.graph.number_of_nodes()}, "
          f"edges={pipe.graph.graph.number_of_edges()}, communities={len(pipe.graph.communities)}")

    # simple mode
    res = pipe.ask("Which enterprise customers renewed and opened an SSO ticket?", mode="simple")
    assert res.answer.text and res.retrieval is not None
    print("simple mode OK; evidence:", len(res.retrieval.evidence), "grade:", res.retrieval.grade)

    # agentic mode
    res2 = pipe.ask("Which enterprise customers renewed in Q2 and opened an SSO ticket?", mode="agentic")
    assert res2.agent is not None and res2.agent.steps
    print("agentic mode OK; steps:", [(st.tool, st.grade) for st in res2.agent.steps])

    # graph local search path
    ev = pipe.graph.local_search("Who manages Northwind?")
    print("graph local_search returned", len(ev), "passages")

    # conversation memory
    follow = pipe.chat("Who manages those accounts?", mode="simple")
    assert follow.answer.text
    print("chat/memory OK")

    # evaluation harness
    from advanced_rag.evaluate import Evaluator, load_dataset
    report = Evaluator(pipe).run(load_dataset("data/eval.json"), mode="simple")
    print("eval summary:", report.summary())

    print("\nALL INTEGRATION CHECKS PASSED")


if __name__ == "__main__":
    main()
