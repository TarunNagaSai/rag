"""Quickstart: build an index from the sample corpus and ask a multi-hop question.

Run:  python examples/quickstart.py
(Requires GOOGLE_API_KEY in your environment or .env)
"""

from advanced_rag import RAGPipeline


def main() -> None:
    pipe = RAGPipeline()

    # 1) Ingest: load -> chunk (parent/child) -> embed -> build knowledge graph.
    n = pipe.ingest("data/sample", build_graph=True)
    print(f"Indexed {n} chunks.\n")

    # 2) A multi-hop question that needs cross-document reasoning + the graph.
    q = "Which enterprise customers renewed in Q2 2025 and also opened an SSO ticket?"
    res = pipe.ask(q, mode="agentic")
    print(res.render())

    # 3) Inspect the agent's plan/trace (great for learning what happened).
    if res.agent:
        print("\n--- Agent plan ---")
        for step in res.agent.steps:
            print(f"  [{step.tool}] {step.question}  (grade={step.grade}, ev={step.n_evidence})")

    # 4) Follow-up using conversation memory (coreference is resolved automatically).
    print("\n--- Follow-up ---")
    follow = pipe.chat("Who manages those accounts?")
    print(follow.render())


if __name__ == "__main__":
    main()
