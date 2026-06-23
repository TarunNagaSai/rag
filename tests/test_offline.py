"""Offline smoke tests — exercise everything that does NOT need the Gemini API.

A FakeGemini supplies deterministic embeddings so we can test chunking, the hybrid
store (dense + BM25 + RRF), filters, and persistence without network or a key.

Run:  python -m pytest tests/ -q     (or)     python tests/test_offline.py
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from advanced_rag.chunking import approx_tokens, chunk_documents, split_sentences
from advanced_rag.config import get_settings
from advanced_rag.loaders import load_text
from advanced_rag.schema import Chunk
from advanced_rag.store import HybridStore, build_filter, reciprocal_rank_fusion


class FakeGemini:
    """Deterministic hash-based embeddings; no network."""

    def __init__(self, settings):
        self.s = settings

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


def test_token_and_sentences():
    assert approx_tokens("a" * 40) == 10
    sents = split_sentences("Hello world. This is a test! Is it? Yes.")
    assert len(sents) == 4


def test_chunking_parent_child():
    text = " ".join(f"Sentence number {i} about topic {i % 3}." for i in range(200))
    chunks = chunk_documents(load_text(text, "doc1"))
    assert chunks, "should produce chunks"
    # Multiple children should share parents (parent-document retrieval).
    parents = {c.parent_id for c in chunks}
    assert len(parents) < len(chunks)
    for c in chunks:
        assert c.parent_text and c.text in c.parent_text or len(c.text) <= len(c.parent_text)


def test_rrf():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "c", "a"]], k=60)
    # 'b' is rank2+rank1, 'a' is rank1+rank3 -> close; both high
    assert set(fused) == {"a", "b", "c"}
    assert fused["b"] > fused["c"]


def test_hybrid_store_and_filter():
    s = get_settings()
    store = HybridStore(s, gemini=FakeGemini(s))
    chunks = [
        Chunk(id="1", text="The SSO ticket SSO-4412 was about Azure AD SAML.",
              source="customers.md", parent_id="p1", parent_text="parent one",
              metadata={"region": "EMEA"}),
        Chunk(id="2", text="Billing invoice currency mismatch in APAC.",
              source="customers.md", parent_id="p2", parent_text="parent two",
              metadata={"region": "APAC"}),
        Chunk(id="3", text="Platinum plan includes SCIM provisioning.",
              source="product.md", parent_id="p3", parent_text="parent three",
              metadata={"region": "ALL"}),
    ]
    store.add(chunks)

    # Lexical recall of a rare token (BM25 strength).
    qvec = store.g.embed_query("SSO-4412")
    hits = store.hybrid_search("SSO-4412 Azure AD", qvec)
    assert hits and hits[0].chunk.id == "1"

    # Metadata filter restricts the candidate set.
    filt = build_filter(where={"region": "APAC"})
    qvec2 = store.g.embed_query("billing")
    hits2 = store.hybrid_search("billing invoice", qvec2, filt=filt)
    assert all(h.chunk.metadata["region"] == "APAC" for h in hits2)


def test_persistence(tmp_path=None):
    import tempfile

    d = Path(tempfile.mkdtemp())
    s = get_settings()
    store = HybridStore(s, gemini=FakeGemini(s))
    store.add([Chunk(id="1", text="hello world", source="a.md",
                     parent_id="p1", parent_text="hello world parent")])
    store.save(d)
    loaded = HybridStore.load(d, settings=s, gemini=FakeGemini(s))
    assert len(loaded.chunks) == 1
    assert loaded.chunks[0].text == "hello world"


def _run_all():
    fns = [v for k, v in globals().items() if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nAll {len(fns)} offline tests passed.")


if __name__ == "__main__":
    _run_all()
