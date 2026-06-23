"""HybridStore — dense vectors + BM25 lexical, fused with Reciprocal Rank Fusion.

This is the heart of retrieval quality. Dense (embedding) search nails *meaning*;
BM25 nails *exact tokens* (IDs, SKUs, acronyms, proper nouns). Neither alone is
enough in production, so we run both and fuse their rankings with RRF — a simple,
hard-to-beat, parameter-light merge.

The store is intentionally a plain in-memory + on-disk structure (numpy matrix +
JSON). No external vector DB to operate. For very large corpora you'd swap the
brute-force search for an ANN index (FAISS/ScaNN) — the interface stays the same.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable

import numpy as np
from rank_bm25 import BM25Okapi

from .config import Settings, get_settings
from .gemini import Gemini
from .schema import Chunk, Scored

_TOKEN = re.compile(r"[A-Za-z0-9_]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]], k: int = 60
) -> dict[str, float]:
    """RRF: score(d) = sum over lists of 1 / (k + rank(d)). Robust to scale
    differences between rankers because it only uses *rank*, not raw score."""
    scores: dict[str, float] = {}
    for lst in ranked_lists:
        for rank, doc_id in enumerate(lst):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return scores


Filter = Callable[[Chunk], bool]


def build_filter(
    sources: list[str] | None = None,
    where: dict[str, Any] | None = None,
) -> Filter | None:
    """Construct a metadata predicate. ``where`` matches metadata key==value
    (or membership when the value is a list)."""
    if not sources and not where:
        return None

    def _f(c: Chunk) -> bool:
        if sources and not any(s in c.source for s in sources):
            return False
        if where:
            for key, val in where.items():
                cv = c.metadata.get(key)
                if isinstance(val, (list, tuple, set)):
                    if cv not in val:
                        return False
                elif cv != val:
                    return False
        return True

    return _f


class HybridStore:
    def __init__(self, settings: Settings | None = None, gemini: Gemini | None = None):
        self.s = settings or get_settings()
        self.g = gemini or Gemini(self.s)
        self.chunks: list[Chunk] = []
        self.embeddings: np.ndarray = np.zeros((0, self.s.embed_dim), dtype=np.float32)
        self._id_to_idx: dict[str, int] = {}
        self._bm25: BM25Okapi | None = None
        self._corpus_tokens: list[list[str]] = []

    # ----------------------------------------------------------------- build
    def add(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        new = self.g.embed([c.text for c in chunks])
        self.embeddings = (
            new if self.embeddings.size == 0 else np.vstack([self.embeddings, new])
        )
        for c in chunks:
            self._id_to_idx[c.id] = len(self.chunks)
            self.chunks.append(c)
        self._rebuild_bm25()

    def _rebuild_bm25(self) -> None:
        self._corpus_tokens = [_tokenize(c.text) for c in self.chunks]
        self._bm25 = BM25Okapi(self._corpus_tokens) if self._corpus_tokens else None

    # ---------------------------------------------------------------- search
    def dense_search(self, query_vec: np.ndarray, top_k: int,
                     allowed: set[int] | None = None) -> list[tuple[str, float]]:
        if self.embeddings.size == 0:
            return []
        sims = self.embeddings @ query_vec  # cosine (all normalized)
        if allowed is not None:
            mask = np.full(sims.shape, -np.inf, dtype=np.float32)
            idx = np.fromiter(allowed, dtype=np.int64, count=len(allowed))
            mask[idx] = sims[idx]
            sims = mask
        n = min(top_k, np.count_nonzero(np.isfinite(sims)))
        if n <= 0:
            return []
        top = np.argpartition(-sims, n - 1)[:n]
        top = top[np.argsort(-sims[top])]
        return [(self.chunks[i].id, float(sims[i])) for i in top]

    def lexical_search(self, query: str, top_k: int,
                       allowed: set[int] | None = None) -> list[tuple[str, float]]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        order = np.argsort(-scores)
        out: list[tuple[str, float]] = []
        for i in order:
            if allowed is not None and i not in allowed:
                continue
            if scores[i] <= 0:
                break
            out.append((self.chunks[i].id, float(scores[i])))
            if len(out) >= top_k:
                break
        return out

    def hybrid_search(
        self,
        query: str,
        query_vec: np.ndarray,
        *,
        top_k: int | None = None,
        filt: Filter | None = None,
    ) -> list[Scored]:
        """Run dense + BM25, fuse with RRF, return fused top_k as Scored chunks."""
        top_k = top_k or self.s.fused_top_k
        allowed: set[int] | None = None
        if filt is not None:
            allowed = {i for i, c in enumerate(self.chunks) if filt(c)}
            if not allowed:
                return []

        dense = self.dense_search(query_vec, self.s.dense_top_k, allowed)
        lexical = self.lexical_search(query, self.s.lexical_top_k, allowed)
        fused = reciprocal_rank_fusion(
            [[d for d, _ in dense], [l for l, _ in lexical]], k=self.s.rrf_k
        )
        ranked = sorted(fused.items(), key=lambda kv: -kv[1])[:top_k]
        return [
            Scored(chunk=self.chunks[self._id_to_idx[cid]], score=score, how="rrf")
            for cid, score in ranked
        ]

    # --------------------------------------------------------- parent lookup
    def parent_text_for(self, chunk: Chunk) -> str:
        return chunk.parent_text

    def get(self, chunk_id: str) -> Chunk | None:
        idx = self._id_to_idx.get(chunk_id)
        return self.chunks[idx] if idx is not None else None

    # ------------------------------------------------------------ persistence
    def save(self, path: str | Path | None = None) -> Path:
        d = Path(path) if path else self.s.index_dir
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "embeddings.npy", self.embeddings)
        with open(d / "chunks.jsonl", "w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
        (d / "meta.json").write_text(
            json.dumps({"embed_dim": self.s.embed_dim, "count": len(self.chunks)})
        )
        return d

    @classmethod
    def load(cls, path: str | Path | None = None,
             settings: Settings | None = None, gemini: Gemini | None = None) -> "HybridStore":
        s = settings or get_settings()
        d = Path(path) if path else s.index_dir
        store = cls(s, gemini)
        store.embeddings = np.load(d / "embeddings.npy")
        with open(d / "chunks.jsonl", encoding="utf-8") as f:
            store.chunks = [Chunk.from_dict(json.loads(line)) for line in f]
        store._id_to_idx = {c.id: i for i, c in enumerate(store.chunks)}
        store._rebuild_bm25()
        return store

    @staticmethod
    def exists(path: str | Path | None = None, settings: Settings | None = None) -> bool:
        s = settings or get_settings()
        d = Path(path) if path else s.index_dir
        return (d / "chunks.jsonl").exists() and (d / "embeddings.npy").exists()
