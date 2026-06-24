"""Chunking — turn raw documents into retrievable child chunks that carry a larger
parent block for context.

Two strategies live here:

  * ``chunk_documents``  — structure-aware recursive splitting (the sane default:
    deterministic, free, respects paragraph/sentence boundaries).
  * ``SemanticChunker``  — embedding-based splitting that cuts where the topic
    actually shifts. Costs embeddings up front but yields cleaner boundaries on
    messy prose. Opt in when structure is poor.

Both produce parent/child pairs so downstream retrieval can match on small,
precise children but feed the model the surrounding parent block.
"""

from __future__ import annotations

import re
from socket import close

import numpy as np

from advanced_rag.core.config import Settings, get_settings
from advanced_rag.llm.gemini import Gemini
from advanced_rag.schema.schema import Chunk, Document

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
_PARA_SPLIT = re.compile(r"\n\s*\n")


def approx_tokens(text: str) -> int:
    """Cheap, dependency-free token estimate (~4 chars/token)."""
    return max(1, len(text) // 4)


def split_sentences(text: str) -> list[str]:
    sents: list[str] = []
    for para in _PARA_SPLIT.split(text):
        para = para.strip()
        if not para:
            continue
        sents.extend(s.strip() for s in _SENT_SPLIT.split(para) if s.strip())
    return sents


def _pack(units: list[str], max_tokens: int, overlap_tokens: int) -> list[str]:
    """Greedily pack text units into windows of ~max_tokens with overlap."""
    out: list[str] = []
    cur: list[str] = []
    cur_tok = 0
    for u in units:
        ut = approx_tokens(u)
        if cur and cur_tok + ut > max_tokens:
            out.append(" ".join(cur))
            # carry trailing units as overlap
            back, btok = [], 0
            for prev in reversed(cur):
                ptok = approx_tokens(prev)
                if btok + ptok > overlap_tokens:
                    break
                back.insert(0, prev)
                btok += ptok
            cur, cur_tok = list(back), btok
        cur.append(u)
        cur_tok += ut
    if cur:
        out.append(" ".join(cur))
    return out


def chunk_documents(
    documents: list[Document], settings: Settings | None = None
) -> list[Chunk]:
    s = settings or get_settings()
    chunks: list[Chunk] = []
    for doc in documents:
        sentences = split_sentences(doc.text)
        if not sentences:
            continue
        # 1) Build parent blocks (large, context-preserving).
        parents = _pack(sentences, s.parent_chunk_tokens, overlap_tokens=0)
        for pi, parent_text in enumerate(parents):
            parent_id = Chunk.make_id(doc.source, parent_text, pi)
            # 2) Split each parent into precise children with overlap.
            child_sents = split_sentences(parent_text)
            children = _pack(child_sents, s.chunk_tokens, s.chunk_overlap_tokens)
            for ci, child_text in enumerate(children):
                meta = dict(doc.metadata)
                meta["loc"] = f"p{pi}.c{ci}"
                chunks.append(
                    Chunk(
                        id=Chunk.make_id(doc.source, child_text, pi * 1000 + ci),
                        text=child_text,
                        source=doc.source,
                        parent_id=parent_id,
                        parent_text=parent_text,
                        metadata=meta,
                    )
                )
    return chunks


class SemanticChunker:
    """Split on topic shifts detected via embedding distance between sentences.

    Algorithm: embed each sentence, measure cosine distance to a small rolling
    window, and start a new chunk wherever the distance exceeds a percentile
    threshold (the classic 'semantic chunking' breakpoint heuristic).
    """

    def __init__(self, gemini: Gemini | None = None, settings: Settings | None = None,
                 percentile: float = 90.0):
        self.s = settings or get_settings()
        self.g = gemini or Gemini(self.s)
        self.percentile = percentile

    def chunk(self, documents: list[Document]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for doc in documents:
            sents = split_sentences(doc.text)
            if len(sents) < 4:
                chunks.extend(chunk_documents([doc], self.s))
                continue
            emb = self.g.embed(sents)  # already normalized
            # distance between consecutive sentences = 1 - cos_sim
            dists = 1.0 - np.sum(emb[:-1] * emb[1:], axis=1)
            threshold = float(np.percentile(dists, self.percentile))
            groups: list[list[str]] = [[sents[0]]]
            for i, d in enumerate(dists):
                if d > threshold:
                    groups.append([])
                groups[-1].append(sents[i + 1])
            parent_texts = [" ".join(g) for g in groups if g]
            for pi, parent_text in enumerate(parent_texts):
                parent_id = Chunk.make_id(doc.source, parent_text, pi)
                child_sents = split_sentences(parent_text)
                children = _pack(child_sents, self.s.chunk_tokens, self.s.chunk_overlap_tokens)
                for ci, child_text in enumerate(children):
                    meta = dict(doc.metadata)
                    meta["loc"] = f"s{pi}.c{ci}"
                    chunks.append(
                        Chunk(
                            id=Chunk.make_id(doc.source, child_text, pi * 1000 + ci),
                            text=child_text,
                            source=doc.source,
                            parent_id=parent_id,
                            parent_text=parent_text,
                            metadata=meta,
                        )
                    )
        return chunks
