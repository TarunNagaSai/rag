"""Chunking — turn raw Documents into parent/child Chunk pairs.

Strategy (page-as-parent):
  - Parent  = the full Document text (one page of the PDF, or one text file).
              Stored alongside every child so the LLM always gets full-page
              context at generation time.
  - Children = overlapping sentence windows within that parent, small enough
              for precise dense/BM25 retrieval.

SemanticChunker is an optional upgrade that detects topic-shift boundaries
via embedding distance before creating children — useful when page text mixes
several unrelated topics (e.g. a page that covers both revenue data and a
risk disclosure paragraph).
"""

from __future__ import annotations

import re

import numpy as np

from advanced_rag.core.config import Settings, get_settings
from advanced_rag.llm.gemini import Gemini
from advanced_rag.schema.schema import Chunk, Document

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
_PARA_SPLIT = re.compile(r"\n\s*\n")


def approx_tokens(text: str) -> int:
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


def _make_children(
    parent_text: str,
    parent_id: str,
    source: str,
    base_meta: dict,
    page: int,
    settings: Settings,
) -> list[Chunk]:
    sentences = split_sentences(parent_text)
    if not sentences:
        return []
    windows = _pack(sentences, settings.chunk_tokens, settings.chunk_overlap_tokens)
    chunks: list[Chunk] = []
    for ci, child_text in enumerate(windows):
        meta = dict(base_meta)
        meta["loc"] = f"p{page}.c{ci}"
        chunks.append(
            Chunk(
                id=Chunk.make_id(source, child_text, page * 1000 + ci),
                text=child_text,
                source=source,
                parent_id=parent_id,
                parent_text=parent_text,
                metadata=meta,
            )
        )
    return chunks


def chunk_documents(
    documents: list[Document], settings: Settings | None = None
) -> list[Chunk]:
    """Page-as-parent chunking.

    Each Document produces one parent (the full page/file text) and N children
    (overlapping sentence windows). Retrieval matches on children; generation
    receives the full parent for context.
    """
    s = settings or get_settings()
    chunks: list[Chunk] = []
    for doc in documents:
        if not doc.text.strip():
            continue
        page = doc.metadata.get("page", 0)
        parent_id = Chunk.make_id(doc.source, doc.text, page)
        chunks.extend(
            _make_children(doc.text, parent_id, doc.source, doc.metadata, page, s)
        )
    return chunks


class SemanticChunker:
    """Optional: split on topic-shift boundaries before creating children.

    Embeds each sentence, measures cosine distance between consecutive
    sentences, and starts a new semantic section wherever distance exceeds a
    percentile threshold. Each section becomes its own parent, with children
    built from that section's text.

    Use when a single page mixes unrelated topics (e.g. financial data
    followed by an unrelated risk disclosure on the same page).
    """

    def __init__(
        self,
        gemini: Gemini | None = None,
        settings: Settings | None = None,
        percentile: float = 90.0,
    ):
        self.s = settings or get_settings()
        self.g = gemini or Gemini(self.s)
        self.percentile = percentile

    def chunk(self, documents: list[Document]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for doc in documents:
            if not doc.text.strip():
                continue
            page = doc.metadata.get("page", 0)
            sents = split_sentences(doc.text)

            if len(sents) < 4:
                # Too short to detect shifts — use full page as parent
                parent_id = Chunk.make_id(doc.source, doc.text, page)
                chunks.extend(
                    _make_children(doc.text, parent_id, doc.source, doc.metadata, page, self.s)
                )
                continue

            emb = self.g.embed(sents)
            dists = 1.0 - np.sum(emb[:-1] * emb[1:], axis=1)
            threshold = float(np.percentile(dists, self.percentile))

            groups: list[list[str]] = [[sents[0]]]
            for i, d in enumerate(dists):
                if d > threshold:
                    groups.append([])
                groups[-1].append(sents[i + 1])

            for si, group in enumerate(groups):
                if not group:
                    continue
                section_text = " ".join(group)
                parent_id = Chunk.make_id(doc.source, section_text, page * 100 + si)
                chunks.extend(
                    _make_children(
                        section_text, parent_id, doc.source, doc.metadata,
                        page * 100 + si, self.s
                    )
                )
        return chunks
