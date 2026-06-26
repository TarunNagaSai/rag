"""pgvector-backed vector store.

Schema (created on first use):
  chunks table — one row per child chunk, embedding column is a pgvector
  HNSW index for fast approximate cosine search.

Hybrid retrieval combines:
  - Dense:   pgvector cosine similarity  (<=> operator)
  - Lexical: PostgreSQL full-text search (tsvector / tsquery)
  - Fusion:  Reciprocal Rank Fusion in Python
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable

import numpy as np
import psycopg2
import psycopg2.extras
from pgvector.psycopg2 import register_vector

from advanced_rag.core.config import Settings, get_settings
from advanced_rag.llm.gemini import Gemini
from advanced_rag.schema.schema import Chunk, Scored

Filter = Callable[[Chunk], bool]

_DDL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
    id          TEXT PRIMARY KEY,
    text        TEXT        NOT NULL,
    source      TEXT        NOT NULL,
    parent_id   TEXT        NOT NULL,
    parent_text TEXT        NOT NULL,
    metadata    JSONB       NOT NULL DEFAULT '{}',
    embedding   vector({dim})
);

CREATE INDEX IF NOT EXISTS chunks_hnsw_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);
"""


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9_]+", text.lower())


def _rrf(ranked_lists: list[list[str]], k: int = 60) -> dict[str, float]:
    scores: dict[str, float] = {}
    for lst in ranked_lists:
        for rank, doc_id in enumerate(lst):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return scores


def build_filter(
    sources: list[str] | None = None,
    where: dict[str, Any] | None = None,
) -> Filter | None:
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
        self._conn: psycopg2.extensions.connection | None = None

    # ---------------------------------------------------------------- connect
    def _connect(self) -> psycopg2.extensions.connection:
        if self._conn is None or self._conn.closed:
            self._conn = psycopg2.connect(self.s.database_url)
            register_vector(self._conn)
        return self._conn

    def _setup(self) -> None:
        conn = self._connect()
        with conn.cursor() as cur:
            cur.execute(_DDL.format(dim=self.s.embed_dim))
        conn.commit()

    # ------------------------------------------------------------------- add
    def add(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        self._setup()

        embeddings = self.g.embed([c.text for c in chunks])
        conn = self._connect()
        with conn.cursor() as cur:
            psycopg2.extras.execute_values(
                cur,
                """
                INSERT INTO chunks (id, text, source, parent_id, parent_text, metadata, embedding)
                VALUES %s
                ON CONFLICT (id) DO NOTHING
                """,
                [
                    (
                        c.id,
                        c.text,
                        c.source,
                        c.parent_id,
                        c.parent_text,
                        json.dumps(c.metadata),
                        embeddings[i].tolist(),
                    )
                    for i, c in enumerate(chunks)
                ],
            )
        conn.commit()

    # --------------------------------------------------------------- search
    def dense_search(
        self, query_vec: np.ndarray, top_k: int, source_filter: str | None = None
    ) -> list[tuple[str, float]]:
        conn = self._connect()
        sql = """
            SELECT id, 1 - (embedding <=> %s::vector) AS score
            FROM chunks
            {where}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """.format(where="WHERE source LIKE %s" if source_filter else "")

        params: list[Any] = [query_vec.tolist()]
        if source_filter:
            params.append(f"%{source_filter}%")
        params += [query_vec.tolist(), top_k]

        with conn.cursor() as cur:
            cur.execute(sql, params)
            return [(row[0], row[1]) for row in cur.fetchall()]

    def lexical_search(
        self, query: str, top_k: int, source_filter: str | None = None
    ) -> list[tuple[str, float]]:
        conn = self._connect()
        ts_query = " & ".join(_tokenize(query)) or "x"
        sql = """
            SELECT id,
                   ts_rank(to_tsvector('english', text), to_tsquery('english', %s)) AS score
            FROM chunks
            WHERE to_tsvector('english', text) @@ to_tsquery('english', %s)
            {where}
            ORDER BY score DESC
            LIMIT %s
        """.format(where="AND source LIKE %s" if source_filter else "")

        params: list[Any] = [ts_query, ts_query]
        if source_filter:
            params.append(f"%{source_filter}%")
        params.append(top_k)

        with conn.cursor() as cur:
            cur.execute(sql, params)
            return [(row[0], float(row[1])) for row in cur.fetchall()]

    def hybrid_search(
        self,
        query: str,
        query_vec: np.ndarray,
        *,
        top_k: int | None = None,
        filt: Filter | None = None,
    ) -> list[Scored]:
        top_k = top_k or self.s.fused_top_k

        dense = self.dense_search(query_vec, self.s.dense_top_k)
        lexical = self.lexical_search(query, self.s.lexical_top_k)

        fused = _rrf(
            [[d for d, _ in dense], [l for l, _ in lexical]], k=self.s.rrf_k
        )
        ranked = sorted(fused.items(), key=lambda kv: -kv[1])[:top_k]

        # Fetch the actual chunk rows
        ids = [cid for cid, _ in ranked]
        if not ids:
            return []

        chunks_by_id = self._fetch_by_ids(ids)
        results = [
            Scored(chunk=chunks_by_id[cid], score=score, how="rrf")
            for cid, score in ranked
            if cid in chunks_by_id
        ]

        if filt:
            results = [r for r in results if filt(r.chunk)]

        return results

    def _fetch_by_ids(self, ids: list[str]) -> dict[str, Chunk]:
        conn = self._connect()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, text, source, parent_id, parent_text, metadata "
                "FROM chunks WHERE id = ANY(%s)",
                (ids,),
            )
            return {
                row[0]: Chunk(
                    id=row[0],
                    text=row[1],
                    source=row[2],
                    parent_id=row[3],
                    parent_text=row[4],
                    metadata=row[5] if isinstance(row[5], dict) else json.loads(row[5]),
                )
                for row in cur.fetchall()
            }

    # ---------------------------------------------------------- chunk access
    @property
    def chunks(self) -> list[Chunk]:
        conn = self._connect()
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, text, source, parent_id, parent_text, metadata FROM chunks"
            )
            return [
                Chunk(
                    id=row[0],
                    text=row[1],
                    source=row[2],
                    parent_id=row[3],
                    parent_text=row[4],
                    metadata=row[5] if isinstance(row[5], dict) else json.loads(row[5]),
                )
                for row in cur.fetchall()
            ]

    def get(self, chunk_id: str) -> Chunk | None:
        result = self._fetch_by_ids([chunk_id])
        return result.get(chunk_id)

    def parent_text_for(self, chunk: Chunk) -> str:
        return chunk.parent_text

    # ----------------------------------------------------------- persistence
    def save(self, path=None):
        # Data lives in PostgreSQL — nothing to flush
        return path

    @classmethod
    def load(
        cls,
        path=None,
        settings: Settings | None = None,
        gemini: Gemini | None = None,
    ) -> "HybridStore":
        s = settings or get_settings()
        store = cls(s, gemini or Gemini(s))
        store._setup()
        return store

    @staticmethod
    def exists(path=None, settings: Settings | None = None) -> bool:
        s = settings or get_settings()
        try:
            conn = psycopg2.connect(s.database_url)
            register_vector(conn)
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM chunks")
                count = cur.fetchone()[0]
            conn.close()
            return count > 0
        except Exception:
            return False
