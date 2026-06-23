"""Central configuration.

Everything tunable lives here so the rest of the code reads cleanly. Values can be
overridden with environment variables (see ``.env.example``) which makes it easy to
A/B different models or dimensions without touching code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


@dataclass(frozen=True)
class Settings:
    # --- Auth -------------------------------------------------------------
    # Accept either name; GOOGLE_API_KEY wins (matches the SDK's own precedence).
    api_key: str = field(
        default_factory=lambda: os.environ.get("GOOGLE_API_KEY")
        or os.environ.get("GEMINI_API_KEY", "")
    )

    # --- Models -----------------------------------------------------------
    # Fast workhorse model for routing, grading, expansion, synthesis.
    gen_model: str = field(default_factory=lambda: _env("ARAG_GEN_MODEL", "gemini-2.5-flash"))
    # Stronger model reserved for hard multi-hop reasoning / final synthesis.
    reasoning_model: str = field(default_factory=lambda: _env("ARAG_REASONING_MODEL", "gemini-2.5-pro"))
    embed_model: str = field(default_factory=lambda: _env("ARAG_EMBED_MODEL", "gemini-embedding-001"))

    # --- Embeddings -------------------------------------------------------
    # gemini-embedding-001 supports Matryoshka truncation (128..3072).
    # 1536 is a strong quality/footprint trade-off; we always L2-normalize.
    embed_dim: int = field(default_factory=lambda: int(_env("ARAG_EMBED_DIM", "1536")))
    embed_batch_size: int = field(default_factory=lambda: int(_env("ARAG_EMBED_BATCH", "32")))

    # --- Chunking ---------------------------------------------------------
    chunk_tokens: int = 320          # target child-chunk size (approx tokens)
    chunk_overlap_tokens: int = 60   # overlap between adjacent chunks
    parent_chunk_tokens: int = 1200  # parent block size for parent-doc retrieval

    # --- Retrieval --------------------------------------------------------
    dense_top_k: int = 20            # candidates from the vector index
    lexical_top_k: int = 20          # candidates from BM25
    rrf_k: int = 60                  # Reciprocal Rank Fusion constant
    fused_top_k: int = 12            # candidates kept after fusion -> reranker
    final_top_k: int = 6             # passages actually sent to the LLM
    rerank_enabled: bool = True
    hyde_enabled: bool = True
    multiquery_enabled: bool = True
    multiquery_n: int = 3            # number of query rewrites
    crag_enabled: bool = True

    # --- Agent ------------------------------------------------------------
    agent_max_hops: int = 5
    agent_max_subquestions: int = 6

    # --- Persistence ------------------------------------------------------
    index_dir: Path = field(
        default_factory=lambda: Path(_env("ARAG_INDEX_DIR", ".arag_index"))
    )

    def require_key(self) -> str:
        if not self.api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY is not set. Copy .env.example to .env and add your key, "
                "or export GOOGLE_API_KEY in your shell. Get one at "
                "https://aistudio.google.com/apikey"
            )
        return self.api_key


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
