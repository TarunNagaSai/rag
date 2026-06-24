"""Thin, well-behaved wrapper around the Google Gen AI SDK.

Why a wrapper at all? Three reasons:
  1. One place to apply retries/backoff (the network is flaky; quotas exist).
  2. Embeddings need *different task types* for documents vs queries — getting this
     right is one of the highest-leverage, least-known RAG tricks.
  3. We always L2-normalize embeddings so cosine similarity == dot product, which
     keeps the vector store fast and correct after Matryoshka truncation.
"""

from __future__ import annotations

from typing import Sequence, Type, TypeVar

import numpy as np
from advanced_rag.core.config import Settings, get_settings
from google import genai
from google.genai import types
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

T = TypeVar("T", bound=BaseModel)

# Task types tell the embedding model how the text will be used. Documents and
# queries are embedded into a shared space but with asymmetric optimization.
TASK_DOCUMENT = "RETRIEVAL_DOCUMENT"
TASK_QUERY = "RETRIEVAL_QUERY"


class GeminiEmbadding:
    def __init__(self, settings: Settings | None = None):
        self.s = settings or get_settings()
        self.client = genai.Client(api_key=self.s.require_key())

    # ------------------------------------------------------------------ embed
    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, max=30))
    def _embed_raw(self, batch: list[str], task_type: str) -> list[list[float] | None]:
        resp = self.client.models.embed_content(
            model=self.s.embed_model,
            contents=list(batch),
            config=types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=self.s.embed_dim,
            ),
        )
        return [e.values for e in resp.embeddings]

    def embed(
        self, texts: Sequence[str], *, task_type: str = TASK_DOCUMENT
    ) -> np.ndarray:
        """Embed texts -> (n, dim) float32 array, L2-normalized.

        Batches requests and degrades gracefully: if a batch fails (e.g. one bad
        item), it retries the items one at a time so a single document can't sink
        an entire ingestion run.
        """
        if not texts:
            return np.zeros((0, self.s.embed_dim), dtype=np.float32)

        vectors: list[list[float]] = []
        bs = self.s.embed_batch_size
        for start in range(0, len(texts), bs):
            batch = list(texts[start : start + bs])
            try:
                vectors.extend(self._embed_raw(batch, task_type))
            except Exception:
                for item in batch:  # one-at-a-time fallback
                    vectors.extend(self._embed_raw([item], task_type))

        arr = np.asarray(vectors, dtype=np.float32)
        # Matryoshka truncation (<3072) requires renormalization for cosine to hold.
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return arr / norms

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed([text], task_type=TASK_QUERY)[0]
