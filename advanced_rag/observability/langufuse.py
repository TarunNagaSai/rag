"""Langfuse observability — wiring only.

Two jobs:

1. Hand out a single, process-wide ("singleton") Langfuse client. One client owns one
   background flushing thread and one connection; creating a client per request would
   spawn a thread each time and defeat the batching that keeps the app fast.
2. Provide a FastAPI lifespan that verifies credentials on startup and flushes the
   outbox on shutdown so the final batch of traces is not lost when the process exits.

Tracing is ALWAYS ON. The client self-configures from environment variables
(LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST) — set those in your env / .env.
The actual instrumentation (spans / generations) lives in the agent code, NOT here.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from langfuse import Langfuse

_client: Langfuse | None = None


def get_langfuse_client() -> Langfuse:
    """Return the shared Langfuse client, creating it once on first use.

    No args: the SDK reads LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_HOST
    from the environment, and tracing is enabled by default.
    """
    global _client
    if _client is None:
        _client = Langfuse()
    return _client


@asynccontextmanager
async def langfuse_lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup: verify the keys. Shutdown: flush the outbox, then stop cleanly.

    Everything before ``yield`` runs once at boot; everything after runs once as the
    server stops — the one guaranteed moment to flush before the process dies.
    """
    client = get_langfuse_client()

    try:
        client.auth_check()
    except Exception as exc:  # a bad/missing key must not stop the server booting
        print(f"[langfuse] auth check failed: {exc}")

    yield  # ← the app serves requests for its entire lifetime here

    client.flush()      # send whatever is still in the outbox
    client.shutdown()   # stop the background thread cleanly
