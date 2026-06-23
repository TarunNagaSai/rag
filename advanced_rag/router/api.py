"""FastAPI server — exposes the RAGPipeline over HTTP.

This is a thin transport layer. All the intelligence lives in ``RAGPipeline``;
here we just (de)serialize requests, hold a single shared pipeline, and guard the
few mutating operations (ingest / chat-memory) with a lock.

Run it:

    uvicorn advanced_rag.api:app --reload          # dev
    arag-api                                        # console script (see pyproject)

The Gemini API key is read from the environment exactly like the CLI
(GOOGLE_API_KEY or GEMINI_API_KEY). Endpoints that need the model return HTTP 503
with a clear message if no valid key is configured, so the server still boots for
health checks and docs without one.

Interactive docs are served at /docs (Swagger) and /redoc once running.
"""

from __future__ import annotations

import json
import re
import tempfile
import threading
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any, Literal

from agents.pipeline import AskResult, RAGPipeline
from core.config import get_settings
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from tools.store import HybridStore, build_filter

# --------------------------------------------------------------------------- app

app = FastAPI(
    title="Advanced RAG API",
    version="0.1.0",
    description="Hybrid + GraphRAG + agentic retrieval over Google Gemini.",
)

# The Next.js UI runs on :3000 in dev; allow it (and common local hosts).
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------------------------------------- pipeline state

_pipeline: RAGPipeline | None = None
_lock = threading.Lock()  # serialize mutations (ingest, chat memory)


def get_pipeline() -> RAGPipeline:
    """Lazily build (or load) the single shared pipeline.

    Construction touches Gemini, which requires an API key — so a missing key
    surfaces here as a clean 503 rather than a 500 stack trace.
    """
    global _pipeline
    if _pipeline is not None:
        return _pipeline
    with _lock:
        if _pipeline is not None:  # double-checked under lock
            return _pipeline
        try:
            get_settings().require_key()
        except Exception as e:  # noqa: BLE001 - want the message verbatim
            raise HTTPException(
                status_code=503,
                detail=(
                    "No Gemini API key configured. Set GOOGLE_API_KEY (or "
                    f"GEMINI_API_KEY) and retry. ({e})"
                ),
            )
        # Reuse a previously-built index if one exists on disk.
        _pipeline = RAGPipeline.load() if HybridStore.exists() else RAGPipeline()
        return _pipeline


# ------------------------------------------------------------------------ models


class IngestTextRequest(BaseModel):
    text: str | None = Field(default=None, description="Raw text to ingest")
    source: str | None = Field(default=None, description="File or directory path")
    build_graph: bool = True
    semantic: bool = False


class IngestResponse(BaseModel):
    chunks_added: int
    total_chunks: int
    graph_built: bool


class AskRequest(BaseModel):
    question: str
    mode: Literal["agentic", "simple"] = "agentic"
    sources: list[str] | None = Field(
        default=None, description="Restrict to sources whose path contains these"
    )
    where: dict[str, Any] | None = Field(
        default=None, description="Metadata equality/membership filter"
    )


class ChatRequest(BaseModel):
    question: str
    mode: Literal["agentic", "simple"] = "agentic"


class SourceOut(BaseModel):
    n: int
    citation: str
    cited: bool


class StepOut(BaseModel):
    question: str
    tool: str
    grade: str
    n_evidence: int


class AskResponse(BaseModel):
    answer: str
    rendered: str
    mode: str
    sources: list[SourceOut] = []
    # retrieval (simple mode)
    grade: str | None = None
    grade_reason: str | None = None
    queries_used: list[str] = []
    # agent (agentic mode)
    plan: list[str] = []
    steps: list[StepOut] = []
    trace: list[str] = []


class InfoResponse(BaseModel):
    indexed_chunks: int
    has_graph: bool
    index_dir: str
    embed_model: str
    gen_model: str
    reasoning_model: str
    features: dict[str, bool]


# ---------------------------------------------------------------- serialization


def _to_ask_response(result: AskResult) -> AskResponse:
    ans = result.answer
    cited = set(ans.cited)
    sources = [SourceOut(n=n, citation=c, cited=n in cited) for n, c in ans.sources]
    out = AskResponse(
        answer=ans.text,
        rendered=ans.render(),
        mode=result.mode,
        sources=sources,
    )
    if result.retrieval is not None:
        rr = result.retrieval
        out.grade = rr.grade
        out.grade_reason = rr.grade_reason
        out.queries_used = rr.queries_used
        out.trace = rr.trace
    if result.agent is not None:
        ar = result.agent
        out.plan = ar.plan
        out.steps = [
            StepOut(
                question=s.question, tool=s.tool, grade=s.grade, n_evidence=s.n_evidence
            )
            for s in ar.steps
        ]
        out.trace = ar.trace
    return out


# ------------------------------------------------------------------- endpoints


@app.get("/")
def root() -> str:
    return "welcome to the advanced rag api"


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/info", response_model=InfoResponse)
def info() -> InfoResponse:
    s = get_settings()
    p = get_pipeline()
    return InfoResponse(
        indexed_chunks=len(p.store.chunks),
        has_graph=p.graph is not None,
        index_dir=str(s.index_dir),
        embed_model=s.embed_model,
        gen_model=s.gen_model,
        reasoning_model=s.reasoning_model,
        features={
            "rerank": s.rerank_enabled,
            "hyde": s.hyde_enabled,
            "multiquery": s.multiquery_enabled,
            "crag": s.crag_enabled,
        },
    )


@app.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestTextRequest) -> IngestResponse:
    if not req.text and not req.source:
        raise HTTPException(
            status_code=400, detail="Provide either 'text' or 'source'."
        )
    p = get_pipeline()
    with _lock:
        added = p.ingest(
            source=req.source,
            text=req.text,
            build_graph=req.build_graph,
            semantic=req.semantic,
        )
        total = len(p.store.chunks)
        graph_built = p.graph is not None
    return IngestResponse(
        chunks_added=added, total_chunks=total, graph_built=graph_built
    )


@app.post("/ingest/file", response_model=IngestResponse)
async def ingest_file(
    file: UploadFile = File(...),
    build_graph: bool = True,
    semantic: bool = False,
) -> IngestResponse:
    p = get_pipeline()
    suffix = Path(file.filename or "upload.txt").suffix or ".txt"
    data = await file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        with _lock:
            added = p.ingest(
                source=tmp_path, build_graph=build_graph, semantic=semantic
            )
            total = len(p.store.chunks)
            graph_built = p.graph is not None
    finally:
        Path(tmp_path).unlink(missing_ok=True)
    return IngestResponse(
        chunks_added=added, total_chunks=total, graph_built=graph_built
    )


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    p = get_pipeline()
    if not p.store.chunks:
        raise HTTPException(
            status_code=409, detail="Index is empty. Ingest documents first."
        )
    result = p.ask(req.question, mode=req.mode, sources=req.sources, where=req.where)
    return _to_ask_response(result)


@app.post("/chat", response_model=AskResponse)
def chat(req: ChatRequest) -> AskResponse:
    p = get_pipeline()
    if not p.store.chunks:
        raise HTTPException(
            status_code=409, detail="Index is empty. Ingest documents first."
        )
    with _lock:  # chat mutates conversation memory
        result = p.chat(req.question, mode=req.mode)
    return _to_ask_response(result)


@app.post("/ask/stream")
async def ask_stream(req: AskRequest) -> StreamingResponse:
    """Stream the answer as Server-Sent Events (simple retrieval mode only).

    Each event is a JSON object on the ``data:`` line:
      - ``{"type": "chunk",   "text": "..."}``  — model text as it arrives
      - ``{"type": "sources", "sources": [...]}`` — citation list with cited flags
      - ``data: [DONE]``                          — end of stream sentinel
    """
    p = get_pipeline()
    if not p.store.chunks:
        raise HTTPException(
            status_code=409, detail="Index is empty. Ingest documents first."
        )

    filt = build_filter(req.sources, req.where)
    rr = p.retriever.retrieve(req.question, filt)
    sources, stream = p.generator.answer_stream(req.question, rr.evidence)

    async def event_gen() -> AsyncIterator[str]:
        accumulated: list[str] = []
        for chunk in stream:
            accumulated.append(chunk)
            yield f"data: {json.dumps({'type': 'chunk', 'text': chunk})}\n\n"
        full_text = "".join(accumulated)
        cited = sorted({int(m) for m in re.findall(r"\[(\d+)\]", full_text)})
        sources_out = [
            {"n": n, "citation": c, "cited": n in cited} for n, c in sources
        ]
        yield f"data: {json.dumps({'type': 'sources', 'sources': sources_out})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")


@app.post("/reset")
def reset_memory() -> dict[str, str]:
    """Clear conversation memory (does not touch the index)."""
    p = get_pipeline()
    with _lock:
        p.memory.turns.clear()
    return {"status": "memory cleared"}


def main() -> None:
    """Console-script entry point (``arag-api``)."""
    import uvicorn

    uvicorn.run("advanced_rag.api:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
