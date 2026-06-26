# Learning Log

Spaced-repetition log of things I've learned and integrated. Quiz yourself with the
**Q** lines before reading the **A**. Re-visit dated entries after 1 day, 1 week, 1 month.

---

## 2026-06-24 — Langfuse observability → FastAPI RAG (v4 SDK)

**X:** Langfuse tracing. **Y:** the `advanced_rag` FastAPI app. Passed the teach-back gate.

### Core concepts (flashcards)

**Q: Session vs Trace vs Observation?**
A: Session = the whole conversation (one chat window). Trace = one turn (user question →
LLM answer). Observation = one step *inside* a turn. Nesting: session ⊃ trace ⊃ observations.

**Q: span vs generation?**
A: Both are observation types. A **span** is a generic stopwatch around any unit of work
(retrieval, a tool call). A **generation** is a span *for an LLM call* — it additionally
captures **model, token usage, and cost**. Input/output exist on *both*; the model+tokens+cost
are the generation-only payoff. Downgrade a generation to a span and you lose exactly those.

**Q: How does an observation "connect" to the real work (e.g. Gemini)?**
A: It doesn't — automatically. An observation is a **dumb recorder**: a stopwatch + notepad.
It captures nothing you don't hand it. You wrap the real call in a `with` block (stopwatch) and
you `.update(output=..., usage_details=...)` it yourself (notepad). Forget the update → blank
output in the UI, because nothing told it what came back. The Langfuse generation never *calls*
Gemini; it just *watches* you call it and records what you pass.

**Q: What makes nesting work (parent/child)?**
A: Whoever is the **current** observation when a new one *starts* becomes its parent. A `with`
block sets itself "current" on entry and reverts on exit. So two blocks that open-and-close in
sequence (under a root that stays open) become **siblings**; indenting one *inside* the other's
block makes it a **child**. The tree is built from runtime context, not text indentation — it
flows through the call stack across files/functions.

**Q: Agentic RAG (ReAct) observation shape?**
A: One **root span per turn**, created ONCE *outside* the loop (root inside the loop = one trace
per iteration — wrong). Inside the loop, each iteration = a **generation** (the LLM "think/decide"
step) + a **conditional span** (the tool/retrieval action, guarded by `if response.tool_calls`).
llm and tool come out as siblings under root.

**Q: Flush — why and when?**
A: The SDK batches traces and a **background thread auto-flushes** on a timer (that's why traces
appear during normal running, no manual flush needed — never flush per request, it kills batching).
The one data-loss moment is **shutdown**: the process dies with items still in the outbox. Fix =
`flush()` then `shutdown()` in the FastAPI **lifespan shutdown**. `shutdown()` flushes *and* stops
the background thread cleanly so the process can exit.

### Python prerequisites that clicked

**Q: What does `with X() as y:` guarantee?**
A: Two blocks run — entry and exit — and the **exit always runs, even on exception** (it's a
built-in try/finally). That's why Langfuse uses it: spans never get stuck open with no end time.
`with` is a Python keyword; objects work in it by defining `__enter__`/`__exit__`.

**Q: What does `@asynccontextmanager` do?**
A: Turns an `async def` function with a single `yield` into an `async with`-able context manager.
Code **before yield** = startup; `yield` = hand control to the app for its lifetime; code **after
yield** = shutdown. Exactly one yield is required — my old lifespan had none, so it wasn't a valid
context manager and the app wouldn't start. FastAPI uses it via `FastAPI(lifespan=...)`.

### Gotchas / version notes

- **Installed SDK is Langfuse v4.9.1.** v4 unified all constructors: use
  `start_as_current_observation(name=..., as_type="generation", model=...)` — NOT the v3
  `start_as_current_generation(...)`. Same concept, different dialect. Always `pip show langfuse`
  before writing; APIs shift between major versions.
- **Langfuse is LLM observability, not an APM.** It only traces what you instrument. Wrap the
  agent loop / retrieval / tool calls / Gemini generations — NOT arbitrary HTTP/DB calls
  (that's Datadog/OTel/Sentry territory). Langfuse v3+ runs on OpenTelemetry under the hood.

### Debugging war story (2026-06-24/25) — "traces not showing"

Symptom: code instrumented correctly, `auth_check()` returned **True**, but no traces in the UI.
Cause: **self-hosted Langfuse v3 ingestion needs an S3/MinIO blob store**, and the `minio`
container had been removed (mistaken belief it was "only for image uploads"). Ingestion flow is
**Web → S3(MinIO) → Redis → Worker → ClickHouse** — the *first* hop writes every event blob to
`events/otel/...json`. No S3 → `getaddrinfo ENOTFOUND minio` → ingestion 500s → zero traces.

Key traps:
- `auth_check()` hits Postgres, NOT the ingestion path — it passes even when ingestion is broken.
- The smoke-test error to look for: `Failed to upload JSON to S3 events/...` in the
  `langfuse-web` container logs (`docker logs <web> | grep -i s3`).
- `LANGFUSE_S3_EVENT_UPLOAD_*` = mandatory (all traces). `LANGFUSE_S3_MEDIA_UPLOAD_*` = the
  images/audio part (optional). They share one bucket but are different roles.
- The running stack was `bakery_os`'s compose (`infra` project, `infra_default` network); container
  names `bakery-os-*` are just `container_name:` labels. Fix = bring `minio` up on the SAME
  network so langfuse-web resolves host `minio`. No restart needed — docker resolves DNS per-connection.

**Q: auth_check passes but no traces appear — first thing to check on self-hosted Langfuse?**
A: The blob store. `docker logs <langfuse-web> | grep -i s3` for `Failed to upload JSON to S3`.
Ingestion needs MinIO/S3; auth_check doesn't, so it lies to you.

### Status of the integration

- ✅ `langufuse.py`: singleton getter (`Langfuse()` self-configures from env) + lifespan
  (auth_check on startup, flush+shutdown on exit). **Tracing always-on** — no enable/disable
  setting, keys read straight from env (LANGFUSE_PUBLIC_KEY / SECRET_KEY / HOST).
- ⬜ `router/api.py`: pass `lifespan=langfuse_lifespan` into `FastAPI(...)` (must be at construction)
- ⬜ Instrument the agent loop (root span → generation per LLM call → conditional tool span)
