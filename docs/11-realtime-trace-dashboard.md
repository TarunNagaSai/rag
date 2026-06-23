# 11 · Real-Time Agent Trace Dashboard

> Goal: expose every internal step of `AgenticRAG` to the browser in real time —
> plan, tool routing, retrieval grade, synthesis — so the pipeline is never a black box.

## How it fits the existing project

`AgenticRAG.answer` already produces a rich `AgentResult` (plan, steps, trace).
Today those arrive all-at-once after the full run. This feature streams each event
**as it happens** via SSE, renders them in a new side panel in the existing Next.js UI.

No existing logic is rewritten; we only add:

1. A streaming generator method to `agent.py`
2. A new SSE endpoint in `api.py`
3. A hook + panel in the Next.js UI

---

## Event protocol (the lingua franca between backend and frontend)

Every SSE message is `data: <JSON>\n\n`. The `event` field is the discriminator.

```typescript
type TraceEvent =
  | { event: "trace_start" }
  | { event: "plan_ready";    subquestions: { question: string; tool: string; reason: string }[] }
  | { event: "step_start";    index: number; question: string; tool: string }
  | { event: "step_end";      index: number; grade: string; n_evidence: number }
  | { event: "synthesis_start" }
  | { event: "final_answer";  answer: string; rendered: string;
                               sources: SourceOut[];
                               input_tokens: number; output_tokens: number }
  | { event: "error";         message: string }
```

`SourceOut` mirrors the existing API type (`{ n, citation, cited }`).

Token counts come from the Gemini response metadata (`usage_metadata.prompt_token_count`
and `candidates[0].token_count`). The `Generator` class already receives this in the
raw API response — surface it on `Answer` and pass it through to `final_answer`.

---

## Phase 1 — Backend streaming

### 1a. `advanced_rag/generate.py` — add token fields to `Answer`

Small, isolated change needed by the streaming path:

```python
@dataclass
class Answer:
    text: str
    sources: list[tuple[int, str]]
    cited: set[int]
    input_tokens: int = 0   # add
    output_tokens: int = 0  # add
```

Populate them in `Generator.generate` from the Gemini response:

```python
answer.input_tokens = response.usage_metadata.prompt_token_count
answer.output_tokens = response.usage_metadata.candidates_token_count
```

### 1b. `advanced_rag/agent.py` — add `answer_stream()`

New async generator method on `AgenticRAG`. The existing `answer()` is untouched.

All blocking Gemini/retrieval calls are offloaded to a thread pool via
`asyncio.run_in_executor(None, fn)` so the event loop is never stalled waiting on
network I/O. Without this, `yield` between steps would appear to batch at Gemini
call boundaries, defeating the real-time effect.

```python
import asyncio

async def answer_stream(self, question: str) -> AsyncIterator[dict]:
    loop = asyncio.get_running_loop()
    run = lambda fn, *args: loop.run_in_executor(None, fn, *args)

    yield {"event": "trace_start"}

    # _plan calls Gemini — run in executor so the loop stays responsive
    plan = await run(self._plan, question)
    subs = plan.subquestions or [_SubQ(question=question, tool=Tool.HYBRID)]
    yield {"event": "plan_ready",
           "subquestions": [s.model_dump() for s in subs]}

    all_evidence: list[Scored] = []
    for i, sub in enumerate(subs[: self.s.agent_max_subquestions]):
        yield {"event": "step_start", "index": i,
               "question": sub.question, "tool": sub.tool}

        # _act + crag.grade both hit Gemini/vector store — offload each
        ev = await run(self._act, sub)
        grade = await run(self.crag.grade, sub.question, ev)
        if grade.grade == Grade.LOW and sub.tool != Tool.HYBRID:
            ev = (await run(self.retriever.retrieve, sub.question)).evidence
            grade = await run(self.crag.grade, sub.question, ev)
        all_evidence.extend(ev)

        yield {"event": "step_end", "index": i,
               "grade": grade.grade, "n_evidence": len(ev)}

        if len(all_evidence) >= self.s.agent_max_hops * self.s.top_k:
            break

    yield {"event": "synthesis_start"}
    answer = await run(self.generator.generate, question, all_evidence)

    yield {
        "event": "final_answer",
        "answer": answer.text,
        "rendered": answer.render(),
        "sources": [{"n": n, "citation": c, "cited": n in answer.cited}
                    for n, c in answer.sources],
        "input_tokens": answer.input_tokens,
        "output_tokens": answer.output_tokens,
    }
```

### 1c. `advanced_rag/api.py` — add `/api/chat-stream`

The endpoint is split into two requests to avoid query-string fragility (URL length
limits, encoding issues with complex questions):

1. `POST /api/chat-stream` — accepts the query body, stores it under a short-lived
   `session_id`, returns `{ session_id }`.
2. `GET /api/chat-stream/{session_id}` — the `EventSource` URL; looks up the query
   by `session_id` and streams events.

```python
import asyncio, json, uuid
from collections import OrderedDict

# In-memory session store — keyed by session_id, max 50 entries (LRU-style).
# Sessions are consumed once; the GET handler deletes the entry on first use.
_stream_sessions: OrderedDict[str, tuple[str, str]] = OrderedDict()
_MAX_SESSIONS = 50

class StreamRequest(BaseModel):
    question: str
    mode: Literal["agentic", "simple"] = "agentic"
    backend: Literal["native", "crewai"] = "native"

@app.post("/api/chat-stream")
def create_stream_session(req: StreamRequest) -> dict:
    """Register a query and return a session_id for the SSE endpoint."""
    sid = uuid.uuid4().hex
    _stream_sessions[sid] = (req.question, req.mode, req.backend)
    if len(_stream_sessions) > _MAX_SESSIONS:
        _stream_sessions.popitem(last=False)
    return {"session_id": sid}

@app.get("/api/chat-stream/{session_id}")
async def chat_stream(session_id: str):
    """SSE endpoint — streams agent trace events as JSON lines."""
    entry = _stream_sessions.pop(session_id, None)
    if entry is None:
        raise HTTPException(404, "Session not found or already consumed.")
    query, mode, backend = entry

    p = get_pipeline()
    if not p.store.chunks:
        raise HTTPException(409, "Index is empty. Ingest documents first.")

    stream_fn = (
        p.agent.answer_stream_crew if backend == "crewai"
        else p.agent.answer_stream
    )

    async def generate():
        try:
            async for event in stream_fn(query):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'event': 'error', 'message': str(exc)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
```

Note: `mode="simple"` falls through to a single-step stream (one `step_start`,
one `step_end`, then `final_answer`) using the existing `AdvancedRetriever`.

---

## Phase 2 — Frontend

### 2a. `ui/lib/use-agent-stream.ts` (new file)

Custom React hook that:
1. POSTs `{ question, mode, backend }` to `POST /api/chat-stream` to get a `session_id`
2. Opens `EventSource("/api/chat-stream/{session_id}")` — avoids query-string limits
3. Parses each message into a `TraceEvent`
4. Returns `{ events, status, finalAnswer, totalTokens }`

```typescript
type Status = "idle" | "running" | "done" | "error";

export function useAgentStream(query: string | null, mode = "agentic", backend = "native"): {
  events: TraceEvent[];
  status: Status;
  finalAnswer: string | null;
  rendered: string | null;
  sources: SourceOut[];
  totalTokens: number;   // input_tokens + output_tokens from final_answer event
}
```

- Fires `POST /api/chat-stream` when `query` changes from `null` to a string.
- Opens `EventSource` with the returned `session_id`.
- Closes the source on `final_answer` or `error`.
- Accumulates events in state so the UI can replay the full timeline.

### 2b. `ui/components/agent-thoughts-panel.tsx` (new file)

A collapsible right-side panel (same pattern as `RagPanel`) with three sub-sections:

#### Execution Stepper

Rendered as a vertical list of step cards:

```
┌────────────────────────────────────────────┐
│ ⬡  Planning…                               │
│    2 sub-questions identified              │
├────────────────────────────────────────────┤
│ ① Q: "Who manages enterprise accounts?"   │
│    Tool: HYBRID  ·  Grade: HIGH  ·  4 ev  │
├────────────────────────────────────────────┤
│ ② Q: "Which customers renewed in Q2?"     │
│    Tool: GRAPH_LOCAL  ·  Grade: MEDIUM     │
│    ↳ Fell back to HYBRID  ·  Grade: HIGH  │
└────────────────────────────────────────────┘
```

- Each card animates in on `step_start`, fills grade on `step_end`.
- A spinner icon replaces the checkmark while the step is running.
- Grade is colored: HIGH = green, MEDIUM = amber, LOW = red.

#### Thought Console

A terminal-style `<pre>` that appends a new line on each event:

```
[PLAN]   Decomposing into 2 sub-questions
[TOOL]   hybrid → "Who manages enterprise accounts?"
[GRADE]  HIGH  (4 evidence chunks)
[TOOL]   graph_local → "Which customers renewed in Q2?"
[GRADE]  MEDIUM — falling back to hybrid
[GRADE]  HIGH  (6 evidence chunks)
[SYNTH]  Synthesizing final answer…
[DONE]   Answer ready
```

#### Footer bar

- **Live evidence counter**: "12 chunks collected" — updates on each `step_end`.
- **Token counter**: "1 842 tokens" — populated from `input_tokens + output_tokens`
  on the `final_answer` event. Shows "—" while the run is in progress.

### 2c. `ui/app/page.tsx` — wire up

Replace the `setTimeout` stub in `handleSend` with:

```typescript
const { events, status, finalAnswer, rendered, sources, totalTokens }
  = useAgentStream(activeQuery);

useEffect(() => {
  if (status === "done" && finalAnswer) {
    setMessages(prev => prev.map(m =>
      m.id === pendingId
        ? { ...m, pending: false, content: rendered, citations: sources }
        : m
    ));
    setThinking(false);
  }
}, [status, finalAnswer]);
```

Add `<AgentThoughtsPanel>` to the layout alongside `<RagPanel>`:

```tsx
{showThoughts && (
  <AgentThoughtsPanel
    events={events}
    status={status}
    totalTokens={totalTokens}
  />
)}
```

Toggle `showThoughts` via a brain/sparkle icon in `TopBar` (next to the existing
panel toggle).

---

## Custom ReAct Loop variant

Using a hand-rolled ReAct loop instead of a framework. The loop is adapted from an
existing CLI research agent (`/Projects/Agentic AI/CLI Research Agent/agent.py`) and
wired into the RAG pipeline. **The event protocol, SSE endpoint, and all frontend
code are unchanged** — only the Python layer that drives the loop is swapped.

This is the production-grade approach: every line of the loop is owned, every
decision is explicit, and every failure mode is handled in code you wrote.

### What the custom loop replaces (and what it dissolves)

| Custom `AgenticRAG` code | In the ReAct variant |
|--------------------------|----------------------|
| `_plan()` — structured Gemini call to decompose | ReAct loop decomposes naturally by calling `retrieve` per sub-question as it reasons |
| `_act()` — routes to hybrid / graph / global | **Dissolved into `retrieve` tool** — routing + grading + fallback live inside one function |
| `crag.grade()` — verify and fallback | **Dissolved into `retrieve` tool** — grading runs before the tool returns |
| Manual `for` loop + budget check | `max_iterations` counter in the ReAct loop — you control the stopping condition |

The retrieval intelligence is not thrown away — it moves inside the `retrieve` tool.
The ReAct loop replaces the planning loop, not the retrieval logic.

### 1d. `advanced_rag/react_agent.py` (new file)

Contains three things: the tool schema Gemini understands, the `run_retrieve_tool`
function that dissolves the full RAG pipeline into a single call, and the `run_react_agent`
loop — a production-adapted version of the existing CLI research agent.

```python
import json
from queue import Queue
from .agent import AgenticRAG, Tool as RagTool, _SubQ
from .crag import Grade
from .schema import Scored

# ------------------------------------------------------------------ tool schema
# Passed to Gemini as a function definition — this is what the LLM sees when
# deciding whether and how to call the tool.

RETRIEVE_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "retrieve",
        "description": (
            "Retrieve grounded evidence passages for a specific question. "
            "Call once per distinct sub-question. Returns source-labelled passages."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The specific question to retrieve evidence for."
                }
            },
            "required": ["query"],
        },
    },
}

# ----------------------------------------------------------------- tool logic
# This is where the full RAG pipeline intelligence lives.
# The ReAct loop calls this; it never knows which retriever ran internally.

def run_retrieve_tool(
    query: str,
    rag: AgenticRAG,
    event_queue: Queue,
    step_index: int,
) -> str:
    # 1. hybrid first — always available, always reliable
    ev: list[Scored] = rag._act(_SubQ(question=query, tool=RagTool.HYBRID))
    grade = rag.crag.grade(query, ev)

    # 2. if graph exists and hybrid didn't grade HIGH, try graph_local
    if grade.grade != Grade.HIGH and rag.graph:
        graph_ev = rag._act(_SubQ(question=query, tool=RagTool.GRAPH_LOCAL))
        graph_grade = rag.crag.grade(query, graph_ev)

        if graph_grade.grade > grade.grade:
            ev, grade = graph_ev, graph_grade
        else:
            ev = ev + graph_ev   # merge — more evidence beats picking one

    event_queue.put({
        "event": "step_end",
        "index": step_index,
        "tool": "retrieve",
        "grade": grade.grade,
        "n_evidence": len(ev),
    })

    return "\n\n".join(f"[source {i+1}] {c.chunk.text}" for i, c in enumerate(ev))


# ------------------------------------------------------------------ ReAct loop

REACT_SYSTEM_PROMPT = """You are a grounded research assistant with access to a retrieval tool.

Your process:
1. Understand the question and decide if it needs to be broken into sub-questions.
2. For each sub-question, call the retrieve tool to get evidence.
3. Reason over the evidence — do not guess or add facts not in the sources.
4. When you have enough evidence, produce a final answer with citations [source N].

Rules:
- Only claim what retrieved evidence supports.
- If evidence is thin or contradictory, say so explicitly.
- Cite every factual claim with [source N].
- End your response with: Final Answer: <your complete grounded answer>
"""

def run_react_agent(
    question: str,
    rag: AgenticRAG,
    event_queue: Queue,
    max_iterations: int = 8,   # hard budget — prevents runaway loops
) -> str:
    """Synchronous ReAct loop. Run inside run_in_executor from the async wrapper."""
    from .gemini import Gemini   # use the project's existing Gemini wrapper

    gemini: Gemini = rag.g
    messages = [
        {"role": "system", "content": REACT_SYSTEM_PROMPT},
        {"role": "user",   "content": question},
    ]
    step_index = 0

    for iteration in range(max_iterations):
        response = gemini.chat(messages, tools=[RETRIEVE_TOOL_SCHEMA])
        messages.append({"role": "assistant", "content": response.content})

        # --- tool call branch ---
        if response.tool_calls:
            tool_call = response.tool_calls[0]
            args = json.loads(tool_call.function.arguments)
            query = args["query"]

            event_queue.put({
                "event": "step_start",
                "index": step_index,
                "thought": response.content or "",
                "tool": "retrieve",
                "question": query,
            })

            result = run_retrieve_tool(query, rag, event_queue, step_index)
            messages.append({"role": "tool", "content": result})
            step_index += 1

        # --- final answer branch ---
        if response.content and "Final Answer:" in response.content:
            event_queue.put({"event": "synthesis_start"})
            return response.content

    # budget exhausted — return best effort
    event_queue.put({"event": "synthesis_start"})
    return messages[-1]["content"] or "Could not produce a grounded answer within the iteration budget."
```

> **One addition to `gemini.py`**: add a `chat(messages, tools)` method that passes
> tool schemas to the Gemini API and returns a response object with `.content` and
> `.tool_calls`. The existing `generate()` method only handles single-turn — `chat`
> handles the multi-turn message list the ReAct loop builds up.

### 1e. `answer_stream_react()` on `AgenticRAG`

Same queue-drain pattern as before — `run_react_agent` runs in a thread, events
flow through the queue to the async generator.

```python
async def answer_stream_react(self, question: str) -> AsyncIterator[dict]:
    from queue import Queue, Empty
    from .react_agent import run_react_agent

    loop = asyncio.get_running_loop()
    event_queue: Queue = Queue()

    yield {"event": "trace_start"}

    # run the blocking ReAct loop in a thread pool
    future = loop.run_in_executor(
        None, run_react_agent, question, self, event_queue
    )

    # drain events in real time while the loop runs
    while not future.done():
        try:
            yield event_queue.get_nowait()
        except Empty:
            await asyncio.sleep(0.05)

    # drain any remaining events after the loop finishes
    while not event_queue.empty():
        yield event_queue.get_nowait()

    raw_output = await future

    # extract "Final Answer: ..." and generate a grounded, cited answer
    final_text = raw_output.split("Final Answer:")[-1].strip()
    answer = await loop.run_in_executor(
        None, self.generator.generate_from_text, question, final_text
    )

    yield {
        "event": "final_answer",
        "answer": answer.text,
        "rendered": answer.render(),
        "sources": [{"n": n, "citation": c, "cited": n in answer.cited}
                    for n, c in answer.sources],
        "input_tokens": answer.input_tokens,
        "output_tokens": answer.output_tokens,
    }
```

### Thought console mapping

Every part of the ReAct cycle maps to an explicit event you control:

| Loop stage | SSE event | Console line |
|---|---|---|
| LLM thought before tool call | `step_start.thought` | `[THOUGHT] <text>` |
| Tool called with query | `step_start.question` | `[TOOL]    retrieve → "<query>"` |
| Grade + evidence count returned | `step_end` | `[GRADE]   HIGH  (6 chunks)` |
| "Final Answer:" detected | `synthesis_start` | `[SYNTH]   Producing final answer…` |
| Budget exhausted | `synthesis_start` | `[BUDGET]  Max iterations reached` |

### What stays the same vs what changes

**Unchanged** (zero edits needed):
- Event protocol / SSE event shapes
- `api.py` POST + GET session endpoints
- All frontend code (`use-agent-stream.ts`, `agent-thoughts-panel.tsx`, `page.tsx`)
- `generate.py` `Answer` with token fields

**Changed / new**:
- `advanced_rag/react_agent.py` (new) — tool schema + `run_retrieve_tool` + `run_react_agent`
- `advanced_rag/gemini.py` — add `chat(messages, tools)` method
- `advanced_rag/agent.py` — add `answer_stream_react()` alongside `answer_stream()`
- `advanced_rag/api.py` — `backend` param becomes `"native" | "react"`

---

## Phase 3 — Visual answer components

Instead of rendering the final answer as plain markdown text in the chat bubble,
the LLM classifies its own output and extracts structured data. The frontend maps
that to the right React component — table, metric card, comparison, etc.

The text answer is always preserved as a fallback. If classification fails or the
answer is genuinely narrative, it renders as markdown as before. This phase touches
nothing in Phase 1 or 2 — it layers on top cleanly.

---

### 3a. Component types

Six component types cover the common shapes of RAG answers over documents:

```typescript
// ui/types/index.ts — add alongside existing types

type AnswerComponent =
  | { type: "text";        data: { content: string } }
  | { type: "metric_card"; data: MetricCardData }
  | { type: "table";       data: TableData }
  | { type: "card_list";   data: CardListData }
  | { type: "comparison";  data: ComparisonData }
  | { type: "timeline";    data: TimelineData }

type MetricCardData = {
  label: string        // "Q3 Revenue"
  value: string        // "$1.82B"
  change?: string      // "+14% YoY"
  context?: string     // "Led by platform segment growth"
}

type TableData = {
  caption?: string
  headers: string[]
  rows: string[][]
}

type CardListData = {
  items: { title: string; body: string }[]
}

type ComparisonData = {
  left_label: string
  right_label: string
  rows: { attribute: string; left: string; right: string }[]
}

type TimelineData = {
  events: { date: string; title: string; description: string }[]
}
```

Add `component?: AnswerComponent` to the existing `Message` type so the bubble
can render it once the `final_answer` SSE event arrives.

---

### 3b. `advanced_rag/generate.py` — component classification

After the text answer is generated, a second lightweight structured call classifies
it and extracts the data. Wrapped in `try/except` so a classification failure
never breaks the answer delivery — it just falls back to `type: "text"`.

```python
# Pydantic models (add to generate.py or a new components.py)

class ComponentType(str, Enum):
    TEXT        = "text"
    METRIC_CARD = "metric_card"
    TABLE       = "table"
    CARD_LIST   = "card_list"
    COMPARISON  = "comparison"
    TIMELINE    = "timeline"

class ComponentPayload(BaseModel):
    type: ComponentType
    data: dict   # validated shape depends on type

# In Generator.generate() — add after the existing generation call:

def _classify_component(self, question: str, answer_text: str) -> ComponentPayload:
    return self.g.generate_structured(
        prompt=(
            f"Question: {question}\n\nAnswer: {answer_text}\n\n"
            "Choose the visual component that best represents this answer and "
            "extract the structured data from the answer text.\n\n"
            "Component types:\n"
            "- text: narrative answer, no structured data worth extracting\n"
            "- metric_card: a single key number/metric (revenue, ratio, count)\n"
            "- table: multiple items with consistent attributes\n"
            "- card_list: list of items each with a title and short description\n"
            "- comparison: side-by-side of exactly two things across several attributes\n"
            "- timeline: sequence of dated events\n\n"
            "If unsure, use text."
        ),
        schema=ComponentPayload,
        temperature=0.0,
    )

# Inside generate():
answer = ...  # existing generation
try:
    answer.component = self._classify_component(question, answer.text)
except Exception:
    answer.component = ComponentPayload(
        type=ComponentType.TEXT, data={"content": answer.text}
    )
```

Add `component: ComponentPayload` to the `Answer` dataclass.

---

### 3c. Event protocol update

Add `component` to the `final_answer` SSE event:

```typescript
| { event: "final_answer";  answer: string; rendered: string;
                             sources: SourceOut[];
                             input_tokens: number; output_tokens: number;
                             component: AnswerComponent }   // ← new
```

`useAgentStream` extracts `component` from the `final_answer` event and returns it
alongside `finalAnswer`. `page.tsx` stores it on the `Message` object.

---

### 3d. `ui/components/answer-renderer.tsx` (new file)

The single switcher component. `MessageBubble` calls this instead of rendering
markdown directly.

```tsx
import { TableAnswer }      from "./answers/table-answer"
import { MetricCardAnswer } from "./answers/metric-card-answer"
import { CardListAnswer }   from "./answers/card-list-answer"
import { ComparisonAnswer } from "./answers/comparison-answer"
import { TimelineAnswer }   from "./answers/timeline-answer"
import { TextAnswer }       from "./answers/text-answer"
import type { AnswerComponent } from "@/types"

export function AnswerRenderer({ component, fallback }: {
  component: AnswerComponent | undefined
  fallback: string   // always the markdown text — rendered if component is absent
}) {
  if (!component || component.type === "text") return <TextAnswer content={fallback} />
  if (component.type === "metric_card")  return <MetricCardAnswer  data={component.data} />
  if (component.type === "table")        return <TableAnswer        data={component.data} />
  if (component.type === "card_list")    return <CardListAnswer     data={component.data} />
  if (component.type === "comparison")   return <ComparisonAnswer   data={component.data} />
  if (component.type === "timeline")     return <TimelineAnswer     data={component.data} />
  return <TextAnswer content={fallback} />
}
```

---

### 3e. Answer component sketches

Each file lives in `ui/components/answers/`.

#### `text-answer.tsx`
Renders the existing `rendered` markdown string. Unchanged behavior, just extracted
into its own file for consistency.

#### `metric-card-answer.tsx`
```
┌──────────────────────────────────┐
│  Q3 Revenue                      │
│  $1.82B          ▲ +14% YoY      │
│  Led by platform segment growth  │
└──────────────────────────────────┘
```
Large value, colored change badge (green/red based on sign), optional context line.

#### `table-answer.tsx`
Standard `<table>` with sticky header, zebra rows, and the existing border/surface
CSS variables. Optional caption above.

#### `card-list-answer.tsx`
```
┌──────────────────┐  ┌──────────────────┐
│ Supply-chain risk│  │ FX exposure      │
│ Disruption in... │  │ 38% of revenue...│
└──────────────────┘  └──────────────────┘
```
Responsive two-column grid of cards matching the existing suggestion card style.

#### `comparison-answer.tsx`
```
┌─────────────────┬────────────┬────────────┐
│                 │   Q3 2025  │   Q2 2025  │
├─────────────────┼────────────┼────────────┤
│ Revenue         │ $1.82B     │ $1.60B     │
│ Operating margin│ 24.2%      │ 22.0%      │
│ Headcount       │ 12 400     │ 12 100     │
└─────────────────┴────────────┴────────────┘
```
First column is the attribute label, next two are left/right values. The higher
value in numeric rows is subtly highlighted.

#### `timeline-answer.tsx`
Vertical timeline with a connecting line, date chip on the left, title + description
on the right. Animates in sequentially on mount.

---

### 3f. `ui/components/message-bubble.tsx` — use `AnswerRenderer`

One-line change — replace the existing markdown render with:

```tsx
<AnswerRenderer component={message.component} fallback={message.content} />
```

Citations stay below the component, exactly as they do today.

---

## File change summary

| File | Action | What changes |
|------|--------|-------------|
| `advanced_rag/gemini.py` | **Edit** | Add `chat(messages, tools)` for multi-turn + tool-calling |
| `advanced_rag/generate.py` | **Edit** | Add `input_tokens`/`output_tokens` + `component` to `Answer`; add `_classify_component()` |
| `advanced_rag/react_agent.py` | **Create** | `RETRIEVE_TOOL_SCHEMA` + `run_retrieve_tool` + `run_react_agent` loop |
| `advanced_rag/agent.py` | **Edit** | Add `answer_stream()` (native) + `answer_stream_react()` — both use `run_in_executor` |
| `advanced_rag/api.py` | **Edit** | Add `POST /api/chat-stream` + `GET /api/chat-stream/{id}`; `backend: "native" \| "react"` |
| `ui/types/index.ts` | **Edit** | Add `AnswerComponent` discriminated union; add `component` to `Message` |
| `ui/lib/use-agent-stream.ts` | **Create** | `useAgentStream` hook — POST then EventSource; extracts `component` from `final_answer` |
| `ui/components/agent-thoughts-panel.tsx` | **Create** | Execution stepper + thought console + token counter |
| `ui/components/answer-renderer.tsx` | **Create** | Switcher — maps `AnswerComponent.type` to the right component |
| `ui/components/answers/text-answer.tsx` | **Create** | Markdown render (existing behavior, extracted) |
| `ui/components/answers/metric-card-answer.tsx` | **Create** | Single KPI card |
| `ui/components/answers/table-answer.tsx` | **Create** | Styled table |
| `ui/components/answers/card-list-answer.tsx` | **Create** | Two-column card grid |
| `ui/components/answers/comparison-answer.tsx` | **Create** | Side-by-side attribute table |
| `ui/components/answers/timeline-answer.tsx` | **Create** | Vertical event timeline |
| `ui/components/message-bubble.tsx` | **Edit** | Replace markdown render with `<AnswerRenderer>` |
| `ui/app/page.tsx` | **Edit** | Wire hook, replace stub, add thoughts panel, store `component` on `Message` |
| `ui/components/top-bar.tsx` | **Edit** | Add thoughts-panel toggle icon |

---

## Scope boundaries

- **Simple mode** (`/ask`) is unchanged — non-streaming endpoint stays as-is.
- **Chat endpoint** (`/chat`) keeps its lock semantics; stream endpoint is stateless.
- `backend=native` is the default — ReAct path is opt-in via `backend=react`.
- Component classification is best-effort — failure silently falls back to `type: "text"`.
- No charting library. All components use plain HTML + Tailwind — no new frontend deps.
- No new Python dependencies — no framework, just the existing Gemini SDK.

---

## Implementation order

1. `gemini.py` — add `chat()` method; test it standalone with a tool schema
2. `generate.py` — add token fields + `ComponentPayload`; implement `_classify_component()`; test standalone
3. `react_agent.py` — implement `run_retrieve_tool` + `run_react_agent`; test with `asyncio.run` and a hardcoded question; verify `event_queue` receives events in the right order
4. `agent.py` `answer_stream()` (native path) — test standalone
5. `agent.py` `answer_stream_react()` — test the queue-drain pattern standalone
6. `api.py` SSE endpoints — verify both `backend=native` and `backend=react` with `curl`
7. `ui/types/index.ts` — add component types
8. `use-agent-stream.ts` — test with a mock `EventSource`
9. `agent-thoughts-panel.tsx` — build with static fixture events first
10. Answer components — build each with hardcoded fixture data, one at a time
11. `answer-renderer.tsx` + `message-bubble.tsx` — wire switcher
12. `page.tsx` + `top-bar.tsx` — final integration
