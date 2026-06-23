/**
 * Client for the Advanced RAG backend.
 *
 * The base URL is configurable via NEXT_PUBLIC_API_URL so the same build can
 * point at a local dev server (default) or a deployed API.
 */
const API_URL =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") ?? "http://localhost:8000";

/** A grounding source as emitted by the backend's SSE `sources` event. */
export interface StreamSource {
  n: number;
  citation: string;
  cited: boolean;
}

export interface StreamHandlers {
  /** Called with each text fragment as it arrives from the model. */
  onChunk: (text: string) => void;
  /** Called once after the text stream ends, with the citation list. */
  onSources?: (sources: StreamSource[]) => void;
  /** Called when the stream finishes cleanly ([DONE] received). */
  onDone?: () => void;
  /** Called on transport/parse failure. */
  onError?: (err: unknown) => void;
}

/**
 * POST /ask/stream and dispatch Server-Sent Events to the supplied handlers.
 *
 * Returns once the stream is exhausted. Pass an AbortSignal to cancel early
 * (e.g. when the component unmounts or the user starts a new question).
 */
export async function streamAsk(
  question: string,
  handlers: StreamHandlers,
  opts: { signal?: AbortSignal } = {}
): Promise<void> {
  const { onChunk, onSources, onDone, onError } = handlers;
  try {
    const res = await fetch(`${API_URL}/ask/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, mode: "simple" }),
      signal: opts.signal,
    });

    if (!res.ok || !res.body) {
      const detail = await res.text().catch(() => res.statusText);
      throw new Error(`Request failed (${res.status}): ${detail}`);
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      // SSE frames are separated by a blank line.
      let sep: number;
      while ((sep = buffer.indexOf("\n\n")) !== -1) {
        const frame = buffer.slice(0, sep);
        buffer = buffer.slice(sep + 2);

        const line = frame
          .split("\n")
          .find((l) => l.startsWith("data:"));
        if (!line) continue;

        const payload = line.slice(5).trim();
        if (payload === "[DONE]") {
          onDone?.();
          return;
        }

        try {
          const event = JSON.parse(payload);
          if (event.type === "chunk") onChunk(event.text as string);
          else if (event.type === "sources") onSources?.(event.sources);
        } catch {
          // Ignore malformed frames rather than aborting the whole stream.
        }
      }
    }
    onDone?.();
  } catch (err) {
    if ((err as Error)?.name === "AbortError") return;
    onError?.(err);
  }
}
