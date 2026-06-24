"use client";

import MarkdownIt from "markdown-it";
import { Copy, FileText, TrendingUp } from "lucide-react";
import { cn } from "@/lib/utils";
import type { Message } from "@/types";

const md = new MarkdownIt({ linkify: true, breaks: true });

export function MessageBubble({ message }: { message: Message }) {
  const isUser = message.role === "user";

  if (isUser) {
    return (
      <div className="flex animate-rise justify-end">
        <div className="max-w-[80%] rounded-2xl rounded-br-md bg-[var(--accent)] px-4 py-2.5 text-sm leading-relaxed text-[var(--accent-fg)] shadow-[var(--shadow-sm)]">
          <p className="whitespace-pre-wrap">{message.content}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex animate-rise gap-3">
      <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-[var(--accent-soft)] text-[var(--accent)] ring-1 ring-[var(--border)]">
        <TrendingUp className="h-4 w-4" strokeWidth={2.4} />
      </div>

      <div className="group min-w-0 flex-1">
        <div className="mb-1 flex items-center gap-2">
          <span className="text-sm font-semibold tracking-tight">Atlas</span>
          <span className="rounded bg-[var(--surface-2)] px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wide text-[var(--subtle)]">
            Grounded
          </span>
        </div>

        {message.pending ? (
          <TypingIndicator />
        ) : (
          <div
            className="markdown"
            // markdown-it output is sanitised (no HTML input from users)
            dangerouslySetInnerHTML={{ __html: md.render(message.content) }}
          />
        )}

        {message.citations && message.citations.length > 0 && (
          <Citations message={message} />
        )}

        {!message.pending && (
          <div className="mt-2 flex items-center gap-1 opacity-0 transition-opacity group-hover:opacity-100">
            <button
              type="button"
              onClick={() => navigator.clipboard?.writeText(message.content)}
              className="flex items-center gap-1 rounded-md px-1.5 py-1 text-xs text-[var(--subtle)] transition-colors hover:bg-[var(--surface-2)] hover:text-[var(--foreground)]"
            >
              <Copy className="h-3.5 w-3.5" />
              Copy
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

function Citations({ message }: { message: Message }) {
  return (
    <div className="mt-3 rounded-xl border border-[var(--border)] bg-[var(--surface-2)] p-3">
      <div className="mb-2 text-[11px] font-semibold uppercase tracking-wider text-[var(--subtle)]">
        Sources
      </div>
      <div className="space-y-1.5">
        {message.citations!.map((c, i) => (
          <div
            key={c.id}
            className="flex items-start gap-2 rounded-lg border border-[var(--border)] bg-[var(--surface)] px-2.5 py-2"
          >
            <span className="mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-md bg-[var(--accent-soft)] text-[10px] font-semibold text-[var(--accent)]">
              {i + 1}
            </span>
            <div className="min-w-0 flex-1">
              <div className="flex items-center gap-1.5 text-xs font-medium text-[var(--foreground)]">
                <FileText className="h-3.5 w-3.5 shrink-0 text-[var(--accent)]" />
                <span className="truncate">{c.source}</span>
                {c.page != null && (
                  <span className="shrink-0 text-[var(--subtle)]">
                    · p.{c.page}
                  </span>
                )}
              </div>
              <p className="mt-0.5 line-clamp-2 text-xs leading-relaxed text-[var(--muted)]">
                {c.snippet}
              </p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function TypingIndicator() {
  return (
    <div className="flex items-center gap-1.5 py-1">
      <span className={cn("dot h-2 w-2 rounded-full bg-[var(--subtle)]")} />
      <span className={cn("dot h-2 w-2 rounded-full bg-[var(--subtle)]")} />
      <span className={cn("dot h-2 w-2 rounded-full bg-[var(--subtle)]")} />
    </div>
  );
}
