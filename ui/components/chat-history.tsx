"use client";

import { useMemo, useState } from "react";
import { MessageSquare, Plus, Search, Settings, TrendingUp } from "lucide-react";
import { cn } from "@/lib/utils";
import type { ChatSession } from "@/types";

interface ChatHistoryProps {
  sessions: ChatSession[];
  activeId: string | null;
  onSelect: (id: string) => void;
  onNewChat: () => void;
}

const DAY = 24 * 60 * 60 * 1000;

/** Bucket sessions into human-readable time groups. */
function groupSessions(sessions: ChatSession[]) {
  const now = Date.now();
  const groups: { label: string; items: ChatSession[] }[] = [
    { label: "Today", items: [] },
    { label: "Previous 7 days", items: [] },
    { label: "Earlier", items: [] },
  ];
  for (const s of [...sessions].sort((a, b) => b.updatedAt - a.updatedAt)) {
    const age = now - s.updatedAt;
    if (age < DAY) groups[0].items.push(s);
    else if (age < 7 * DAY) groups[1].items.push(s);
    else groups[2].items.push(s);
  }
  return groups.filter((g) => g.items.length > 0);
}

export function ChatHistory({
  sessions,
  activeId,
  onSelect,
  onNewChat,
}: ChatHistoryProps) {
  const [query, setQuery] = useState("");

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return sessions;
    return sessions.filter((s) => s.title.toLowerCase().includes(q));
  }, [sessions, query]);

  const groups = useMemo(() => groupSessions(filtered), [filtered]);

  return (
    <aside className="flex h-full w-72 shrink-0 flex-col border-r border-[var(--border)] bg-[var(--surface)]">
      {/* Brand */}
      <div className="flex items-center gap-2.5 px-4 pb-3 pt-4">
        <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-[var(--accent)] text-[var(--accent-fg)] shadow-[var(--shadow-sm)]">
          <TrendingUp className="h-[18px] w-[18px]" strokeWidth={2.4} />
        </div>
        <div className="leading-tight">
          <div className="text-[15px] font-semibold tracking-tight">Atlas</div>
          <div className="text-[11px] font-medium text-[var(--subtle)]">
            Research Copilot
          </div>
        </div>
      </div>

      <div className="px-3">
        <button
          type="button"
          onClick={onNewChat}
          className="flex w-full items-center justify-center gap-2 rounded-lg bg-[var(--accent)] px-3 py-2 text-sm font-medium text-[var(--accent-fg)] shadow-[var(--shadow-sm)] transition-colors hover:bg-[var(--accent-hover)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--accent-ring)]"
        >
          <Plus className="h-4 w-4" strokeWidth={2.4} />
          New analysis
        </button>
      </div>

      {/* Search */}
      <div className="px-3 pb-1 pt-3">
        <div className="flex items-center gap-2 rounded-lg border border-[var(--border)] bg-[var(--surface-2)] px-2.5 py-1.5 focus-within:border-[var(--accent)] focus-within:bg-[var(--surface)]">
          <Search className="h-4 w-4 shrink-0 text-[var(--subtle)]" />
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search conversations"
            className="w-full bg-transparent text-sm text-[var(--foreground)] outline-none placeholder:text-[var(--subtle)]"
          />
        </div>
      </div>

      {/* History */}
      <nav className="flex-1 overflow-y-auto px-2 py-2">
        {groups.length === 0 ? (
          <p className="px-3 py-6 text-center text-sm text-[var(--subtle)]">
            {query ? "No matches found." : "No conversations yet."}
          </p>
        ) : (
          groups.map((group) => (
            <div key={group.label} className="mb-3">
              <div className="px-2 pb-1 pt-2 text-[11px] font-semibold uppercase tracking-wider text-[var(--subtle)]">
                {group.label}
              </div>
              <div className="space-y-0.5">
                {group.items.map((session) => {
                  const active = session.id === activeId;
                  return (
                    <button
                      key={session.id}
                      type="button"
                      onClick={() => onSelect(session.id)}
                      className={cn(
                        "group flex w-full items-start gap-2.5 rounded-lg px-2.5 py-2 text-left transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--accent-ring)]",
                        active
                          ? "bg-[var(--accent-soft)]"
                          : "hover:bg-[var(--surface-2)]"
                      )}
                    >
                      <MessageSquare
                        className={cn(
                          "mt-0.5 h-4 w-4 shrink-0",
                          active ? "text-[var(--accent)]" : "text-[var(--subtle)]"
                        )}
                      />
                      <span className="min-w-0 flex-1">
                        <span
                          className={cn(
                            "block truncate text-sm",
                            active
                              ? "font-medium text-[var(--accent-hover)]"
                              : "text-[var(--foreground)]"
                          )}
                        >
                          {session.title}
                        </span>
                        {session.preview && (
                          <span className="block truncate text-xs text-[var(--subtle)]">
                            {session.preview}
                          </span>
                        )}
                      </span>
                    </button>
                  );
                })}
              </div>
            </div>
          ))
        )}
      </nav>

      {/* User footer */}
      <div className="border-t border-[var(--border)] p-2">
        <button
          type="button"
          className="flex w-full items-center gap-2.5 rounded-lg px-2 py-2 text-left transition-colors hover:bg-[var(--surface-2)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--accent-ring)]"
        >
          <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-[var(--surface-3)] text-xs font-semibold text-[var(--muted)]">
            TK
          </div>
          <div className="min-w-0 flex-1 leading-tight">
            <div className="truncate text-sm font-medium">Tarun Kodali</div>
            <div className="truncate text-xs text-[var(--subtle)]">
              Pro workspace
            </div>
          </div>
          <Settings className="h-4 w-4 shrink-0 text-[var(--subtle)]" />
        </button>
      </div>
    </aside>
  );
}
