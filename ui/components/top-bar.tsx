"use client";

import { ChevronDown, Database, PanelRight, Share2, Sparkles } from "lucide-react";
import { cn } from "@/lib/utils";

interface TopBarProps {
  title: string;
  fileCount: number;
  panelOpen: boolean;
  onTogglePanel: () => void;
}

export function TopBar({
  title,
  fileCount,
  panelOpen,
  onTogglePanel,
}: TopBarProps) {
  return (
    <header className="flex h-14 shrink-0 items-center justify-between border-b border-[var(--border)] bg-[var(--surface)] px-4">
      <div className="flex min-w-0 items-center gap-3">
        <h1 className="truncate text-sm font-semibold tracking-tight">
          {title}
        </h1>
        <span className="hidden items-center gap-1.5 rounded-full border border-[var(--border)] bg-[var(--surface-2)] px-2.5 py-1 text-xs font-medium text-[var(--muted)] sm:inline-flex">
          <Database className="h-3.5 w-3.5 text-[var(--accent)]" />
          {fileCount} {fileCount === 1 ? "source" : "sources"}
        </span>
      </div>

      <div className="flex items-center gap-2">
        {/* Model selector */}
        <button
          type="button"
          className="hidden items-center gap-1.5 rounded-lg border border-[var(--border)] bg-[var(--surface)] px-2.5 py-1.5 text-xs font-medium text-[var(--foreground)] transition-colors hover:bg-[var(--surface-2)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--accent-ring)] md:inline-flex"
        >
          <Sparkles className="h-3.5 w-3.5 text-[var(--accent)]" />
          Gemini 2.5 Pro
          <ChevronDown className="h-3.5 w-3.5 text-[var(--subtle)]" />
        </button>

        <button
          type="button"
          aria-label="Share conversation"
          className="flex h-8 w-8 items-center justify-center rounded-lg border border-[var(--border)] text-[var(--muted)] transition-colors hover:bg-[var(--surface-2)] hover:text-[var(--foreground)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--accent-ring)]"
        >
          <Share2 className="h-4 w-4" />
        </button>

        <button
          type="button"
          onClick={onTogglePanel}
          aria-label="Toggle knowledge base"
          aria-pressed={panelOpen}
          className={cn(
            "flex h-8 w-8 items-center justify-center rounded-lg border transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--accent-ring)]",
            panelOpen
              ? "border-[var(--accent)] bg-[var(--accent-soft)] text-[var(--accent)]"
              : "border-[var(--border)] text-[var(--muted)] hover:bg-[var(--surface-2)] hover:text-[var(--foreground)]"
          )}
        >
          <PanelRight className="h-4 w-4" />
        </button>
      </div>
    </header>
  );
}
