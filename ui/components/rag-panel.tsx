"use client";

import { useRef, useState } from "react";
import {
  AlertCircle,
  CheckCircle2,
  FileText,
  Loader2,
  Trash2,
  UploadCloud,
} from "lucide-react";
import { cn } from "@/lib/utils";
import type { RagFile } from "@/types";

interface RagPanelProps {
  files: RagFile[];
  onAddFiles: (files: FileList) => void;
  onRemove: (id: string) => void;
}

function formatSize(bytes: number) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export function RagPanel({ files, onAddFiles, onRemove }: RagPanelProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);

  const totalChunks = files.reduce((sum, f) => sum + (f.chunks ?? 0), 0);
  const ready = files.filter((f) => f.status === "ready").length;

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    if (e.dataTransfer.files?.length) onAddFiles(e.dataTransfer.files);
  };

  return (
    <aside className="flex h-full w-80 shrink-0 flex-col border-l border-[var(--border)] bg-[var(--surface)]">
      <div className="border-b border-[var(--border)] px-4 py-4">
        <h2 className="text-sm font-semibold tracking-tight">Knowledge base</h2>
        <p className="mt-0.5 text-xs text-[var(--subtle)]">
          Documents indexed for retrieval
        </p>

        {/* Stat strip */}
        <div className="mt-3 grid grid-cols-2 gap-2">
          <Stat label="Sources" value={`${ready}/${files.length}`} />
          <Stat label="Chunks" value={totalChunks.toLocaleString()} />
        </div>
      </div>

      {/* Dropzone */}
      <div className="px-4 pt-4">
        <button
          type="button"
          onClick={() => inputRef.current?.click()}
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={handleDrop}
          className={cn(
            "flex w-full flex-col items-center gap-2 rounded-xl border border-dashed px-4 py-6 text-center transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--accent-ring)]",
            dragging
              ? "border-[var(--accent)] bg-[var(--accent-soft)]"
              : "border-[var(--border-strong)] bg-[var(--surface-2)] hover:border-[var(--accent)] hover:bg-[var(--accent-soft)]"
          )}
        >
          <div className="flex h-10 w-10 items-center justify-center rounded-full bg-[var(--surface)] text-[var(--accent)] shadow-[var(--shadow-sm)]">
            <UploadCloud className="h-5 w-5" />
          </div>
          <span className="text-sm font-medium text-[var(--foreground)]">
            Drop files or browse
          </span>
          <span className="text-xs text-[var(--subtle)]">
            PDF, TXT, MD, DOCX · up to 25 MB
          </span>
        </button>
        <input
          ref={inputRef}
          type="file"
          multiple
          accept=".pdf,.txt,.md,.docx"
          className="hidden"
          onChange={(e) => {
            if (e.target.files?.length) onAddFiles(e.target.files);
            e.target.value = "";
          }}
        />
      </div>

      <div className="px-4 pb-1 pt-4 text-[11px] font-semibold uppercase tracking-wider text-[var(--subtle)]">
        Documents
      </div>

      <div className="flex-1 space-y-1.5 overflow-y-auto px-3 py-2">
        {files.length === 0 ? (
          <div className="px-3 py-8 text-center">
            <FileText className="mx-auto h-6 w-6 text-[var(--subtle)]" />
            <p className="mt-2 text-sm text-[var(--muted)]">No documents yet</p>
            <p className="text-xs text-[var(--subtle)]">
              Upload sources to ground answers.
            </p>
          </div>
        ) : (
          files.map((file) => (
            <div
              key={file.id}
              className="group flex items-center gap-2.5 rounded-lg border border-[var(--border)] bg-[var(--surface)] px-3 py-2.5 transition-colors hover:border-[var(--border-strong)]"
            >
              <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-[var(--surface-2)] text-[var(--accent)]">
                <FileText className="h-4 w-4" />
              </div>
              <div className="min-w-0 flex-1">
                <p className="truncate text-sm font-medium text-[var(--foreground)]">
                  {file.name}
                </p>
                <p className="flex items-center gap-1.5 text-xs text-[var(--subtle)]">
                  <span className="tabular">{formatSize(file.size)}</span>
                  {file.chunks != null && (
                    <>
                      <span>·</span>
                      <span className="tabular">{file.chunks} chunks</span>
                    </>
                  )}
                </p>
              </div>
              <StatusIcon status={file.status} />
              <button
                type="button"
                onClick={() => onRemove(file.id)}
                aria-label={`Remove ${file.name}`}
                className="text-[var(--subtle)] opacity-0 transition-opacity hover:text-[var(--negative)] focus-visible:opacity-100 group-hover:opacity-100"
              >
                <Trash2 className="h-4 w-4" />
              </button>
            </div>
          ))
        )}
      </div>
    </aside>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-[var(--border)] bg-[var(--surface-2)] px-2.5 py-2">
      <div className="tabular text-base font-semibold tracking-tight">
        {value}
      </div>
      <div className="text-[11px] font-medium text-[var(--subtle)]">{label}</div>
    </div>
  );
}

function StatusIcon({ status }: { status: RagFile["status"] }) {
  if (status === "processing") {
    return (
      <Loader2
        className="h-4 w-4 shrink-0 animate-spin text-[var(--warning)]"
        aria-label="Processing"
      />
    );
  }
  if (status === "error") {
    return (
      <AlertCircle
        className="h-4 w-4 shrink-0 text-[var(--negative)]"
        aria-label="Error"
      />
    );
  }
  return (
    <CheckCircle2
      className="h-4 w-4 shrink-0 text-[var(--positive)]"
      aria-label="Ready"
    />
  );
}
