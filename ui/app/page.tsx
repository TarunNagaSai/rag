"use client";

import { useEffect, useRef, useState } from "react";
import {
  BarChart3,
  FileSearch,
  ScrollText,
  ShieldAlert,
  TrendingUp,
} from "lucide-react";
import { ChatHistory } from "@/components/chat-history";
import { ChatInput } from "@/components/chat-input";
import { MessageBubble } from "@/components/message-bubble";
import { RagPanel } from "@/components/rag-panel";
import { TopBar } from "@/components/top-bar";
import { streamAsk, type StreamSource } from "@/lib/api";
import type { ChatSession, Citation, Message, RagFile } from "@/types";

const HOUR = 60 * 60 * 1000;
const DAY = 24 * HOUR;

const initialSessions: ChatSession[] = [
  {
    id: "s1",
    title: "Q3 earnings review",
    preview: "Revenue grew 14% YoY driven by…",
    updatedAt: Date.now() - 2 * HOUR,
  },
  {
    id: "s2",
    title: "Credit agreement covenants",
    preview: "Leverage ratio must stay below…",
    updatedAt: Date.now() - 3 * DAY,
  },
  {
    id: "s3",
    title: "Risk factor comparison",
    preview: "Supply-chain exposure increased…",
    updatedAt: Date.now() - 12 * DAY,
  },
];

const initialFiles: RagFile[] = [
  {
    id: "f1",
    name: "q3-2025-10q.pdf",
    size: 482_113,
    status: "ready",
    chunks: 214,
  },
  {
    id: "f2",
    name: "credit-agreement.pdf",
    size: 1_240_000,
    status: "ready",
    chunks: 538,
  },
  {
    id: "f3",
    name: "fy24-annual-report.pdf",
    size: 3_180_000,
    status: "processing",
    chunks: 0,
  },
];

const suggestions = [
  {
    icon: BarChart3,
    title: "Summarize Q3 earnings",
    prompt: "Summarize the key takeaways from the Q3 earnings report.",
  },
  {
    icon: ShieldAlert,
    title: "Surface risk factors",
    prompt: "What are the most significant risk factors disclosed this year?",
  },
  {
    icon: ScrollText,
    title: "Extract covenants",
    prompt: "List all financial covenants in the credit agreement.",
  },
  {
    icon: FileSearch,
    title: "Compare segments",
    prompt: "Compare revenue growth across business segments year over year.",
  },
];

function uid() {
  return Math.random().toString(36).slice(2, 10);
}

/** Turn a backend citation string ("file.pdf#12") into a UI Citation. */
function toCitation(src: StreamSource): Citation {
  const [source, loc] = src.citation.split("#");
  const page = loc && /^\d+$/.test(loc) ? Number(loc) : undefined;
  return {
    id: `c${src.n}`,
    source: source || src.citation,
    snippet: "",
    page,
  };
}

export default function Home() {
  const [sessions, setSessions] = useState<ChatSession[]>(initialSessions);
  const [activeId, setActiveId] = useState<string | null>("s1");
  const [messages, setMessages] = useState<Message[]>([]);
  const [files, setFiles] = useState<RagFile[]>(initialFiles);
  const [panelOpen, setPanelOpen] = useState(true);
  const [thinking, setThinking] = useState(false);

  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({
      top: scrollRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [messages]);

  const handleSend = (text: string) => {
    const userMsg: Message = {
      id: uid(),
      role: "user",
      content: text,
      createdAt: Date.now(),
    };
    const pendingId = uid();
    const pendingMsg: Message = {
      id: pendingId,
      role: "assistant",
      content: "",
      createdAt: Date.now() + 1,
      pending: true,
    };
    setMessages((prev) => [...prev, userMsg, pendingMsg]);
    setThinking(true);

    const patch = (fn: (m: Message) => Message) =>
      setMessages((prev) => prev.map((m) => (m.id === pendingId ? fn(m) : m)));

    streamAsk(text, {
      onChunk: (chunk) =>
        // First chunk also clears the typing indicator.
        patch((m) => ({
          ...m,
          pending: false,
          content: m.content + chunk,
        })),
      onSources: (sources) =>
        patch((m) => ({ ...m, citations: sources.map(toCitation) })),
      onDone: () => {
        patch((m) => ({ ...m, pending: false }));
        setThinking(false);
      },
      onError: (err) => {
        patch((m) => ({
          ...m,
          pending: false,
          content:
            m.content ||
            `Sorry — something went wrong reaching the backend.\n\n${
              err instanceof Error ? err.message : String(err)
            }`,
        }));
        setThinking(false);
      },
    });
  };

  const handleNewChat = () => {
    const id = uid();
    setSessions((prev) => [
      { id, title: "New analysis", updatedAt: Date.now() },
      ...prev,
    ]);
    setActiveId(id);
    setMessages([]);
  };

  const handleSelectSession = (id: string) => {
    setActiveId(id);
    setMessages([]);
  };

  const handleAddFiles = (fileList: FileList) => {
    const added: RagFile[] = Array.from(fileList).map((f) => ({
      id: uid(),
      name: f.name,
      size: f.size,
      status: "processing",
      chunks: 0,
    }));
    setFiles((prev) => [...added, ...prev]);
  };

  const handleRemoveFile = (id: string) => {
    setFiles((prev) => prev.filter((f) => f.id !== id));
  };

  const activeTitle =
    sessions.find((s) => s.id === activeId)?.title ?? "New analysis";

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-[var(--background)]">
      <ChatHistory
        sessions={sessions}
        activeId={activeId}
        onSelect={handleSelectSession}
        onNewChat={handleNewChat}
      />

      <main className="flex min-w-0 flex-1 flex-col">
        <TopBar
          title={activeTitle}
          fileCount={files.length}
          panelOpen={panelOpen}
          onTogglePanel={() => setPanelOpen((v) => !v)}
        />

        <div ref={scrollRef} className="flex-1 overflow-y-auto">
          <div className="mx-auto flex max-w-3xl flex-col gap-6 px-4 py-8">
            {messages.length === 0 ? (
              <EmptyState onPick={handleSend} />
            ) : (
              messages.map((m) => <MessageBubble key={m.id} message={m} />)
            )}
          </div>
        </div>

        <ChatInput onSend={handleSend} disabled={thinking} />
      </main>

      {panelOpen && (
        <RagPanel
          files={files}
          onAddFiles={handleAddFiles}
          onRemove={handleRemoveFile}
        />
      )}
    </div>
  );
}

function EmptyState({ onPick }: { onPick: (prompt: string) => void }) {
  return (
    <div className="flex flex-1 flex-col items-center justify-center py-16 text-center">
      <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-[var(--accent)] text-[var(--accent-fg)] shadow-[var(--shadow-md)]">
        <TrendingUp className="h-7 w-7" strokeWidth={2.4} />
      </div>
      <h1 className="mt-5 text-2xl font-semibold tracking-tight">
        What can I help you analyze?
      </h1>
      <p className="mt-2 max-w-md text-sm text-[var(--muted)]">
        Ask questions about your filings, contracts, and reports. Atlas retrieves
        the relevant passages and answers with citations.
      </p>

      <div className="mt-8 grid w-full max-w-2xl grid-cols-1 gap-3 sm:grid-cols-2">
        {suggestions.map((s) => (
          <button
            key={s.title}
            type="button"
            onClick={() => onPick(s.prompt)}
            className="group flex items-start gap-3 rounded-xl border border-[var(--border)] bg-[var(--surface)] p-3.5 text-left shadow-[var(--shadow-sm)] transition-all hover:border-[var(--accent)] hover:shadow-[var(--shadow-md)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--accent-ring)]"
          >
            <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-[var(--accent-soft)] text-[var(--accent)]">
              <s.icon className="h-[18px] w-[18px]" />
            </div>
            <div className="min-w-0">
              <div className="text-sm font-medium text-[var(--foreground)]">
                {s.title}
              </div>
              <div className="mt-0.5 text-xs leading-relaxed text-[var(--muted)]">
                {s.prompt}
              </div>
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}
