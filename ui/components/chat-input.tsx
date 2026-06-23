"use client";

import { useRef, useState } from "react";
import { ArrowUp, Paperclip } from "lucide-react";

interface ChatInputProps {
  onSend: (text: string) => void;
  disabled?: boolean;
}

export function ChatInput({ onSend, disabled }: ChatInputProps) {
  const [value, setValue] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const submit = () => {
    const text = value.trim();
    if (!text || disabled) return;
    onSend(text);
    setValue("");
    if (textareaRef.current) textareaRef.current.style.height = "auto";
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  };

  const autoGrow = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setValue(e.target.value);
    const el = e.target;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`;
  };

  const canSend = !!value.trim() && !disabled;

  return (
    <div className="bg-[var(--background)] px-4 pb-4 pt-2">
      <div className="mx-auto max-w-3xl">
        <div className="flex items-end gap-2 rounded-2xl border border-[var(--border)] bg-[var(--surface)] p-2 shadow-[var(--shadow-md)] transition-colors focus-within:border-[var(--accent)]">
          <button
            type="button"
            aria-label="Attach files"
            className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl text-[var(--subtle)] transition-colors hover:bg-[var(--surface-2)] hover:text-[var(--foreground)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--accent-ring)]"
          >
            <Paperclip className="h-[18px] w-[18px]" />
          </button>

          <textarea
            ref={textareaRef}
            value={value}
            onChange={autoGrow}
            onKeyDown={handleKeyDown}
            rows={1}
            placeholder="Ask about your filings, contracts, or reports…"
            className="max-h-[200px] flex-1 resize-none bg-transparent py-2 text-sm text-[var(--foreground)] outline-none placeholder:text-[var(--subtle)]"
          />

          <button
            type="button"
            onClick={submit}
            disabled={!canSend}
            aria-label="Send message"
            className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-[var(--accent)] text-[var(--accent-fg)] shadow-[var(--shadow-sm)] transition-all hover:bg-[var(--accent-hover)] disabled:cursor-not-allowed disabled:bg-[var(--surface-3)] disabled:text-[var(--subtle)] disabled:shadow-none focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--accent-ring)]"
          >
            <ArrowUp className="h-[18px] w-[18px]" strokeWidth={2.4} />
          </button>
        </div>

        <p className="mt-2 text-center text-xs text-[var(--subtle)]">
          Atlas grounds answers in your sources. Verify figures before relying on
          them.
        </p>
      </div>
    </div>
  );
}
