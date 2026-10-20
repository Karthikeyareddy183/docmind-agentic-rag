"use client";

import { ArrowUp, FileSearch, Loader2, Square } from "lucide-react";
import { useEffect, useRef, useState } from "react";

import AnswerView from "@/components/AnswerView";
import type { ViewerTarget } from "@/components/DocViewer";
import { api, postSse } from "@/lib/api";
import type { ChatMessage, Citation, Computation, Step, StoredMessage } from "@/lib/types";

const EXAMPLES = [
  "What was total revenue in FY2025, and how did it grow?",
  "What percentage of revenue came from the Cloud Platform?",
  "Summarize the key decisions in the latest board minutes.",
  "How many days of annual leave do employees get?",
];

interface Props {
  conversationId: string | null;
  readyCount: number;
  onConversationCreated: (id: string) => void;
  onOpenCitation: (target: ViewerTarget) => void;
}

function fromStored(m: StoredMessage): ChatMessage {
  return {
    id: m.id,
    role: m.role,
    content: m.content,
    citations: m.citations,
    confidence: m.confidence,
    steps: m.trace?.map((t) => ({ ...t, type: "step" as const })) ?? [],
    usage: m.usage,
    latency_ms: m.latency_ms,
  };
}

export default function ChatPanel({ conversationId, readyCount, onConversationCreated, onOpenCitation }: Props) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  // A conversation created by this panel already has its messages in local state
  const createdHere = useRef<string | null>(null);

  useEffect(() => {
    if (!conversationId || conversationId === createdHere.current) return;
    let cancelled = false;
    api<{ messages: StoredMessage[] }>(`/conversations/${conversationId}`).then((c) => {
      if (!cancelled) setMessages(c.messages.map(fromStored));
    });
    return () => {
      cancelled = true;
    };
  }, [conversationId]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages]);

  // Leaving this conversation cancels any answer still streaming into it
  useEffect(() => () => abortRef.current?.abort(), []);

  function patchLast(update: (m: ChatMessage) => ChatMessage) {
    setMessages((ms) => [...ms.slice(0, -1), update(ms[ms.length - 1])]);
  }

  async function send(text: string) {
    const question = text.trim();
    if (!question || busy) return;
    setInput("");
    setBusy(true);
    const controller = new AbortController();
    abortRef.current = controller;

    setMessages((ms) => [
      ...ms,
      { id: crypto.randomUUID(), role: "user", content: question, citations: [], confidence: null, steps: [] },
      { id: crypto.randomUUID(), role: "assistant", content: "", citations: [], confidence: null, steps: [], streaming: true },
    ]);

    try {
      let id = conversationId;
      if (!id) {
        const conv = await api<{ id: string }>("/conversations", { method: "POST", body: JSON.stringify({}) });
        id = conv.id;
        createdHere.current = id;
        onConversationCreated(id);
      }
      for await (const { event, data } of postSse(`/conversations/${id}/messages`, { content: question }, controller.signal)) {
        const payload = data as Record<string, unknown>;
        if (event === "token") {
          patchLast((m) => ({ ...m, content: m.content + String(payload.text ?? "") }));
        } else if (event === "step") {
          patchLast((m) => ({ ...m, steps: [...m.steps, payload as unknown as Step] }));
        } else if (event === "revision") {
          // The fact-check rejected the draft; the revised answer streams next
          patchLast((m) => ({ ...m, content: "", revised: true }));
        } else if (event === "final") {
          patchLast((m) => ({
            ...m,
            id: String(payload.message_id ?? m.id),
            content: String(payload.answer ?? m.content),
            citations: (payload.citations as Citation[]) ?? [],
            confidence: (payload.confidence as number) ?? null,
            verified: payload.verified as boolean | null,
            computations: (payload.computations as Computation[]) ?? [],
            usage: payload.usage as ChatMessage["usage"],
            latency_ms: payload.latency_ms as number,
            streaming: false,
          }));
        } else if (event === "error") {
          patchLast((m) => ({ ...m, streaming: false, error: String(payload.message ?? "Something went wrong") }));
        }
      }
    } catch (err) {
      const aborted = err instanceof DOMException && err.name === "AbortError";
      patchLast((m) => ({ ...m, streaming: false, error: aborted ? "Stopped." : "Could not reach the assistant." }));
    } finally {
      patchLast((m) => ({ ...m, streaming: false }));
      setBusy(false);
      abortRef.current = null;
    }
  }

  return (
    <div className="flex h-full flex-col">
      <div className="min-h-0 flex-1 overflow-y-auto">
        <div className="mx-auto max-w-3xl px-6 py-8">
          {messages.length === 0 ? (
            <div className="mt-16 text-center">
              <FileSearch className="mx-auto h-10 w-10 text-brand-500" />
              <h1 className="mt-4 text-xl font-semibold">Ask anything about your documents</h1>
              <p className="mt-1 text-sm text-slate-500">
                {readyCount
                  ? `${readyCount} document${readyCount === 1 ? "" : "s"} ready. Answers cite the exact page and region.`
                  : "Upload a PDF, scan, or image to begin."}
              </p>
              {readyCount > 0 && (
                <div className="mt-6 grid gap-2 sm:grid-cols-2">
                  {EXAMPLES.map((q) => (
                    <button
                      key={q}
                      onClick={() => send(q)}
                      className="rounded-xl border border-slate-200 bg-white px-4 py-3 text-left text-sm text-slate-600 shadow-sm transition hover:border-brand-500 hover:text-slate-900"
                    >
                      {q}
                    </button>
                  ))}
                </div>
              )}
            </div>
          ) : (
            <div className="space-y-6">
              {messages.map((m) =>
                m.role === "user" ? (
                  <div key={m.id} className="flex justify-end">
                    <div className="max-w-[80%] rounded-2xl rounded-br-md bg-brand-600 px-4 py-2.5 text-sm text-white">
                      {m.content}
                    </div>
                  </div>
                ) : (
                  <AnswerView key={m.id} message={m} onOpenCitation={onOpenCitation} />
                ),
              )}
              <div ref={bottomRef} />
            </div>
          )}
        </div>
      </div>

      <div className="border-t border-slate-200 bg-white/80 px-6 py-4 backdrop-blur">
        <form
          onSubmit={(e) => {
            e.preventDefault();
            void send(input);
          }}
          className="mx-auto flex max-w-3xl items-end gap-2 rounded-2xl border border-slate-300 bg-white px-3 py-2 shadow-sm focus-within:border-brand-500 focus-within:ring-2 focus-within:ring-brand-100"
        >
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                void send(input);
              }
            }}
            rows={1}
            placeholder={readyCount ? "Ask a question about your documents…" : "Upload a document first…"}
            className="max-h-40 min-h-[2.25rem] flex-1 resize-none bg-transparent py-1.5 text-sm outline-none"
          />
          {busy ? (
            <button
              type="button"
              onClick={() => abortRef.current?.abort()}
              className="flex h-9 w-9 items-center justify-center rounded-xl bg-slate-800 text-white"
              title="Stop"
            >
              <Square className="h-3.5 w-3.5" />
            </button>
          ) : (
            <button
              type="submit"
              disabled={!input.trim()}
              className="flex h-9 w-9 items-center justify-center rounded-xl bg-brand-600 text-white transition hover:bg-brand-700 disabled:opacity-40"
              title="Send"
            >
              {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <ArrowUp className="h-4 w-4" />}
            </button>
          )}
        </form>
        <p className="mx-auto mt-2 max-w-3xl text-center text-[11px] text-slate-400">
          Answers are generated from your documents and fact-checked against the cited passages. Verify important figures.
        </p>
      </div>
    </div>
  );
}
