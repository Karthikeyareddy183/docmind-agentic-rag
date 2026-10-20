"use client";

import clsx from "clsx";
import {
  BadgeCheck,
  Brain,
  Calculator,
  ChevronDown,
  FileText,
  Search,
  ShieldAlert,
  ShieldCheck,
  ThumbsDown,
  ThumbsUp,
  Wand2,
} from "lucide-react";
import { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import type { ViewerTarget } from "@/components/DocViewer";
import { api } from "@/lib/api";
import type { ChatMessage, Citation, Step } from "@/lib/types";

/**
 * Turn citation groups into markdown links the renderer swaps for clickable chips.
 * Handles "[S1, S3]" and decorated forms seen mid-stream, e.g. "[S1, page 3]" or "【S1】".
 */
function linkCitations(text: string): string {
  return text.replace(/[[【]([^[\]【】]*?\bS\d+\b[^[\]【】]*?)[\]】]/g, (_, group: string) =>
    [...new Set(group.match(/\bS\d+\b/g) ?? [])].map((id) => `[${id}](#cite-${id})`).join(""),
  );
}

function confidenceStyle(c: number) {
  if (c >= 0.75) return { label: "High confidence", cls: "bg-emerald-50 text-emerald-700 ring-emerald-200" };
  if (c >= 0.5) return { label: "Medium confidence", cls: "bg-amber-50 text-amber-700 ring-amber-200" };
  return { label: "Low confidence", cls: "bg-red-50 text-red-700 ring-red-200" };
}

function stepView(step: Step): { icon: React.ReactNode; text: string } | null {
  const s = step as Record<string, unknown>;
  switch (step.name) {
    case "understand":
      return { icon: <Brain className="h-3.5 w-3.5" />, text: `Understood as: “${s.standalone}”` };
    case "plan": {
      const tools = (s.tools as string[]) ?? [];
      if (tools.length === 1 && tools[0] === "done") return { icon: <Brain className="h-3.5 w-3.5" />, text: "Enough evidence gathered" };
      return { icon: <Brain className="h-3.5 w-3.5" />, text: `Planned: ${tools.map((t) => t.split("(")[0]).join(", ")}` };
    }
    case "search":
      return {
        icon: <Search className="h-3.5 w-3.5" />,
        text: `Searched “${s.query}” → ${s.hits} passages (top relevance ${Number(s.top_score ?? 0).toFixed(2)})`,
      };
    case "calculate":
      return { icon: <Calculator className="h-3.5 w-3.5" />, text: `Calculated ${s.expression}` };
    case "summarize":
      return { icon: <FileText className="h-3.5 w-3.5" />, text: `Read ${s.passages} passages of ${s.document}` };
    case "generate":
      return s.outcome === "not_found" ? { icon: <Wand2 className="h-3.5 w-3.5" />, text: "No relevant evidence found" } : null;
    case "revise":
      return { icon: <Wand2 className="h-3.5 w-3.5" />, text: "Rewrote the answer after fact-check" };
    case "verify":
      if (s.outcome === "skipped") return { icon: <ShieldAlert className="h-3.5 w-3.5" />, text: "Fact-check skipped" };
      return s.supported
        ? { icon: <ShieldCheck className="h-3.5 w-3.5" />, text: "Fact-check passed: every claim is supported by its source" }
        : { icon: <ShieldAlert className="h-3.5 w-3.5" />, text: `Fact-check flagged: ${((s.problems as string[]) ?? []).join("; ")}` };
    default:
      return null;
  }
}

interface Props {
  message: ChatMessage;
  onOpenCitation: (target: ViewerTarget) => void;
}

export default function AnswerView({ message, onOpenCitation }: Props) {
  const [showSteps, setShowSteps] = useState(false);
  const [rating, setRating] = useState<1 | -1 | null>(null);
  const byId = new Map(message.citations.map((c) => [c.id, c]));
  const steps = message.steps.map(stepView).filter(Boolean) as { icon: React.ReactNode; text: string }[];

  function open(c: Citation) {
    onOpenCitation({
      documentId: c.document_id,
      page: c.page,
      bboxes: c.bboxes.filter((b) => b.page === c.page || c.bboxes.length === 1),
      title: c.document_title,
      contentType: "",
      label: c.id,
    });
  }

  async function rate(value: 1 | -1) {
    setRating(value);
    try {
      await api(`/messages/${message.id}/feedback`, { method: "POST", body: JSON.stringify({ rating: value }) });
    } catch {
      setRating(null);
    }
  }

  return (
    <div className="rounded-2xl border border-slate-200 bg-white px-5 py-4 shadow-sm">
      {steps.length > 0 && (
        <button
          onClick={() => setShowSteps((v) => !v)}
          className="mb-2 flex items-center gap-1.5 text-xs font-medium text-slate-500 hover:text-slate-700"
        >
          {message.streaming ? (
            <span className="h-2 w-2 animate-pulse rounded-full bg-brand-500" />
          ) : (
            <ChevronDown className={clsx("h-3.5 w-3.5 transition", showSteps && "rotate-180")} />
          )}
          {message.streaming ? steps[steps.length - 1]?.text : `How I answered · ${steps.length} steps`}
        </button>
      )}
      {showSteps && (
        <ol className="mb-3 space-y-1.5 border-l-2 border-slate-100 pl-3">
          {steps.map((s, i) => (
            <li key={i} className="flex items-start gap-2 text-xs text-slate-500">
              <span className="mt-0.5 text-slate-400">{s.icon}</span>
              <span>{s.text}</span>
            </li>
          ))}
        </ol>
      )}

      {message.revised && message.streaming && (
        <p className="mb-2 text-xs text-amber-700">Fact-check found an issue — revising the answer…</p>
      )}

      <div className={clsx("prose-answer text-[15px] text-slate-800", message.streaming && "caret")}>
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          components={{
            a: ({ href, children }) => {
              const id = href?.startsWith("#cite-") ? href.slice(6) : null;
              const c = id ? byId.get(id) : undefined;
              if (!id) return <a href={href}>{children}</a>;
              return (
                <button
                  onClick={() => c && open(c)}
                  disabled={!c}
                  title={c ? `${c.document_title} — page ${c.page}` : id}
                  className="mx-0.5 inline-flex -translate-y-px items-center rounded-md bg-brand-50 px-1.5 py-px align-baseline text-[11px] font-semibold text-brand-700 ring-1 ring-brand-100 transition hover:bg-brand-100"
                >
                  {c ? `p.${c.page}` : id}
                </button>
              );
            },
          }}
        >
          {linkCitations(message.content)}
        </ReactMarkdown>
      </div>

      {message.error && <p className="mt-2 text-sm text-red-600">{message.error}</p>}

      {!message.streaming && message.citations.length > 0 && (
        <div className="mt-4 grid gap-2 sm:grid-cols-2">
          {message.citations.map((c) => (
            <button
              key={c.id}
              onClick={() => open(c)}
              className="rounded-xl border border-slate-200 px-3 py-2 text-left transition hover:border-brand-500 hover:bg-brand-50/40"
            >
              <div className="flex items-center justify-between gap-2">
                <span className="truncate text-xs font-semibold text-slate-700">{c.document_title}</span>
                <span className="shrink-0 rounded bg-slate-100 px-1.5 py-px text-[10px] font-medium text-slate-500">
                  p.{c.page} · {c.chunk_type}
                </span>
              </div>
              <p className="mt-1 line-clamp-2 text-xs text-slate-500">{c.snippet}</p>
            </button>
          ))}
        </div>
      )}

      {!message.streaming && message.confidence !== null && (
        <div className="mt-4 flex flex-wrap items-center gap-2 border-t border-slate-100 pt-3 text-[11px]">
          {message.confidence > 0 && (
            <span className={clsx("rounded-full px-2 py-0.5 font-medium ring-1", confidenceStyle(message.confidence).cls)}>
              {confidenceStyle(message.confidence).label} · {Math.round(message.confidence * 100)}%
            </span>
          )}
          {message.verified === true && (
            <span className="inline-flex items-center gap-1 rounded-full bg-emerald-50 px-2 py-0.5 font-medium text-emerald-700 ring-1 ring-emerald-200">
              <BadgeCheck className="h-3 w-3" /> Fact-checked
            </span>
          )}
          {(message.computations?.length ?? 0) > 0 && (
            <span className="inline-flex items-center gap-1 rounded-full bg-slate-100 px-2 py-0.5 font-medium text-slate-600">
              <Calculator className="h-3 w-3" /> Exact calculation
            </span>
          )}
          <span className="text-slate-400">
            {message.latency_ms ? `${(message.latency_ms / 1000).toFixed(1)}s` : ""}
            {message.usage?.total_tokens ? ` · ${message.usage.total_tokens.toLocaleString()} tokens` : ""}
            {message.usage?.cost_usd ? ` · $${message.usage.cost_usd.toFixed(4)}` : ""}
          </span>
          <span className="ml-auto flex gap-1">
            <button
              onClick={() => rate(1)}
              className={clsx("rounded p-1 hover:bg-slate-100", rating === 1 ? "text-emerald-600" : "text-slate-400")}
              title="Helpful"
            >
              <ThumbsUp className="h-3.5 w-3.5" />
            </button>
            <button
              onClick={() => rate(-1)}
              className={clsx("rounded p-1 hover:bg-slate-100", rating === -1 ? "text-red-600" : "text-slate-400")}
              title="Not helpful"
            >
              <ThumbsDown className="h-3.5 w-3.5" />
            </button>
          </span>
        </div>
      )}
    </div>
  );
}
