"use client";

import clsx from "clsx";
import {
  AlertCircle,
  CheckCircle2,
  FileImage,
  FileSearch,
  FileText,
  Loader2,
  LogOut,
  MessageSquare,
  Plus,
  RotateCw,
  Trash2,
  Upload,
} from "lucide-react";
import { useRouter } from "next/navigation";
import { useRef, useState } from "react";

import { ApiError, api } from "@/lib/api";
import type { Conversation, DocumentItem } from "@/lib/types";

const STAGE_LABEL: Record<string, string> = {
  parsing: "Reading layout & OCR",
  summarizing: "Summarizing",
  enriching: "Describing tables & charts",
  embedding: "Embedding",
  indexing: "Indexing",
};

interface Props {
  email: string;
  documents: DocumentItem[];
  conversations: Conversation[];
  activeConversation: string | null;
  onSelectConversation: (id: string) => void;
  onNewConversation: () => void;
  onDocumentsChanged: () => Promise<void>;
  onConversationsChanged: () => Promise<void>;
  onOpenDocument: (doc: DocumentItem) => void;
}

export default function Sidebar(props: Props) {
  const { documents, conversations, activeConversation } = props;
  const router = useRouter();
  const inputRef = useRef<HTMLInputElement>(null);
  const [uploading, setUploading] = useState(0);
  const [error, setError] = useState<string | null>(null);

  async function upload(files: FileList | null) {
    if (!files?.length) return;
    setError(null);
    setUploading(files.length);
    for (const file of Array.from(files)) {
      const form = new FormData();
      form.append("file", file);
      try {
        await api("/documents", { method: "POST", body: form });
      } catch (err) {
        setError(`${file.name}: ${err instanceof ApiError ? err.message : "upload failed"}`);
      }
      setUploading((n) => n - 1);
    }
    await props.onDocumentsChanged();
    if (inputRef.current) inputRef.current.value = "";
  }

  async function remove(doc: DocumentItem) {
    if (!window.confirm(`Delete "${doc.title ?? doc.filename}"? This cannot be undone.`)) return;
    await api(`/documents/${doc.id}`, { method: "DELETE" });
    await props.onDocumentsChanged();
  }

  async function retry(doc: DocumentItem) {
    await api(`/documents/${doc.id}/reprocess`, { method: "POST" });
    await props.onDocumentsChanged();
  }

  async function removeConversation(id: string) {
    await api(`/conversations/${id}`, { method: "DELETE" });
    if (id === activeConversation) props.onNewConversation();
    await props.onConversationsChanged();
  }

  async function signOut() {
    await fetch("/api/session", { method: "DELETE" });
    router.replace("/login");
    router.refresh();
  }

  return (
    <aside className="flex w-72 shrink-0 flex-col border-r border-slate-200 bg-white">
      <div className="flex items-center gap-2 px-4 py-4">
        <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-brand-600 text-white">
          <FileSearch className="h-4 w-4" />
        </div>
        <span className="text-base font-semibold tracking-tight">DocMind</span>
      </div>

      <div className="px-3">
        <button
          onClick={() => inputRef.current?.click()}
          className="flex w-full items-center justify-center gap-2 rounded-lg border border-dashed border-slate-300 py-2.5 text-sm font-medium text-slate-600 transition hover:border-brand-500 hover:bg-brand-50 hover:text-brand-700"
        >
          {uploading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Upload className="h-4 w-4" />}
          {uploading ? `Uploading ${uploading}…` : "Upload PDF or image"}
        </button>
        <input
          ref={inputRef}
          type="file"
          multiple
          accept="application/pdf,image/png,image/jpeg,image/webp,image/tiff"
          className="hidden"
          onChange={(e) => upload(e.target.files)}
        />
        {error && <p className="mt-2 rounded bg-red-50 px-2 py-1 text-xs text-red-700">{error}</p>}
      </div>

      <section className="mt-4 min-h-0 flex-1 overflow-y-auto px-3">
        <h2 className="px-1 text-xs font-semibold tracking-wide text-slate-400 uppercase">
          Documents · {documents.length}
        </h2>
        <ul className="mt-2 space-y-1">
          {documents.length === 0 && (
            <li className="px-1 py-2 text-xs text-slate-400">Upload a document to get started.</li>
          )}
          {documents.map((doc) => (
            <li key={doc.id} className="group rounded-lg px-2 py-2 hover:bg-slate-50">
              <div className="flex items-start gap-2">
                {doc.content_type === "application/pdf" ? (
                  <FileText className="mt-0.5 h-4 w-4 shrink-0 text-slate-400" />
                ) : (
                  <FileImage className="mt-0.5 h-4 w-4 shrink-0 text-slate-400" />
                )}
                <button
                  className="min-w-0 flex-1 text-left"
                  onClick={() => doc.status === "ready" && props.onOpenDocument(doc)}
                  title={doc.summary ?? doc.filename}
                >
                  <p className="truncate text-sm font-medium text-slate-800">{doc.title ?? doc.filename}</p>
                  <p className="truncate text-xs text-slate-400">
                    {doc.status === "ready"
                      ? `${doc.pages ?? "?"} pages · ${doc.chunk_count} chunks` +
                        (doc.embedded_chunks < doc.chunk_count
                          ? ` · semantic index ${Math.floor((100 * doc.embedded_chunks) / Math.max(1, doc.chunk_count))}%`
                          : "")
                      : doc.status === "failed"
                        ? doc.error ?? "Failed"
                        : STAGE_LABEL[doc.stage ?? ""] ?? "Queued"}
                  </p>
                </button>
                <span className="flex items-center gap-1">
                  {doc.status === "ready" && <CheckCircle2 className="h-3.5 w-3.5 text-emerald-500" />}
                  {doc.status === "failed" && (
                    <button onClick={() => retry(doc)} title="Retry processing">
                      <RotateCw className="h-3.5 w-3.5 text-amber-600" />
                    </button>
                  )}
                  {doc.status === "failed" && <AlertCircle className="h-3.5 w-3.5 text-red-500" />}
                  <button
                    onClick={() => remove(doc)}
                    title="Delete"
                    className="opacity-0 transition group-hover:opacity-100"
                  >
                    <Trash2 className="h-3.5 w-3.5 text-slate-400 hover:text-red-600" />
                  </button>
                </span>
              </div>
              {(doc.status === "processing" || doc.status === "uploaded") && (
                <div className="mt-1.5 ml-6 h-1 overflow-hidden rounded bg-slate-100">
                  <div
                    className="h-full rounded bg-brand-500 transition-all duration-500"
                    style={{ width: `${Math.max(4, doc.progress * 100)}%` }}
                  />
                </div>
              )}
            </li>
          ))}
        </ul>

        <div className="mt-5 flex items-center justify-between px-1">
          <h2 className="text-xs font-semibold tracking-wide text-slate-400 uppercase">Conversations</h2>
          <button onClick={props.onNewConversation} title="New conversation" className="text-slate-400 hover:text-brand-600">
            <Plus className="h-4 w-4" />
          </button>
        </div>
        <ul className="mt-2 mb-4 space-y-0.5">
          {conversations.map((c) => (
            <li key={c.id} className="group flex items-center">
              <button
                onClick={() => props.onSelectConversation(c.id)}
                className={clsx(
                  "flex min-w-0 flex-1 items-center gap-2 rounded-lg px-2 py-1.5 text-left text-sm",
                  c.id === activeConversation ? "bg-brand-50 text-brand-700" : "text-slate-600 hover:bg-slate-50",
                )}
              >
                <MessageSquare className="h-3.5 w-3.5 shrink-0" />
                <span className="truncate">{c.title}</span>
              </button>
              <button
                onClick={() => removeConversation(c.id)}
                className="ml-1 opacity-0 transition group-hover:opacity-100"
                title="Delete conversation"
              >
                <Trash2 className="h-3.5 w-3.5 text-slate-400 hover:text-red-600" />
              </button>
            </li>
          ))}
        </ul>
      </section>

      <div className="flex items-center justify-between border-t border-slate-200 px-4 py-3">
        <span className="truncate text-xs text-slate-500">{props.email}</span>
        <button onClick={signOut} title="Sign out" className="text-slate-400 hover:text-slate-700">
          <LogOut className="h-4 w-4" />
        </button>
      </div>
    </aside>
  );
}
