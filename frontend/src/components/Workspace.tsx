"use client";

import { useRouter } from "next/navigation";
import { useCallback, useEffect, useState } from "react";

import ChatPanel from "@/components/ChatPanel";
import DocViewer, { type ViewerTarget } from "@/components/DocViewer";
import Sidebar from "@/components/Sidebar";
import { UNAUTHORIZED_EVENT, api, eventsUrl } from "@/lib/api";
import type { Conversation, DocumentItem } from "@/lib/types";

export default function Workspace() {
  const router = useRouter();
  const [documents, setDocuments] = useState<DocumentItem[]>([]);
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [activeConversation, setActiveConversation] = useState<string | null>(null);
  // Remount the chat panel only when the user switches conversations, not when the panel
  // itself creates one mid-stream (that would drop the answer being streamed)
  const [panelKey, setPanelKey] = useState(0);
  const switchConversation = useCallback((id: string | null) => {
    setActiveConversation(id);
    setPanelKey((k) => k + 1);
  }, []);
  const [viewer, setViewer] = useState<ViewerTarget | null>(null);
  const [email, setEmail] = useState<string>("");

  const refreshDocuments = useCallback(async () => {
    const data = await api<{ items: DocumentItem[] }>("/documents?limit=100");
    setDocuments(data.items);
  }, []);

  const refreshConversations = useCallback(async () => {
    setConversations(await api<Conversation[]>("/conversations"));
  }, []);

  useEffect(() => {
    // Initial load: state is set from the async results, and ignored after unmount
    let active = true;
    api<{ items: DocumentItem[] }>("/documents?limit=100").then((d) => active && setDocuments(d.items));
    api<Conversation[]>("/conversations").then((c) => active && setConversations(c));
    fetch("/api/session")
      .then((r) => (r.ok ? r.json() : null))
      .then((me) => active && me && setEmail(me.email));
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    const toLogin = () => router.replace("/login");
    window.addEventListener(UNAUTHORIZED_EVENT, toLogin);
    return () => window.removeEventListener(UNAUTHORIZED_EVENT, toLogin);
  }, [router]);

  // Live ingestion progress over SSE for documents still being processed
  const pendingIds = documents
    .filter(
      (d) =>
        d.status === "uploaded" ||
        d.status === "processing" ||
        (d.status === "ready" && d.embedded_chunks < d.chunk_count),
    )
    .map((d) => d.id)
    .join(",");
  useEffect(() => {
    if (!pendingIds) return;
    const sources = pendingIds.split(",").map((id) => {
      const source = new EventSource(eventsUrl(id));
      source.addEventListener("progress", (e) => {
        const updated = JSON.parse((e as MessageEvent).data) as DocumentItem;
        setDocuments((docs) => docs.map((d) => (d.id === updated.id ? updated : d)));
        const done =
          updated.status === "failed" ||
          (updated.status === "ready" && updated.embedded_chunks >= updated.chunk_count);
        if (done) source.close();
      });
      source.onerror = () => source.close();
      return source;
    });
    return () => sources.forEach((s) => s.close());
  }, [pendingIds]);

  const openCitation = useCallback(
    (target: ViewerTarget) => {
      const doc = documents.find((d) => d.id === target.documentId);
      setViewer({ ...target, contentType: doc?.content_type ?? target.contentType });
    },
    [documents],
  );

  return (
    <div className="flex h-full">
      <Sidebar
        email={email}
        documents={documents}
        conversations={conversations}
        activeConversation={activeConversation}
        onSelectConversation={switchConversation}
        onNewConversation={() => switchConversation(null)}
        onDocumentsChanged={refreshDocuments}
        onConversationsChanged={refreshConversations}
        onOpenDocument={(doc) =>
          setViewer({ documentId: doc.id, page: 1, bboxes: [], title: doc.title ?? doc.filename, contentType: doc.content_type })
        }
      />
      <main className="flex min-w-0 flex-1 flex-col">
        <ChatPanel
          key={panelKey}
          conversationId={activeConversation}
          readyCount={documents.filter((d) => d.status === "ready").length}
          onConversationCreated={(id) => {
            setActiveConversation(id);
            void refreshConversations();
          }}
          onOpenCitation={openCitation}
        />
      </main>
      {viewer && <DocViewer target={viewer} onClose={() => setViewer(null)} />}
    </div>
  );
}
