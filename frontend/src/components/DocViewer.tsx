"use client";

import { X } from "lucide-react";
import dynamic from "next/dynamic";

import { fileUrl } from "@/lib/api";
import type { BBox } from "@/lib/types";

export interface ViewerTarget {
  documentId: string;
  page: number;
  bboxes: BBox[];
  title: string;
  contentType: string;
  label?: string;
}

// pdf.js needs browser APIs: load it only on the client
const PdfView = dynamic(() => import("@/components/PdfView"), {
  ssr: false,
  loading: () => <p className="p-6 text-sm text-slate-400">Loading viewer…</p>,
});

function Highlights({ boxes }: { boxes: BBox[] }) {
  return (
    <>
      {boxes.map((b, i) => (
        <div
          key={i}
          className="cite-highlight"
          style={{
            left: `${b.x0 * 100}%`,
            top: `${b.y0 * 100}%`,
            width: `${(b.x1 - b.x0) * 100}%`,
            height: `${(b.y1 - b.y0) * 100}%`,
          }}
        />
      ))}
    </>
  );
}

export default function DocViewer({ target, onClose }: { target: ViewerTarget; onClose: () => void }) {
  const isPdf = target.contentType === "application/pdf" || target.contentType === "";
  const url = fileUrl(target.documentId);

  return (
    <aside className="flex w-[46%] max-w-[760px] min-w-[420px] shrink-0 flex-col border-l border-slate-200 bg-slate-100">
      <header className="flex items-center gap-3 border-b border-slate-200 bg-white px-4 py-3">
        <div className="min-w-0 flex-1">
          <p className="truncate text-sm font-semibold">{target.title}</p>
          <p className="text-xs text-slate-500">
            {target.label
              ? `Source ${target.label} · cited on page ${target.page}` +
                (target.bboxes.length ? " · region highlighted" : "")
              : "Full document"}
          </p>
        </div>
        <a href={url} target="_blank" rel="noreferrer" className="text-xs font-medium text-brand-600 hover:underline">
          Open original
        </a>
        <button onClick={onClose} className="rounded p-1 text-slate-400 hover:bg-slate-100 hover:text-slate-700" title="Close">
          <X className="h-4 w-4" />
        </button>
      </header>
      <div className="min-h-0 flex-1 overflow-auto p-4">
        {isPdf ? (
          <PdfView key={`${target.documentId}:${target.page}`} url={url} page={target.page} highlights={target.bboxes} />
        ) : (
          <div className="relative mx-auto w-fit shadow-md">
            {/* eslint-disable-next-line @next/next/no-img-element -- authenticated same-origin file, not optimisable */}
            <img src={url} alt={target.title} className="block max-w-full" />
            <Highlights boxes={target.bboxes} />
          </div>
        )}
      </div>
    </aside>
  );
}

export { Highlights };
