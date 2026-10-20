"use client";

import { ChevronLeft, ChevronRight } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { Document, Page, pdfjs } from "react-pdf";
import "react-pdf/dist/Page/AnnotationLayer.css";
import "react-pdf/dist/Page/TextLayer.css";

import { Highlights } from "@/components/DocViewer";
import type { BBox } from "@/lib/types";

// Must be set in the same module that renders <Document>/<Page> (react-pdf requirement)
pdfjs.GlobalWorkerOptions.workerSrc = new URL("pdfjs-dist/build/pdf.worker.min.mjs", import.meta.url).toString();

interface Props {
  url: string;
  page: number;
  highlights: BBox[];
}

export default function PdfView({ url, page: initialPage, highlights }: Props) {
  const [numPages, setNumPages] = useState(0);
  const [page, setPage] = useState(initialPage);
  const [width, setWidth] = useState(600);
  const containerRef = useRef<HTMLDivElement>(null);
  const pageRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const observer = new ResizeObserver(([entry]) => setWidth(Math.min(entry.contentRect.width, 900)));
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  const boxes = page === initialPage ? highlights.filter((b) => b.page === page) : [];

  return (
    <div ref={containerRef} className="mx-auto w-full">
      <div className="mb-3 flex items-center justify-center gap-3 text-sm text-slate-600">
        <button
          onClick={() => setPage((p) => Math.max(1, p - 1))}
          disabled={page <= 1}
          className="rounded p-1 hover:bg-white disabled:opacity-30"
          title="Previous page"
        >
          <ChevronLeft className="h-4 w-4" />
        </button>
        <span>
          Page {page} {numPages ? `of ${numPages}` : ""}
        </span>
        <button
          onClick={() => setPage((p) => Math.min(numPages || p, p + 1))}
          disabled={!!numPages && page >= numPages}
          className="rounded p-1 hover:bg-white disabled:opacity-30"
          title="Next page"
        >
          <ChevronRight className="h-4 w-4" />
        </button>
      </div>
      <Document
        file={url}
        onLoadSuccess={({ numPages: n }) => setNumPages(n)}
        loading={<p className="p-6 text-sm text-slate-400">Loading PDF…</p>}
        error={<p className="p-6 text-sm text-red-600">Could not load this PDF.</p>}
      >
        <div ref={pageRef} className="relative mx-auto w-fit shadow-md">
          <Page pageNumber={page} width={width} renderAnnotationLayer={false} />
          {boxes.length > 0 && (
            <div className="pointer-events-none absolute inset-0">
              <Highlights boxes={boxes} />
            </div>
          )}
        </div>
      </Document>
    </div>
  );
}
