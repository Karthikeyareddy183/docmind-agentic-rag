export type DocumentStatus = "uploaded" | "processing" | "ready" | "failed";

export interface DocumentItem {
  id: string;
  filename: string;
  content_type: string;
  size_bytes: number;
  status: DocumentStatus;
  stage: string | null;
  pages: number | null;
  progress: number;
  error: string | null;
  title: string | null;
  summary: string | null;
  chunk_count: number;
  embedded_chunks: number;
  created_at: string;
  updated_at: string;
}

export interface BBox {
  page: number;
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

export interface Citation {
  id: string;
  chunk_id: string;
  document_id: string;
  filename: string;
  document_title: string;
  chunk_type: "text" | "table" | "figure";
  page: number;
  page_end: number;
  bboxes: BBox[];
  snippet: string;
  score: number;
}

export interface Step {
  type: "step";
  name: string;
  [key: string]: unknown;
}

export interface Computation {
  table_source: string;
  explanation: string;
  expression: string;
  formatted: string;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations: Citation[];
  confidence: number | null;
  verified?: boolean | null;
  computations?: Computation[];
  steps: Step[];
  usage?: { total_tokens?: number; cost_usd?: number; calls?: number };
  latency_ms?: number | null;
  streaming?: boolean;
  revised?: boolean;
  error?: string;
}

export interface Conversation {
  id: string;
  title: string;
  document_ids: string[];
  created_at: string;
  updated_at: string;
}

export interface StoredMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations: Citation[];
  confidence: number | null;
  trace: Step[];
  usage: { total_tokens?: number; cost_usd?: number };
  latency_ms: number | null;
  created_at: string;
}
