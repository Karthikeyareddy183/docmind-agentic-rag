/** Browser-side API client. All calls go through the same-origin BFF proxy (cookie auth). */

const BASE = "/api/backend";

/** Dispatched on 401 so the app shell can route to /login (no router access in this module). */
export const UNAUTHORIZED_EVENT = "docmind:unauthorized";

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}

async function parseError(resp: Response): Promise<ApiError> {
  let message = resp.statusText || "Request failed";
  try {
    const body = await resp.json();
    message = body?.error?.message ?? body?.error ?? message;
  } catch {
    /* non-JSON error body */
  }
  if (resp.status === 401 && typeof window !== "undefined") window.dispatchEvent(new Event(UNAUTHORIZED_EVENT));
  return new ApiError(resp.status, message);
}

export async function api<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers);
  if (init.body && !(init.body instanceof FormData)) headers.set("Content-Type", "application/json");
  const resp = await fetch(`${BASE}${path}`, { ...init, headers, cache: "no-store" });
  if (!resp.ok) throw await parseError(resp);
  return (resp.status === 204 ? undefined : await resp.json()) as T;
}

export const fileUrl = (documentId: string) => `${BASE}/documents/${documentId}/file`;
export const eventsUrl = (documentId: string) => `${BASE}/documents/${documentId}/events`;

export interface SseEvent {
  event: string;
  data: unknown;
}

/** POST a JSON body and parse the Server-Sent Events response stream. */
export async function* postSse(path: string, body: unknown, signal?: AbortSignal): AsyncGenerator<SseEvent> {
  const resp = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify(body),
    signal,
  });
  if (!resp.ok || !resp.body) throw await parseError(resp);

  const reader = resp.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += value.replace(/\r\n/g, "\n");
    let boundary: number;
    while ((boundary = buffer.indexOf("\n\n")) !== -1) {
      const raw = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      let event = "message";
      const data: string[] = [];
      for (const line of raw.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) data.push(line.slice(5).trimStart());
      }
      if (!data.length) continue; // comments / keep-alive pings
      try {
        yield { event, data: JSON.parse(data.join("\n")) };
      } catch {
        yield { event, data: data.join("\n") };
      }
    }
  }
}
