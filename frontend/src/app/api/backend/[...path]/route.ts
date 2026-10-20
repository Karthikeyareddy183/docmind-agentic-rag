import type { NextRequest } from "next/server";

import { API_PREFIX, BACKEND_URL, clearToken, getToken } from "@/lib/server/session";

/**
 * Backend-for-frontend proxy: /api/backend/<path> → FastAPI /api/v1/<path>.
 * Adds the bearer token from the httpOnly cookie, streams request bodies (uploads) and
 * response bodies (SSE answers, PDFs) straight through without buffering.
 */
const FORWARDED_REQUEST_HEADERS = ["content-type", "accept", "x-request-id"];
const FORWARDED_RESPONSE_HEADERS = [
  "content-type",
  "content-disposition",
  "content-length",
  "cache-control",
  "x-request-id",
];

async function forward(request: NextRequest, ctx: RouteContext<"/api/backend/[...path]">) {
  const token = await getToken();
  if (!token) {
    return Response.json({ error: { code: "unauthorized", message: "Not signed in" } }, { status: 401 });
  }
  const { path } = await ctx.params;
  const target = new URL(`${BACKEND_URL}${API_PREFIX}/${path.map(encodeURIComponent).join("/")}`);
  target.search = request.nextUrl.search;

  const headers = new Headers({ Authorization: `Bearer ${token}` });
  for (const name of FORWARDED_REQUEST_HEADERS) {
    const value = request.headers.get(name);
    if (value) headers.set(name, value);
  }

  const hasBody = !["GET", "HEAD"].includes(request.method);
  let upstream: Response;
  try {
    upstream = await fetch(target, {
      method: request.method,
      headers,
      body: hasBody ? request.body : undefined,
      // Required by Node's fetch to stream a request body (e.g. large uploads)
      ...(hasBody ? { duplex: "half" } : {}),
      cache: "no-store",
      signal: request.signal,
    } as RequestInit);
  } catch {
    return Response.json(
      { error: { code: "backend_unavailable", message: "The DocMind API is not reachable" } },
      { status: 502 },
    );
  }

  if (upstream.status === 401) await clearToken();

  const responseHeaders = new Headers();
  for (const name of FORWARDED_RESPONSE_HEADERS) {
    const value = upstream.headers.get(name);
    if (value) responseHeaders.set(name, value);
  }
  if (upstream.headers.get("content-type")?.startsWith("text/event-stream")) {
    responseHeaders.set("Cache-Control", "no-cache, no-transform");
    responseHeaders.set("X-Accel-Buffering", "no");
  }
  return new Response(upstream.body, { status: upstream.status, headers: responseHeaders });
}

export const GET = forward;
export const POST = forward;
export const PUT = forward;
export const PATCH = forward;
export const DELETE = forward;
