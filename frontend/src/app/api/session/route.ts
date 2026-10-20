import { API_PREFIX, BACKEND_URL, clearToken, getToken, setToken } from "@/lib/server/session";

type Credentials = { email?: string; password?: string; mode?: "login" | "register" };

async function backendError(resp: Response): Promise<string> {
  try {
    const body = await resp.json();
    return body?.error?.details?.[0]?.msg ?? body?.error?.message ?? resp.statusText;
  } catch {
    return resp.statusText;
  }
}

/** Sign in (or register, then sign in). Stores the backend JWT in an httpOnly cookie. */
export async function POST(request: Request) {
  const { email, password, mode = "login" } = (await request.json()) as Credentials;
  if (!email || !password) {
    return Response.json({ error: "Email and password are required" }, { status: 400 });
  }

  if (mode === "register") {
    const resp = await fetch(`${BACKEND_URL}${API_PREFIX}/auth/register`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password }),
    });
    if (!resp.ok) return Response.json({ error: await backendError(resp) }, { status: resp.status });
  }

  const resp = await fetch(`${BACKEND_URL}${API_PREFIX}/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: new URLSearchParams({ username: email, password }),
  });
  if (!resp.ok) return Response.json({ error: await backendError(resp) }, { status: resp.status });

  const { access_token, expires_in } = (await resp.json()) as { access_token: string; expires_in: number };
  await setToken(access_token, expires_in);
  return Response.json({ ok: true });
}

/** Current user, or 401. */
export async function GET() {
  const token = await getToken();
  if (!token) return Response.json({ error: "Not signed in" }, { status: 401 });
  const resp = await fetch(`${BACKEND_URL}${API_PREFIX}/auth/me`, {
    headers: { Authorization: `Bearer ${token}` },
    cache: "no-store",
  });
  if (!resp.ok) {
    await clearToken();
    return Response.json({ error: "Session expired" }, { status: 401 });
  }
  return Response.json(await resp.json());
}

export async function DELETE() {
  await clearToken();
  return Response.json({ ok: true });
}
