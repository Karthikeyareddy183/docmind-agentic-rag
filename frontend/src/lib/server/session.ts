import "server-only";

import { cookies } from "next/headers";

export const BACKEND_URL = (process.env.BACKEND_URL ?? "http://localhost:8000").replace(/\/$/, "");
export const API_PREFIX = "/api/v1";
export const SESSION_COOKIE = "dm_session";

/** The backend JWT lives only in an httpOnly cookie; browser JavaScript never sees it. */
export async function getToken(): Promise<string | undefined> {
  return (await cookies()).get(SESSION_COOKIE)?.value;
}

export async function setToken(token: string, maxAgeSeconds: number): Promise<void> {
  (await cookies()).set({
    name: SESSION_COOKIE,
    value: token,
    httpOnly: true,
    sameSite: "lax",
    secure: process.env.NODE_ENV === "production",
    path: "/",
    maxAge: maxAgeSeconds,
  });
}

export async function clearToken(): Promise<void> {
  (await cookies()).delete(SESSION_COOKIE);
}
