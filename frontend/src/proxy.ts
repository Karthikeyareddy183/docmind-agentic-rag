import { NextResponse, type NextRequest } from "next/server";

const SESSION_COOKIE = "dm_session";

/**
 * Optimistic auth check: send signed-out visitors to /login (and signed-in ones away from it).
 * Real authorization happens in the API on every request.
 */
export function proxy(request: NextRequest) {
  const signedIn = Boolean(request.cookies.get(SESSION_COOKIE)?.value);
  const onLogin = request.nextUrl.pathname.startsWith("/login");
  if (!signedIn && !onLogin) return NextResponse.redirect(new URL("/login", request.url));
  if (signedIn && onLogin) return NextResponse.redirect(new URL("/", request.url));
  return NextResponse.next();
}

export const config = {
  // Pages only: API routes answer 401 themselves; static assets are always public
  matcher: ["/((?!api|_next/static|_next/image|favicon.ico|.*\\.(?:svg|png|ico|mjs)$).*)"],
};
