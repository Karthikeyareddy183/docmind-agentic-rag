"""API behaviour that needs no database."""

import pytest
from httpx import AsyncClient

from docmind.documents.service import sanitize_filename


async def test_liveness(client: AsyncClient) -> None:
    resp = await client.get("/api/v1/health/live")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


async def test_request_id_generated_and_echoed(client: AsyncClient) -> None:
    resp = await client.get("/api/v1/health/live")
    assert len(resp.headers["X-Request-ID"]) == 16

    resp = await client.get("/api/v1/health/live", headers={"X-Request-ID": "abc-123"})
    assert resp.headers["X-Request-ID"] == "abc-123"


async def test_malicious_request_id_replaced(client: AsyncClient) -> None:
    resp = await client.get("/api/v1/health/live", headers={"X-Request-ID": "bad\nvalue"})
    assert resp.headers["X-Request-ID"] != "bad\nvalue"


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("get", "/api/v1/auth/me"),
        ("get", "/api/v1/documents"),
        ("get", "/api/v1/documents/00000000-0000-0000-0000-000000000000"),
        ("delete", "/api/v1/documents/00000000-0000-0000-0000-000000000000"),
    ],
)
async def test_protected_routes_require_token(client: AsyncClient, method: str, path: str) -> None:
    resp = await getattr(client, method)(path)
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "unauthorized"
    assert resp.headers["WWW-Authenticate"] == "Bearer"


async def test_invalid_token_rejected(client: AsyncClient) -> None:
    resp = await client.get("/api/v1/auth/me", headers={"Authorization": "Bearer not-a-jwt"})
    assert resp.status_code == 401


async def test_validation_error_shape(client: AsyncClient) -> None:
    resp = await client.post("/api/v1/auth/register", json={"email": "nope", "password": "x"})
    assert resp.status_code == 422
    body = resp.json()["error"]
    assert body["code"] == "validation_error"
    assert {tuple(d["loc"]) for d in body["details"]} >= {("body", "email"), ("body", "password")}


async def test_unknown_route_uses_error_shape(client: AsyncClient) -> None:
    resp = await client.get("/api/v1/nope")
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "http_error"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("report.pdf", "report.pdf"),
        ("../../etc/passwd", "passwd"),
        ("..\\..\\windows\\evil.pdf", "evil.pdf"),
        ("C:\\Users\\me\\scan.png", "scan.png"),
        ("bad\x00name\n.pdf", "badname.pdf"),
        ("", "upload"),
        (None, "upload"),
        ("..", "upload"),
        ("a" * 300 + ".pdf", "a" * 255),
    ],
)
def test_sanitize_filename(raw: str | None, expected: str) -> None:
    assert sanitize_filename(raw) == expected
