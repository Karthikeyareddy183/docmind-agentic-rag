"""End-to-end API tests against a real Postgres."""

import uuid
from pathlib import Path

import pytest
from httpx import AsyncClient

from docmind.config import Settings
from tests.conftest import PDF_BYTES, PNG_BYTES, register_and_login

pytestmark = pytest.mark.db


async def test_readiness_checks_database(db_client: AsyncClient) -> None:
    resp = await db_client.get("/api/v1/health/ready")
    assert resp.status_code == 200
    assert resp.json()["database"] == "up"


class TestAuth:
    async def test_register_login_me(self, db_client: AsyncClient) -> None:
        headers = await register_and_login(db_client, "Alice@Example.com")
        resp = await db_client.get("/api/v1/auth/me", headers=headers)
        assert resp.status_code == 200
        assert resp.json()["email"] == "alice@example.com"
        assert "password_hash" not in resp.json()

    async def test_duplicate_email_conflicts(self, db_client: AsyncClient) -> None:
        await register_and_login(db_client, "bob@example.com")
        resp = await db_client.post(
            "/api/v1/auth/register", json={"email": "BOB@example.com", "password": "another-pass"}
        )
        assert resp.status_code == 409
        assert resp.json()["error"]["code"] == "conflict"

    @pytest.mark.parametrize("email", ["carol@example.com", "nobody@example.com"])
    async def test_bad_credentials_same_error(self, db_client: AsyncClient, email: str) -> None:
        await register_and_login(db_client, "carol@example.com")
        resp = await db_client.post(
            "/api/v1/auth/login", data={"username": email, "password": "wrong-password"}
        )
        assert resp.status_code == 401
        assert resp.json()["error"]["message"] == "Incorrect email or password"

    async def test_token_for_deleted_user_rejected(
        self, db_client: AsyncClient, settings: Settings
    ) -> None:
        from docmind.auth.security import create_access_token

        token = create_access_token(uuid.uuid4(), settings)
        resp = await db_client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert resp.status_code == 401


class TestDocuments:
    async def test_upload_list_get_delete(self, db_client: AsyncClient, settings: Settings) -> None:
        headers = await register_and_login(db_client, "dana@example.com")

        resp = await db_client.post(
            "/api/v1/documents",
            headers=headers,
            files={"file": ("../../Quarterly Report.pdf", PDF_BYTES, "text/plain")},
        )
        assert resp.status_code == 201, resp.text
        doc = resp.json()
        assert doc["filename"] == "Quarterly Report.pdf"
        assert doc["content_type"] == "application/pdf"  # sniffed, not the client's text/plain
        assert doc["status"] == "uploaded"
        assert doc["size_bytes"] == len(PDF_BYTES)
        stored = list(Path(settings.storage_dir).iterdir())
        assert len(stored) == 1

        resp = await db_client.get("/api/v1/documents", headers=headers)
        assert resp.json()["total"] == 1
        assert resp.json()["items"][0]["id"] == doc["id"]

        resp = await db_client.get(f"/api/v1/documents/{doc['id']}", headers=headers)
        assert resp.status_code == 200

        resp = await db_client.delete(f"/api/v1/documents/{doc['id']}", headers=headers)
        assert resp.status_code == 204
        resp = await db_client.get(f"/api/v1/documents/{doc['id']}", headers=headers)
        assert resp.status_code == 404
        assert list(Path(settings.storage_dir).iterdir()) == []

    async def test_duplicate_upload_returns_existing(
        self, db_client: AsyncClient, settings: Settings
    ) -> None:
        headers = await register_and_login(db_client, "erin@example.com")
        first = await db_client.post(
            "/api/v1/documents", headers=headers, files={"file": ("a.png", PNG_BYTES, "image/png")}
        )
        second = await db_client.post(
            "/api/v1/documents", headers=headers, files={"file": ("b.png", PNG_BYTES, "image/png")}
        )
        assert first.status_code == 201
        assert second.status_code == 200
        assert second.json()["id"] == first.json()["id"]
        assert len(list(Path(settings.storage_dir).iterdir())) == 1

    async def test_rejects_unsupported_type(
        self, db_client: AsyncClient, settings: Settings
    ) -> None:
        headers = await register_and_login(db_client, "finn@example.com")
        resp = await db_client.post(
            "/api/v1/documents",
            headers=headers,
            files={"file": ("evil.pdf", b"MZ\x90\x00 not a pdf at all", "application/pdf")},
        )
        assert resp.status_code == 415
        assert list(Path(settings.storage_dir).iterdir()) == []

    async def test_rejects_oversized_file(self, db_client: AsyncClient, settings: Settings) -> None:
        headers = await register_and_login(db_client, "gina@example.com")
        big = PDF_BYTES + b"0" * (settings.max_upload_bytes + 1)
        resp = await db_client.post(
            "/api/v1/documents",
            headers=headers,
            files={"file": ("big.pdf", big, "application/pdf")},
        )
        assert resp.status_code == 413
        assert list(Path(settings.storage_dir).iterdir()) == []

    async def test_users_cannot_see_each_others_documents(self, db_client: AsyncClient) -> None:
        owner = await register_and_login(db_client, "owner@example.com")
        intruder = await register_and_login(db_client, "intruder@example.com")
        doc = (
            await db_client.post(
                "/api/v1/documents",
                headers=owner,
                files={"file": ("secret.pdf", PDF_BYTES, "application/pdf")},
            )
        ).json()

        assert (await db_client.get("/api/v1/documents", headers=intruder)).json()["total"] == 0
        for method in ("get", "delete"):
            resp = await getattr(db_client, method)(
                f"/api/v1/documents/{doc['id']}", headers=intruder
            )
            assert resp.status_code == 404
        # Still intact for the owner
        assert (
            await db_client.get(f"/api/v1/documents/{doc['id']}", headers=owner)
        ).status_code == 200

    async def test_same_file_allowed_for_different_users(self, db_client: AsyncClient) -> None:
        a = await register_and_login(db_client, "hana@example.com")
        b = await register_and_login(db_client, "ivan@example.com")
        files = {"file": ("x.pdf", PDF_BYTES, "application/pdf")}
        assert (
            await db_client.post("/api/v1/documents", headers=a, files=files)
        ).status_code == 201
        assert (
            await db_client.post("/api/v1/documents", headers=b, files=files)
        ).status_code == 201

    async def test_pagination(self, db_client: AsyncClient) -> None:
        headers = await register_and_login(db_client, "jo@example.com")
        for i in range(3):
            await db_client.post(
                "/api/v1/documents",
                headers=headers,
                files={"file": (f"{i}.pdf", PDF_BYTES + bytes([i]), "application/pdf")},
            )
        resp = await db_client.get("/api/v1/documents?limit=2&offset=0", headers=headers)
        assert resp.json()["total"] == 3
        assert len(resp.json()["items"]) == 2
        resp = await db_client.get("/api/v1/documents?limit=2&offset=2", headers=headers)
        assert len(resp.json()["items"]) == 1
