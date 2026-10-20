"""Test fixtures.

Unit tests need nothing external. Integration tests (marked ``db``) need Postgres with pgvector,
configured by ``DOCMIND_TEST_DATABASE_URL`` (environment or ``backend/.env``). They run entirely
inside the ``docmind_test`` schema, which is dropped and rebuilt by the migrations each session,
so the application's own schema is never touched.
"""

import os
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from dotenv import dotenv_values
from httpx import ASGITransport, AsyncClient
from pydantic import SecretStr
from sqlalchemy import text

from docmind.api.app import create_app
from docmind.config import Settings, get_settings
from docmind.db.session import create_engine

BACKEND_DIR = Path(__file__).resolve().parents[1]
TEST_DATABASE_URL = os.environ.get("DOCMIND_TEST_DATABASE_URL") or dotenv_values(
    BACKEND_DIR / ".env"
).get("DOCMIND_TEST_DATABASE_URL")
TEST_SCHEMA = "docmind_test"


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if TEST_DATABASE_URL:
        return
    skip = pytest.mark.skip(reason="DOCMIND_TEST_DATABASE_URL not set")
    for item in items:
        if "db" in item.keywords:
            item.add_marker(skip)


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "db: needs a Postgres test database")


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return get_settings().model_copy(
        update=dict(
            environment="test",
            database_url=TEST_DATABASE_URL
            or "postgresql+asyncpg://unused:unused@localhost:1/unused",
            db_schema=TEST_SCHEMA,
            db_pool_size=2,
            jwt_secret=SecretStr("test-secret-that-is-at-least-32-bytes-long"),
            storage_dir=tmp_path / "uploads",
            max_upload_mb=1,
            log_level="WARNING",
        )
    )


def _test_settings() -> Settings:
    assert TEST_DATABASE_URL, "db tests require DOCMIND_TEST_DATABASE_URL"
    settings = get_settings().model_copy(
        update={"database_url": TEST_DATABASE_URL, "db_schema": TEST_SCHEMA, "db_pool_size": 2}
    )
    if get_settings().db_schema == TEST_SCHEMA:
        pytest.exit("The application must not use the test schema", returncode=2)
    return settings


@pytest.fixture(scope="session")
def migrated_database() -> Settings:
    """Rebuild the test schema once per session by running the migrations from scratch."""
    settings = _test_settings()
    cfg = Config(str(BACKEND_DIR / "alembic.ini"))
    cfg.attributes["settings"] = settings
    command.downgrade(cfg, "base")
    command.upgrade(cfg, "head")
    return settings


@pytest.fixture
async def clean_db(migrated_database: Settings) -> AsyncIterator[None]:
    yield
    engine = create_engine(migrated_database)
    async with engine.begin() as conn:
        # users cascades to documents/chunks/conversations/messages; jobs has no FK to users
        tables = ", ".join(f'"{TEST_SCHEMA}"."{t}"' for t in ("users", "jobs"))
        await conn.execute(text(f"TRUNCATE {tables} RESTART IDENTITY CASCADE"))
    await engine.dispose()


@pytest.fixture
async def client(settings: Settings) -> AsyncIterator[AsyncClient]:
    app = create_app(settings)
    async with (
        app.router.lifespan_context(app),
        AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c,
    ):
        yield c


@pytest.fixture
async def db_client(clean_db: None, client: AsyncClient) -> AsyncClient:
    return client


async def register_and_login(
    client: AsyncClient, email: str, password: str = "s3cret-pass"
) -> dict[str, str]:
    resp = await client.post("/api/v1/auth/register", json={"email": email, "password": password})
    assert resp.status_code == 201, resp.text
    resp = await client.post("/api/v1/auth/login", data={"username": email, "password": password})
    assert resp.status_code == 200, resp.text
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


# Smallest byte strings that file-type sniffing recognises
PDF_BYTES = b"%PDF-1.4\n1 0 obj<<>>endobj\ntrailer<<>>\n%%EOF\n"
PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
