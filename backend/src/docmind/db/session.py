"""Async SQLAlchemy engine and session factory."""

from collections.abc import AsyncIterator

from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from docmind.config import Settings


def engine_options(settings: Settings, url: str | None = None) -> tuple[str, dict[str, object]]:
    """Resolve the connection URL and asyncpg connect args for the configured Postgres."""
    resolved = make_url(url or settings.database_url)
    connect_args: dict[str, object] = {}
    if settings.db_ssl:
        connect_args["ssl"] = "require"
    if settings.db_transaction_pooler:
        connect_args["statement_cache_size"] = 0  # asyncpg's own cache
        # SQLAlchemy's asyncpg dialect reads its cache size from the URL
        resolved = resolved.update_query_dict({"prepared_statement_cache_size": "0"})
    return resolved.render_as_string(hide_password=False), connect_args


def schema_translate_map(settings: Settings) -> dict[str | None, str] | None:
    """Map unqualified tables to ``db_schema``.

    Works through any connection pooler, unlike a session-level ``search_path``.
    """
    return None if settings.db_schema == "public" else {None: settings.db_schema}


def create_engine(settings: Settings, url: str | None = None) -> AsyncEngine:
    resolved, connect_args = engine_options(settings, url)
    engine = create_async_engine(
        resolved,
        pool_size=settings.db_pool_size,
        pool_pre_ping=True,
        echo=settings.db_echo,
        connect_args=connect_args,
    )
    translate = schema_translate_map(settings)
    if translate:
        return engine.execution_options(schema_translate_map=translate)
    return engine


def create_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False)


async def session_scope(
    factory: async_sessionmaker[AsyncSession],
) -> AsyncIterator[AsyncSession]:
    """Yield a session that is rolled back on error.

    Writers commit explicitly so commit failures surface in the response instead of in
    dependency teardown, which may run after the response has been sent.
    """
    async with factory() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
