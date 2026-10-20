"""Alembic environment (async engine; URL from application settings)."""

import asyncio
from logging.config import fileConfig

from alembic import context
from sqlalchemy import text
from sqlalchemy.engine import Connection

from docmind.config import get_settings
from docmind.db.models import Base
from docmind.db.session import create_engine

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _settings():  # type: ignore[no-untyped-def]
    return config.attributes.get("settings") or get_settings()


def run_migrations_offline() -> None:
    context.configure(
        url=_settings().database_url, target_metadata=target_metadata, literal_binds=True
    )
    with context.begin_transaction():
        context.run_migrations()


def _run(connection: Connection) -> None:
    schema = _settings().db_schema
    if schema != "public":
        # Check first: CREATE SCHEMA IF NOT EXISTS still needs database-level CREATE,
        # which a least-privilege app role (owning pre-provisioned schemas) does not have
        exists = connection.execute(
            text("SELECT 1 FROM pg_namespace WHERE nspname = :schema"), {"schema": schema}
        ).scalar()
        if not exists:
            connection.exec_driver_sql(f'CREATE SCHEMA "{schema}"')
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        version_table_schema=None if schema == "public" else schema,
    )
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    engine = create_engine(_settings())
    async with engine.connect() as connection:
        await connection.run_sync(_run)
        await connection.commit()
    await engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
