"""users and documents; enable pgvector

Revision ID: 0001
Revises:
Create Date: 2026-10-18
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

DOCUMENT_STATUSES = ("uploaded", "processing", "ready", "failed")


def _qualified(table: str) -> str:
    """Schema-qualify a table for raw SQL, which schema_translate_map does not rewrite."""
    translate = op.get_bind().get_execution_options().get("schema_translate_map") or {}
    schema = translate.get(None)
    return f'"{schema}"."{table}"' if schema else f'"{table}"'


def enable_rls(table: str) -> None:
    """Defense in depth on Supabase: with RLS on and no policies, the anon/authenticated
    roles can read nothing even if the schema were ever exposed. The app role owns the
    tables, and owners are not subject to non-forced RLS."""
    op.execute(f"ALTER TABLE {_qualified(table)} ENABLE ROW LEVEL SECURITY")


def _timestamps() -> list[sa.Column[object]]:
    return [
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    ]


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "users",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("email", sa.String(320), nullable=False),
        sa.Column("password_hash", sa.String(255), nullable=False),
        *_timestamps(),
    )
    op.create_index("ix_users_email", "users", ["email"], unique=True)

    op.create_table(
        "documents",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "owner_id", sa.Uuid(), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False
        ),
        sa.Column("filename", sa.String(255), nullable=False),
        sa.Column("content_type", sa.String(100), nullable=False),
        sa.Column("size_bytes", sa.BigInteger(), nullable=False),
        sa.Column("sha256", sa.String(64), nullable=False),
        sa.Column("storage_key", sa.String(512), nullable=False),
        sa.Column(
            "status",
            sa.Enum(
                *DOCUMENT_STATUSES,
                name="document_status",
                native_enum=False,
                create_constraint=True,
                length=20,
            ),
            nullable=False,
            server_default="uploaded",
        ),
        sa.Column("pages", sa.Integer(), nullable=True),
        sa.Column("progress", sa.Float(), nullable=False, server_default="0"),
        sa.Column("error", sa.Text(), nullable=True),
        *_timestamps(),
    )
    op.create_index("ix_documents_owner_id", "documents", ["owner_id"])
    op.create_index("uq_documents_owner_sha256", "documents", ["owner_id", "sha256"], unique=True)

    enable_rls("users")
    enable_rls("documents")


def downgrade() -> None:
    op.drop_table("documents")
    op.drop_table("users")
