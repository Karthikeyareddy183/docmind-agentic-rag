"""chunks (pgvector + full-text), job queue, conversations, messages, feedback

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-18
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

EMBEDDING_DIMENSION = 1024


def _schema() -> str | None:
    """Target schema; add/drop_column ops are not rewritten by schema_translate_map."""
    translate = op.get_bind().get_execution_options().get("schema_translate_map") or {}
    return translate.get(None)


def _qualified(table: str) -> str:
    schema = _schema()
    return f'"{schema}"."{table}"' if schema else f'"{table}"'


def _enum(name: str, *values: str) -> sa.Enum:
    return sa.Enum(*values, name=name, native_enum=False, create_constraint=True, length=20)


def _timestamps() -> list[sa.Column[object]]:
    return [
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column(
            "updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    ]


def _fk(column: str, target: str) -> sa.Column[object]:
    return sa.Column(column, sa.Uuid(), sa.ForeignKey(target, ondelete="CASCADE"), nullable=False)


def upgrade() -> None:
    schema = _schema()
    op.add_column("documents", sa.Column("stage", sa.String(40), nullable=True), schema=schema)
    op.add_column("documents", sa.Column("title", sa.String(500), nullable=True), schema=schema)
    op.add_column("documents", sa.Column("summary", sa.Text(), nullable=True), schema=schema)
    op.add_column(
        "documents",
        sa.Column("chunk_count", sa.Integer(), nullable=False, server_default="0"),
        schema=schema,
    )

    op.create_table(
        "chunks",
        sa.Column("id", sa.Uuid(), primary_key=True),
        _fk("document_id", "documents.id"),
        _fk("owner_id", "users.id"),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("chunk_type", _enum("chunk_type", "text", "table", "figure"), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("context_header", sa.Text(), nullable=False, server_default=""),
        sa.Column("section_path", sa.Text(), nullable=False, server_default=""),
        sa.Column("page_start", sa.SmallInteger(), nullable=False),
        sa.Column("page_end", sa.SmallInteger(), nullable=False),
        sa.Column("bboxes", JSONB(), nullable=False, server_default="[]"),
        sa.Column("token_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("data", JSONB(), nullable=False, server_default="{}"),
        sa.Column("embedding", Vector(EMBEDDING_DIMENSION), nullable=True),
        sa.Column("embedding_model", sa.String(200), nullable=False, server_default=""),
        sa.Column(
            "tsv",
            TSVECTOR(),
            sa.Computed(
                "setweight(to_tsvector('english', coalesce(context_header, '')), 'B') || "
                "setweight(to_tsvector('english', coalesce(content, '')), 'A')",
                persisted=True,
            ),
        ),
    )
    op.create_index("ix_chunks_document_id", "chunks", ["document_id"])
    op.create_index("ix_chunks_owner_document", "chunks", ["owner_id", "document_id"])
    op.create_index(
        "ix_chunks_embedding_hnsw",
        "chunks",
        ["embedding"],
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )
    op.create_index("ix_chunks_tsv", "chunks", ["tsv"], postgresql_using="gin")

    op.create_table(
        "jobs",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("kind", sa.String(50), nullable=False),
        sa.Column("payload", JSONB(), nullable=False, server_default="{}"),
        sa.Column(
            "status",
            _enum("job_status", "queued", "running", "done", "failed"),
            nullable=False,
            server_default="queued",
        ),
        sa.Column("attempts", sa.SmallInteger(), nullable=False, server_default="0"),
        sa.Column("max_attempts", sa.SmallInteger(), nullable=False, server_default="3"),
        sa.Column(
            "run_after", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column("locked_until", sa.DateTime(timezone=True), nullable=True),
        sa.Column("locked_by", sa.String(100), nullable=True),
        sa.Column("last_error", sa.Text(), nullable=True),
        *_timestamps(),
    )
    op.create_index(
        "ix_jobs_claimable",
        "jobs",
        ["status", "run_after"],
        postgresql_where=sa.text("status = 'queued'"),
    )

    op.create_table(
        "conversations",
        sa.Column("id", sa.Uuid(), primary_key=True),
        _fk("owner_id", "users.id"),
        sa.Column("title", sa.String(200), nullable=False, server_default="New conversation"),
        sa.Column("document_ids", JSONB(), nullable=False, server_default="[]"),
        *_timestamps(),
    )
    op.create_index("ix_conversations_owner_id", "conversations", ["owner_id"])

    op.create_table(
        "messages",
        sa.Column("id", sa.Uuid(), primary_key=True),
        _fk("conversation_id", "conversations.id"),
        _fk("owner_id", "users.id"),
        sa.Column("role", _enum("message_role", "user", "assistant"), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("citations", JSONB(), nullable=False, server_default="[]"),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("trace", JSONB(), nullable=False, server_default="[]"),
        sa.Column("model", sa.String(200), nullable=True),
        sa.Column("usage", JSONB(), nullable=False, server_default="{}"),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
    )
    op.create_index(
        "ix_messages_conversation_created", "messages", ["conversation_id", "created_at"]
    )
    op.create_index("ix_messages_owner_id", "messages", ["owner_id"])

    op.create_table(
        "feedback",
        sa.Column(
            "message_id",
            sa.Uuid(),
            sa.ForeignKey("messages.id", ondelete="CASCADE"),
            primary_key=True,
        ),
        _fk("owner_id", "users.id"),
        sa.Column("rating", sa.SmallInteger(), nullable=False),
        sa.Column("comment", sa.Text(), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.CheckConstraint("rating IN (-1, 1)", name="ck_feedback_rating"),
    )
    op.create_index("ix_feedback_owner_id", "feedback", ["owner_id"])

    for table in ("chunks", "jobs", "conversations", "messages", "feedback"):
        op.execute(f"ALTER TABLE {_qualified(table)} ENABLE ROW LEVEL SECURITY")


def downgrade() -> None:
    for table in ("feedback", "messages", "conversations", "jobs", "chunks"):
        op.drop_table(table)
    for column in ("chunk_count", "summary", "title", "stage"):
        op.drop_column("documents", column, schema=_schema())
