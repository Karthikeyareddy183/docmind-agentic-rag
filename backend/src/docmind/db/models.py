"""ORM models. Every user-owned table carries ``owner_id`` so queries can always be scoped."""

import enum
import uuid
from datetime import datetime
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Computed,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Index,
    Integer,
    SmallInteger,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# Fixed by the vector column; changing models of a different size needs a migration + re-embed
EMBEDDING_DIMENSION = 1024


def _str_enum(enum_cls: type[enum.StrEnum], name: str) -> Enum:
    # VARCHAR + CHECK rather than a native enum: no schema-bound type to manage
    return Enum(
        enum_cls,
        name=name,
        native_enum=False,
        create_constraint=True,
        length=20,
        values_callable=lambda e: [m.value for m in e],
    )


class Base(DeclarativeBase):
    pass


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )


class User(TimestampMixin, Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))

    documents: Mapped[list["Document"]] = relationship(
        back_populates="owner", cascade="all, delete-orphan", passive_deletes=True
    )


class DocumentStatus(enum.StrEnum):
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class Document(TimestampMixin, Base):
    __tablename__ = "documents"
    __table_args__ = (
        # One copy of the same file per user; re-uploads return the existing document
        Index("uq_documents_owner_sha256", "owner_id", "sha256", unique=True),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    owner_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    filename: Mapped[str] = mapped_column(String(255))
    content_type: Mapped[str] = mapped_column(String(100))
    size_bytes: Mapped[int] = mapped_column(BigInteger)
    sha256: Mapped[str] = mapped_column(String(64))
    storage_key: Mapped[str] = mapped_column(String(512))
    status: Mapped[DocumentStatus] = mapped_column(
        _str_enum(DocumentStatus, "document_status"), default=DocumentStatus.UPLOADED
    )
    pages: Mapped[int | None] = mapped_column(default=None)
    progress: Mapped[float] = mapped_column(default=0.0)
    stage: Mapped[str | None] = mapped_column(String(40), default=None)
    error: Mapped[str | None] = mapped_column(Text, default=None)
    title: Mapped[str | None] = mapped_column(String(500), default=None)
    summary: Mapped[str | None] = mapped_column(Text, default=None)
    chunk_count: Mapped[int] = mapped_column(default=0)
    # Two-stage indexing: chunks are keyword-searchable as soon as they are written; this
    # counts how many also have a vector for semantic search
    embedded_chunks: Mapped[int] = mapped_column(default=0)

    owner: Mapped[User] = relationship(back_populates="documents")


class ChunkType(enum.StrEnum):
    TEXT = "text"
    TABLE = "table"
    FIGURE = "figure"


class Chunk(Base):
    """A retrievable unit: a run of text within one section, one table, or one figure."""

    __tablename__ = "chunks"
    __table_args__ = (
        Index("ix_chunks_owner_document", "owner_id", "document_id"),
        Index(
            "ix_chunks_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
        Index("ix_chunks_tsv", "tsv", postgresql_using="gin"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True
    )
    # Denormalised from documents so retrieval filters on one table
    owner_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    ordinal: Mapped[int] = mapped_column(Integer)
    chunk_type: Mapped[ChunkType] = mapped_column(_str_enum(ChunkType, "chunk_type"))
    # What the LLM sees and the UI quotes (table markdown, figure description, ...)
    content: Mapped[str] = mapped_column(Text)
    # "Title › Section — page N": prepended for embedding and keyword search only
    context_header: Mapped[str] = mapped_column(Text, default="")
    section_path: Mapped[str] = mapped_column(Text, default="")
    page_start: Mapped[int] = mapped_column(SmallInteger)
    page_end: Mapped[int] = mapped_column(SmallInteger)
    # [{"page": n, "x0":..,"y0":..,"x1":..,"y1":..}] as fractions of the page, origin top-left
    bboxes: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, default=list)
    token_count: Mapped[int] = mapped_column(Integer, default=0)
    # Structured payload: table cells for numeric tools, figure caption, ...
    data: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    embedding: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIMENSION))
    embedding_model: Mapped[str] = mapped_column(String(200), default="")
    tsv: Mapped[Any] = mapped_column(
        TSVECTOR,
        Computed(
            "setweight(to_tsvector('english', coalesce(context_header, '')), 'B') || "
            "setweight(to_tsvector('english', coalesce(content, '')), 'A')",
            persisted=True,
        ),
    )


class JobStatus(enum.StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class Job(TimestampMixin, Base):
    """Postgres-backed work queue (claimed with ``FOR UPDATE SKIP LOCKED``)."""

    __tablename__ = "jobs"
    __table_args__ = (
        Index(
            "ix_jobs_claimable",
            "status",
            "run_after",
            postgresql_where="status = 'queued'",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    kind: Mapped[str] = mapped_column(String(50))
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    status: Mapped[JobStatus] = mapped_column(
        _str_enum(JobStatus, "job_status"), default=JobStatus.QUEUED
    )
    attempts: Mapped[int] = mapped_column(SmallInteger, default=0)
    max_attempts: Mapped[int] = mapped_column(SmallInteger, default=3)
    run_after: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    locked_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    locked_by: Mapped[str | None] = mapped_column(String(100))
    last_error: Mapped[str | None] = mapped_column(Text)


class Conversation(TimestampMixin, Base):
    __tablename__ = "conversations"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    owner_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    title: Mapped[str] = mapped_column(String(200), default="New conversation")
    # Restrict retrieval to these documents (empty = all of the user's documents)
    document_ids: Mapped[list[str]] = mapped_column(JSONB, default=list)

    messages: Mapped[list["Message"]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="Message.created_at",
    )


class MessageRole(enum.StrEnum):
    USER = "user"
    ASSISTANT = "assistant"


class Message(Base):
    __tablename__ = "messages"
    __table_args__ = (Index("ix_messages_conversation_created", "conversation_id", "created_at"),)

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE")
    )
    owner_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    role: Mapped[MessageRole] = mapped_column(_str_enum(MessageRole, "message_role"))
    content: Mapped[str] = mapped_column(Text)
    citations: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, default=list)
    confidence: Mapped[float | None] = mapped_column(Float)
    # Agent steps (tool calls, retrieval stats, verification) for the UI trace view
    trace: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, default=list)
    model: Mapped[str | None] = mapped_column(String(200))
    usage: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    conversation: Mapped[Conversation] = relationship(back_populates="messages")


class Feedback(Base):
    __tablename__ = "feedback"

    message_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("messages.id", ondelete="CASCADE"), primary_key=True
    )
    owner_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    rating: Mapped[int] = mapped_column(SmallInteger)  # +1 / -1
    comment: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
