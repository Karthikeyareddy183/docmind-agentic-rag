import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict

from docmind.db.models import DocumentStatus


class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    filename: str
    content_type: str
    size_bytes: int
    status: DocumentStatus
    stage: str | None
    pages: int | None
    progress: float
    error: str | None
    title: str | None
    summary: str | None
    chunk_count: int
    embedded_chunks: int
    created_at: datetime
    updated_at: datetime


class DocumentList(BaseModel):
    items: list[DocumentOut]
    total: int


class ChunkOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    ordinal: int
    chunk_type: str
    content: str
    context_header: str
    section_path: str
    page_start: int
    page_end: int
    bboxes: list[dict[str, Any]]
    token_count: int
