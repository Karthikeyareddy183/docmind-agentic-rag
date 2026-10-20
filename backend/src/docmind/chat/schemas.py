import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from docmind.agent.citations import Citation


class ConversationCreate(BaseModel):
    title: str | None = Field(default=None, max_length=200)
    document_ids: list[uuid.UUID] = Field(default_factory=list, max_length=50)


class ConversationOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    title: str
    document_ids: list[uuid.UUID]
    created_at: datetime
    updated_at: datetime


class MessageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    role: str
    content: str
    citations: list[dict[str, Any]]
    confidence: float | None
    trace: list[dict[str, Any]]
    model: str | None
    usage: dict[str, Any]
    latency_ms: int | None
    created_at: datetime


class ConversationDetail(ConversationOut):
    messages: list[MessageOut]


class AskRequest(BaseModel):
    question: str = Field(min_length=2, max_length=2000)
    document_ids: list[uuid.UUID] = Field(default_factory=list, max_length=50)


class AskResponse(BaseModel):
    answer: str
    answerable: bool
    citations: list[Citation]
    confidence: float
    verified: bool | None
    computations: list[dict[str, Any]]
    trace: list[dict[str, Any]]
    usage: dict[str, Any]
    model: str
    latency_ms: int


class MessageCreate(BaseModel):
    content: str = Field(min_length=2, max_length=2000)


class FeedbackCreate(BaseModel):
    rating: int = Field(description="+1 or -1")
    comment: str | None = Field(default=None, max_length=2000)


class SearchRequest(BaseModel):
    query: str = Field(min_length=2, max_length=1000)
    document_ids: list[uuid.UUID] = Field(default_factory=list, max_length=50)
    limit: int = Field(default=8, ge=1, le=20)


class SearchHit(BaseModel):
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    filename: str
    document_title: str
    chunk_type: str
    page: int
    section: str
    content: str
    score: float
    dense_rank: int | None
    keyword_rank: int | None


class SearchResponse(BaseModel):
    hits: list[SearchHit]
    timings_ms: dict[str, float]
