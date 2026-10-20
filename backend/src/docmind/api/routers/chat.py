"""Question answering: one-shot ``/ask`` (JSON) and conversations with streamed replies (SSE)."""

import json
import logging
import uuid
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Request, status
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from sse_starlette.sse import EventSourceResponse

from docmind.api.deps import CurrentUser, SessionDep, SettingsDep
from docmind.api.errors import AppError, NotFoundError
from docmind.chat import service
from docmind.chat.schemas import (
    AskRequest,
    AskResponse,
    ConversationCreate,
    ConversationDetail,
    ConversationOut,
    FeedbackCreate,
    MessageCreate,
    SearchHit,
    SearchRequest,
    SearchResponse,
)
from docmind.db.models import Conversation, Document
from docmind.retrieval.search import HybridSearcher

router = APIRouter(tags=["chat"])
logger = logging.getLogger(__name__)


class InvalidRatingError(AppError):
    status_code = 422
    code = "invalid_rating"


async def _owned_document_ids(
    session: SessionDep, owner_id: uuid.UUID, ids: list[uuid.UUID]
) -> list[uuid.UUID]:
    if not ids:
        return []
    found = set(
        await session.scalars(
            select(Document.id).where(Document.owner_id == owner_id, Document.id.in_(ids))
        )
    )
    return [i for i in ids if i in found]


@router.post("/ask", response_model=AskResponse)
async def ask(
    body: AskRequest,
    request: Request,
    user: CurrentUser,
    session: SessionDep,
    settings: SettingsDep,
) -> Any:
    """Answer one question without conversation state (evaluation harness, MCP server)."""
    doc_ids = await _owned_document_ids(session, user.id, body.document_ids)
    deps = service.agent_deps(
        settings, request.app.state.session_factory, request.app.state.llm, user.id, doc_ids
    )
    result = await service.ask(deps, body.question)
    return result.model_dump()


@router.post("/search", response_model=SearchResponse)
async def search(
    body: SearchRequest,
    request: Request,
    user: CurrentUser,
    session: SessionDep,
    settings: SettingsDep,
) -> Any:
    """Hybrid retrieval only (no LLM): the passages the agent would see for this query."""
    doc_ids = await _owned_document_ids(session, user.id, body.document_ids)
    deps = service.agent_deps(
        settings, request.app.state.session_factory, request.app.state.llm, user.id, doc_ids
    )
    searcher = HybridSearcher(session, settings, deps.embedder, deps.reranker)
    result = await searcher.search(
        user.id, body.query, document_ids=doc_ids or None, limit=body.limit
    )
    return SearchResponse(
        hits=[
            SearchHit(
                chunk_id=c.chunk_id,
                document_id=c.document_id,
                filename=c.filename,
                document_title=c.document_title,
                chunk_type=c.chunk_type,
                page=c.page_start,
                section=c.section_path,
                content=c.content,
                score=round(c.score, 4),
                dense_rank=c.dense_rank,
                keyword_rank=c.keyword_rank,
            )
            for c in result.chunks
        ],
        timings_ms={k: round(v, 1) for k, v in result.timings_ms.items()},
    )


@router.post("/conversations", response_model=ConversationOut, status_code=status.HTTP_201_CREATED)
async def create_conversation(
    body: ConversationCreate, user: CurrentUser, session: SessionDep
) -> Conversation:
    doc_ids = await _owned_document_ids(session, user.id, body.document_ids)
    conversation = Conversation(
        owner_id=user.id,
        title=body.title or "New conversation",
        document_ids=[str(d) for d in doc_ids],
    )
    session.add(conversation)
    await session.commit()
    await session.refresh(conversation)
    return conversation


@router.get("/conversations", response_model=list[ConversationOut])
async def list_conversations(user: CurrentUser, session: SessionDep) -> list[Conversation]:
    rows = await session.scalars(
        select(Conversation)
        .where(Conversation.owner_id == user.id)
        .order_by(Conversation.updated_at.desc())
        .limit(100)
    )
    return list(rows)


@router.get("/conversations/{conversation_id}", response_model=ConversationDetail)
async def get_conversation(
    conversation_id: uuid.UUID, user: CurrentUser, session: SessionDep
) -> Conversation:
    conversation = await session.scalar(
        select(Conversation)
        .where(Conversation.id == conversation_id, Conversation.owner_id == user.id)
        .options(selectinload(Conversation.messages))
    )
    if conversation is None:
        raise NotFoundError("Conversation not found")
    return conversation


@router.delete("/conversations/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation(
    conversation_id: uuid.UUID, user: CurrentUser, session: SessionDep
) -> None:
    conversation = await service.get_conversation(session, user.id, conversation_id)
    await session.delete(conversation)
    await session.commit()


@router.post("/conversations/{conversation_id}/messages")
async def send_message(
    conversation_id: uuid.UUID,
    body: MessageCreate,
    request: Request,
    user: CurrentUser,
    session: SessionDep,
    settings: SettingsDep,
) -> EventSourceResponse:
    """Stream the agent's work as Server-Sent Events.

    Events: ``user_message`` · ``step`` (agent actions) · ``token`` (answer text) ·
    ``revision`` (fact-check rewrote the answer) · ``final`` (structured result) · ``error``.
    """
    await service.get_conversation(session, user.id, conversation_id)  # 404 before streaming starts
    factory = request.app.state.session_factory
    deps = service.agent_deps(settings, factory, request.app.state.llm, user.id)

    async def events() -> AsyncIterator[dict[str, str]]:
        try:
            async for event in service.stream_reply(factory, deps, conversation_id, body.content):
                yield {"event": event["type"], "data": json.dumps(event, default=str)}
        except Exception:
            logger.exception("Streaming reply failed")
            yield {
                "event": "error",
                "data": json.dumps(
                    {
                        "type": "error",
                        "message": "The assistant failed to answer. Please try again.",
                    }
                ),
            }

    return EventSourceResponse(events(), ping=15)


@router.post("/messages/{message_id}/feedback", status_code=status.HTTP_204_NO_CONTENT)
async def feedback(
    message_id: uuid.UUID, body: FeedbackCreate, user: CurrentUser, session: SessionDep
) -> None:
    if body.rating not in (-1, 1):
        raise InvalidRatingError("rating must be 1 or -1")
    await service.save_feedback(session, user.id, message_id, body.rating, body.comment)
