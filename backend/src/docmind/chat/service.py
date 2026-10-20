"""Conversation persistence and agent execution."""

import logging
import uuid
from collections.abc import AsyncIterator
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from docmind.agent.graph import AgentDeps, AgentResult, run_agent
from docmind.api.errors import AppError, NotFoundError
from docmind.config import Settings
from docmind.db.models import Conversation, Feedback, Message, MessageRole
from docmind.llm.gateway import LLMGateway
from docmind.retrieval.models import get_embedder, get_reranker

logger = logging.getLogger(__name__)

HISTORY_TURNS = 6


class ModelsLoadingError(AppError):
    status_code = 503
    code = "models_loading"


def agent_deps(
    settings: Settings,
    factory: async_sessionmaker[AsyncSession],
    llm: LLMGateway,
    owner_id: uuid.UUID,
    document_ids: list[uuid.UUID] | None = None,
) -> AgentDeps:
    return AgentDeps(
        settings=settings,
        factory=factory,
        llm=llm,
        embedder=get_embedder(
            settings.embedding_model, settings.embedding_dimension, settings.embedding_batch_size
        ),
        reranker=get_reranker(settings.reranker_model) if settings.rerank_enabled else None,
        owner_id=owner_id,
        document_ids=list(document_ids or []),
    )


async def get_conversation(
    session: AsyncSession, owner_id: uuid.UUID, conversation_id: uuid.UUID
) -> Conversation:
    conversation = await session.scalar(
        select(Conversation).where(
            Conversation.id == conversation_id, Conversation.owner_id == owner_id
        )
    )
    if conversation is None:
        raise NotFoundError("Conversation not found")
    return conversation


async def ask(
    deps: AgentDeps, question: str, history: list[dict[str, str]] | None = None
) -> AgentResult:
    result: AgentResult | None = None
    async for event in run_agent(question, deps, history):
        if event["type"] == "final":
            result = event["result"]
    assert result is not None  # noqa: S101 (run_agent always yields a final event)
    return result


async def stream_reply(
    factory: async_sessionmaker[AsyncSession],
    deps: AgentDeps,
    conversation_id: uuid.UUID,
    content: str,
) -> AsyncIterator[dict[str, Any]]:
    """Persist the user turn, run the agent while yielding events, persist the assistant turn."""
    async with factory() as session:
        conversation = await get_conversation(session, deps.owner_id, conversation_id)
        prior = list(
            await session.scalars(
                select(Message)
                .where(Message.conversation_id == conversation_id)
                .order_by(Message.created_at.desc())
                .limit(HISTORY_TURNS)
            )
        )
        history = [{"role": m.role.value, "content": m.content} for m in reversed(prior)]
        user_message = Message(
            conversation_id=conversation_id,
            owner_id=deps.owner_id,
            role=MessageRole.USER,
            content=content,
        )
        session.add(user_message)
        if not prior:
            conversation.title = content[:80]
        deps.document_ids = [uuid.UUID(str(d)) for d in conversation.document_ids]
        await session.commit()
        yield {"type": "user_message", "id": str(user_message.id)}

    result: AgentResult | None = None
    async for event in run_agent(content, deps, history):
        if event["type"] == "final":
            result = event["result"]
        else:
            yield event
    assert result is not None  # noqa: S101

    async with factory() as session:
        assistant = Message(
            conversation_id=conversation_id,
            owner_id=deps.owner_id,
            role=MessageRole.ASSISTANT,
            content=result.answer,
            citations=[c.model_dump() for c in result.citations],
            confidence=result.confidence,
            trace=result.trace,
            model=result.model,
            usage=result.usage,
            latency_ms=result.latency_ms,
        )
        session.add(assistant)
        conv = await session.get(Conversation, conversation_id)
        if conv is not None:
            conv.title = conv.title  # touch updated_at via onupdate
        await session.commit()
        yield {"type": "final", "message_id": str(assistant.id), **result.model_dump(mode="json")}


async def save_feedback(
    session: AsyncSession,
    owner_id: uuid.UUID,
    message_id: uuid.UUID,
    rating: int,
    comment: str | None,
) -> None:
    message = await session.scalar(
        select(Message).where(
            Message.id == message_id,
            Message.owner_id == owner_id,
            Message.role == MessageRole.ASSISTANT,
        )
    )
    if message is None:
        raise NotFoundError("Message not found")
    existing = await session.get(Feedback, message_id)
    if existing is None:
        session.add(
            Feedback(message_id=message_id, owner_id=owner_id, rating=rating, comment=comment)
        )
    else:
        existing.rating, existing.comment = rating, comment
    await session.commit()
