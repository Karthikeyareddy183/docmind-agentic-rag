"""Ingestion: parse → overview → chunk → enrich → embed → index, reporting progress."""

import asyncio
import logging
import re
import time
import uuid
from typing import Any

import litellm
from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from docmind.config import Settings
from docmind.db.models import Chunk, ChunkType, Document, DocumentStatus
from docmind.documents.storage import Storage
from docmind.ingestion.chunker import chunk_document
from docmind.ingestion.enrich import document_overview, enrich_chunks
from docmind.ingestion.parser import parse_document
from docmind.jobs.queue import EMBED_DOCUMENT, enqueue, extend_lease
from docmind.llm.gateway import LLMGateway, Usage
from docmind.retrieval.models import get_embedder

logger = logging.getLogger(__name__)


class IngestionError(Exception):
    """A non-retryable problem with the document itself (shown to the user)."""


async def _set_progress(
    factory: async_sessionmaker[AsyncSession], document_id: uuid.UUID, **values: Any
) -> None:
    async with factory() as session:
        await session.execute(update(Document).where(Document.id == document_id).values(**values))
        await session.commit()


async def ingest_document(
    document_id: uuid.UUID,
    *,
    settings: Settings,
    factory: async_sessionmaker[AsyncSession],
    storage: Storage,
    llm: LLMGateway,
) -> dict[str, Any]:
    started = time.perf_counter()
    async with factory() as session:
        document = await session.get(Document, document_id)
        if document is None:
            raise IngestionError("Document no longer exists")
        owner_id, filename, storage_key = document.owner_id, document.filename, document.storage_key
        content_type = document.content_type

    async def progress(stage: str, value: float) -> None:
        await _set_progress(
            factory,
            document_id,
            status=DocumentStatus.PROCESSING,
            stage=stage,
            progress=value,
            error=None,
        )

    usage = Usage()

    await progress("parsing", 0.05)
    path = storage.path(storage_key)
    if not path.exists():
        raise IngestionError("Uploaded file is missing from storage")
    try:
        parsed, strategy = await asyncio.to_thread(
            parse_document,
            path,
            content_type=content_type,
            max_pages=settings.max_pages,
            fast_parse_min_pages=settings.fast_parse_min_pages,
            ocr_min_chars=settings.ocr_min_chars,
        )
    except Exception as exc:
        logger.exception("Parse failed", extra={"document_id": str(document_id)})
        raise IngestionError(
            "Could not read this file. Is it a valid, unencrypted PDF or image?"
        ) from exc
    if parsed.text_chars == 0 and not any(e.kind == "figure" for e in parsed.elements):
        raise IngestionError("No readable content was found in this document")

    await progress("summarizing", 0.15)
    overview = await document_overview(parsed, filename, llm, settings, usage)

    await progress("enriching", 0.2)
    drafts = chunk_document(parsed, overview.title, settings.chunk_max_tokens)
    await enrich_chunks(drafts, overview.title, llm, settings, usage)

    # Stage 1: write every chunk now. Postgres full-text indexes them on insert, so the
    # document is keyword-searchable within seconds; vectors are added by the embed job.
    async with factory() as session:
        # Idempotent: a retried job replaces whatever a previous attempt wrote
        await session.execute(delete(Chunk).where(Chunk.document_id == document_id))
        await session.commit()
    batch = max(1, settings.index_batch_size)
    for start in range(0, len(drafts), batch):
        await progress("indexing", 0.25 + 0.7 * start / max(1, len(drafts)))
        async with factory() as session:
            session.add_all(
                Chunk(
                    document_id=document_id,
                    owner_id=owner_id,
                    ordinal=start + i,
                    chunk_type=ChunkType(d.chunk_type),
                    content=d.content,
                    context_header=d.context_header,
                    section_path=d.section_path,
                    page_start=d.page_start,
                    page_end=d.page_end,
                    bboxes=d.bboxes,
                    token_count=d.token_count,
                    data=d.data,
                    embedding=None,
                    embedding_model="",
                )
                for i, d in enumerate(drafts[start : start + batch])
            )
            await session.commit()

    async with factory() as session:
        await session.execute(
            update(Document)
            .where(Document.id == document_id)
            .values(
                status=DocumentStatus.READY,
                stage=None,
                progress=1.0,
                pages=parsed.pages,
                title=overview.title[:500],
                summary=overview.summary,
                chunk_count=len(drafts),
                embedded_chunks=0,
                error=None,
            )
        )
        # Stage 2 runs as its own job: retried/resumed independently of parsing
        enqueue(session, EMBED_DOCUMENT, {"document_id": str(document_id)}, max_attempts=5)
        await session.commit()

    stats = {
        "parser": strategy,
        "pages": parsed.pages,
        "chunks": len(drafts),
        "tables": sum(d.chunk_type == "table" for d in drafts),
        "figures": sum(d.chunk_type == "figure" for d in drafts),
        "seconds": round(time.perf_counter() - started, 1),
        "llm": usage.as_dict(),
    }
    logger.info("Document ingested", extra={"document_id": str(document_id), **stats})
    return stats


async def mark_failed(
    factory: async_sessionmaker[AsyncSession], document_id: uuid.UUID, message: str
) -> None:
    await _set_progress(
        factory, document_id, status=DocumentStatus.FAILED, stage=None, error=message[:1000]
    )


SHORT_WAIT_SECONDS = 60
_RETRY_DELAY = re.compile(r'"retryDelay":\s*"(\d+(?:\.\d+)?)s"')


class EmbeddingDeferredError(Exception):
    """The provider asked us to come back much later (e.g. a daily quota reset)."""

    def __init__(self, seconds: float) -> None:
        super().__init__(f"rate limited; retry in {seconds:.0f}s")
        self.seconds = seconds


DAILY_QUOTA_RETRY_SECONDS = 3600


def _is_rate_limited(exc: Exception) -> bool:
    """Quota/rate errors, however the SDK classifies them (LiteLLM has surfaced Gemini's
    quota errors as BadRequestError, not only RateLimitError)."""
    if isinstance(exc, litellm.exceptions.RateLimitError):
        return True
    text = str(exc)
    return "RESOURCE_EXHAUSTED" in text or "exceeded your current quota" in text


def _retry_after_seconds(exc: Exception) -> float | None:
    """Provider retry hint (Gemini's RetryInfo.retryDelay), if present."""
    match = _RETRY_DELAY.search(str(exc))
    return float(match.group(1)) if match else None


async def embed_document(
    document_id: uuid.UUID,
    *,
    settings: Settings,
    factory: async_sessionmaker[AsyncSession],
    job_id: uuid.UUID,
) -> dict[str, Any]:
    """Stage 2: add vectors to chunks that lack one, in batches; resumable at any point.

    Rate limits are waited out (honouring the provider's retry hints, bounded by
    ``embed_max_wait_seconds``) instead of failing, since keyword search already works.
    """
    started = time.perf_counter()
    embedder = get_embedder(
        settings.embedding_model, settings.embedding_dimension, settings.embedding_batch_size
    )
    waited = 0.0
    embedded_now = 0
    while True:
        async with factory() as session:
            rows = list(
                await session.execute(
                    select(Chunk.id, Chunk.context_header, Chunk.content)
                    .where(
                        Chunk.document_id == document_id,
                        (Chunk.embedding.is_(None)) | (Chunk.embedding_model != embedder.model_id),
                    )
                    .order_by(Chunk.ordinal)
                    .limit(settings.embed_batch_chunks)
                )
            )
        if not rows:
            break
        # Same text as ChunkDraft.index_text: the contextual header is embedded with the chunk
        texts = [f"{header}\n{content}" if header else content for _, header, content in rows]
        try:
            vectors = await embedder.embed_documents(texts)
        except Exception as exc:
            if not _is_rate_limited(exc):
                raise
            hint = _retry_after_seconds(exc)
            if hint is None and "PerDay" in str(exc):
                hint = DAILY_QUOTA_RETRY_SECONDS  # daily quota without a retry hint
            if hint is not None and hint > SHORT_WAIT_SECONDS:
                # e.g. a free-tier daily quota: give the lane back and resume at the reset
                raise EmbeddingDeferredError(hint) from exc
            if waited >= settings.embed_max_wait_seconds:
                raise
            delay = float(hint or SHORT_WAIT_SECONDS)
            logger.info("Embedding rate-limited; waiting", extra={"delay_s": delay})
            await extend_lease(factory, job_id, settings.embed_lease_seconds)
            await asyncio.sleep(delay)
            waited += delay
            continue
        async with factory() as session:
            for (chunk_id, _, _), vector in zip(rows, vectors, strict=True):
                await session.execute(
                    update(Chunk)
                    .where(Chunk.id == chunk_id)
                    .values(embedding=vector, embedding_model=embedder.model_id)
                )
            done = await session.scalar(
                select(func.count())
                .select_from(Chunk)
                .where(Chunk.document_id == document_id, Chunk.embedding_model == embedder.model_id)
            )
            await session.execute(
                update(Document).where(Document.id == document_id).values(embedded_chunks=done or 0)
            )
            await session.commit()
        embedded_now += len(rows)
        await extend_lease(factory, job_id, settings.embed_lease_seconds)

    stats = {
        "embedded": embedded_now,
        "seconds": round(time.perf_counter() - started, 1),
        "rate_limit_wait_s": waited,
    }
    logger.info("Document embedded", extra={"document_id": str(document_id), **stats})
    return stats
