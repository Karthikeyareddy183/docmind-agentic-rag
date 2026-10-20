"""Two-stage indexing: chunks are keyword-searchable before vectors exist; the embed job fills in."""

import uuid

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from docmind.config import Settings
from docmind.db.models import Chunk, ChunkType, Document, DocumentStatus, User
from docmind.db.session import create_engine, create_session_factory
from docmind.ingestion import pipeline
from docmind.jobs import queue
from docmind.retrieval.search import HybridSearcher
from tests.test_search_db import FakeEmbedder

pytestmark = pytest.mark.db

TEXTS = [
    "Endometriosis is the presence of endometrial tissue outside the uterine cavity.",
    "Fibroids are benign smooth muscle tumours of the uterus.",
    "Cervical screening uses cytology and HPV testing.",
]


@pytest.fixture
async def factory(clean_db: None, migrated_database: Settings) -> async_sessionmaker[AsyncSession]:
    engine = create_engine(migrated_database)
    yield create_session_factory(engine)
    await engine.dispose()


async def _seed_unembedded(
    factory: async_sessionmaker[AsyncSession],
) -> tuple[uuid.UUID, uuid.UUID]:
    async with factory() as session:
        user = User(id=uuid.uuid4(), email=f"{uuid.uuid4().hex[:8]}@example.com", password_hash="x")
        doc = Document(
            id=uuid.uuid4(),
            owner_id=user.id,
            filename="gyn.pdf",
            content_type="application/pdf",
            size_bytes=1,
            sha256=uuid.uuid4().hex,
            storage_key=uuid.uuid4().hex,
            status=DocumentStatus.READY,
            chunk_count=len(TEXTS),
            embedded_chunks=0,
        )
        session.add_all([user, doc])
        await session.flush()
        for i, text in enumerate(TEXTS):
            session.add(
                Chunk(
                    document_id=doc.id,
                    owner_id=user.id,
                    ordinal=i,
                    chunk_type=ChunkType.TEXT,
                    content=text,
                    context_header=f"Gynecology — page {i + 1}",
                    section_path="",
                    page_start=i + 1,
                    page_end=i + 1,
                    bboxes=[],
                    token_count=10,
                    data={},
                    embedding=None,
                    embedding_model="",
                )
            )
        queue.enqueue(session, queue.EMBED_DOCUMENT, {"document_id": str(doc.id)})
        await session.commit()
        return user.id, doc.id


async def test_keyword_search_works_before_embedding_then_job_fills_vectors(
    factory: async_sessionmaker[AsyncSession],
    migrated_database: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner, doc_id = await _seed_unembedded(factory)
    settings = migrated_database.model_copy(update={"embed_batch_chunks": 2})
    embedder = FakeEmbedder()

    # Stage 1 state: no vectors, but keyword search already finds the right chunk
    async with factory() as session:
        result = await HybridSearcher(session, settings, embedder, None).search(
            owner, "endometriosis"
        )
    assert result.chunks and result.chunks[0].content.startswith("Endometriosis")
    assert not result.chunks[0].has_vector
    assert result.chunks[0].score >= settings.min_relevance  # reaches the LLM context

    # Stage 2: the embed job (batches of 2) adds vectors and tracks progress
    monkeypatch.setattr(pipeline, "get_embedder", lambda *_a, **_k: embedder)
    job = await queue.claim(factory, "w", 60)
    assert job is not None and job.kind == queue.EMBED_DOCUMENT
    stats = await pipeline.embed_document(doc_id, settings=settings, factory=factory, job_id=job.id)
    assert stats["embedded"] == len(TEXTS)

    async with factory() as session:
        missing = await session.scalar(
            select(func.count())
            .select_from(Chunk)
            .where(Chunk.document_id == doc_id, Chunk.embedding.is_(None))
        )
        document = await session.get(Document, doc_id)
    assert missing == 0
    assert document is not None and document.embedded_chunks == len(TEXTS)

    async with factory() as session:
        result = await HybridSearcher(session, settings, embedder, None).search(
            owner, "endometriosis"
        )
    assert result.chunks[0].has_vector
    assert result.chunks[0].dense_rank is not None

    # Re-running is a no-op (resumable / idempotent)
    again = await pipeline.embed_document(doc_id, settings=settings, factory=factory, job_id=job.id)
    assert again["embedded"] == 0
