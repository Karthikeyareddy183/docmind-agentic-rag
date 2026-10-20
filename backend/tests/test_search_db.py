"""Hybrid search against real Postgres (pgvector + full-text), with a deterministic fake embedder."""

import hashlib
import math
import uuid

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from docmind.config import Settings
from docmind.db.models import EMBEDDING_DIMENSION, Chunk, ChunkType, Document, DocumentStatus, User
from docmind.db.session import create_engine, create_session_factory
from docmind.retrieval.search import HybridSearcher

pytestmark = pytest.mark.db


class FakeEmbedder:
    """Bag-of-words hashing into a unit vector: similar text → similar vector, no model needed."""

    model_id = "fake"
    dimension = EMBEDDING_DIMENSION

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dimension
        for word in text.lower().split():
            v[int(hashlib.md5(word.strip(".,?").encode()).hexdigest(), 16) % self.dimension] += 1.0  # noqa: S324
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    async def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


@pytest.fixture
async def factory(clean_db: None, migrated_database: Settings) -> async_sessionmaker[AsyncSession]:
    engine = create_engine(migrated_database)
    yield create_session_factory(engine)
    await engine.dispose()


async def _seed(
    factory: async_sessionmaker[AsyncSession], email: str, texts: list[str]
) -> tuple[uuid.UUID, uuid.UUID]:
    embedder = FakeEmbedder()
    async with factory() as session:
        user = User(id=uuid.uuid4(), email=email, password_hash="x")
        doc = Document(
            id=uuid.uuid4(),
            owner_id=user.id,
            filename=f"{email}.pdf",
            content_type="application/pdf",
            size_bytes=1,
            sha256=uuid.uuid4().hex,
            storage_key=uuid.uuid4().hex,
            status=DocumentStatus.READY,
        )
        session.add_all([user, doc])
        await session.flush()
        for i, (text, vec) in enumerate(
            zip(texts, await embedder.embed_documents(texts), strict=True)
        ):
            session.add(
                Chunk(
                    document_id=doc.id,
                    owner_id=user.id,
                    ordinal=i,
                    chunk_type=ChunkType.TEXT,
                    content=text,
                    context_header=f"{email} — page {i + 1}",
                    section_path="",
                    page_start=i + 1,
                    page_end=i + 1,
                    bboxes=[],
                    token_count=len(text.split()),
                    data={},
                    embedding=vec,
                    embedding_model="fake",
                )
            )
        await session.commit()
        return user.id, doc.id


async def test_hybrid_search_ranks_and_scopes_by_owner(
    factory: async_sessionmaker[AsyncSession], migrated_database: Settings
) -> None:
    alice, alice_doc = await _seed(
        factory,
        "alice@example.com",
        [
            "Employees receive 24 days of paid annual leave per year.",
            "Total revenue grew 18% to 482.6 million dollars.",
            "Passwords rotate every 90 days and require MFA.",
        ],
    )
    _bob, _ = await _seed(
        factory,
        "bob@example.com",
        [
            "Annual leave for contractors is 12 days.",
            "Bob's secret revenue figure is 999 million.",
        ],
    )
    settings = migrated_database.model_copy(update={"rerank_enabled": False, "context_chunks": 5})
    async with factory() as session:
        searcher = HybridSearcher(session, settings, FakeEmbedder(), None)
        result = await searcher.search(alice, "How many days of annual leave do employees get?")

    assert result.chunks, "expected results"
    assert result.chunks[0].content.startswith("Employees receive 24 days")
    assert all(c.document_id == alice_doc for c in result.chunks), "leaked another user's chunks"
    assert not any("Bob" in c.content or "contractors" in c.content for c in result.chunks)
    top = result.chunks[0]
    assert top.dense_rank is not None and top.keyword_rank is not None  # found by both retrievers
    assert 0.0 < top.score <= 1.0


async def test_document_filter_and_dense_only_mode(
    factory: async_sessionmaker[AsyncSession], migrated_database: Settings
) -> None:
    alice, doc_a = await _seed(factory, "carol@example.com", ["Revenue was 482.6 million."])
    async with factory() as session:  # second document for the same user
        doc_b = Document(
            id=uuid.uuid4(),
            owner_id=alice,
            filename="other.pdf",
            content_type="application/pdf",
            size_bytes=1,
            sha256=uuid.uuid4().hex,
            storage_key=uuid.uuid4().hex,
            status=DocumentStatus.READY,
        )
        session.add(doc_b)
        await session.flush()
        session.add(
            Chunk(
                document_id=doc_b.id,
                owner_id=alice,
                ordinal=0,
                chunk_type=ChunkType.TEXT,
                content="Revenue was 301.4 million.",
                context_header="other",
                section_path="",
                page_start=1,
                page_end=1,
                bboxes=[],
                token_count=4,
                data={},
                embedding=(await FakeEmbedder().embed_documents(["Revenue was 301.4 million."]))[0],
                embedding_model="fake",
            )
        )
        await session.commit()

    settings = migrated_database.model_copy(
        update={"retrieval_mode": "dense", "rerank_enabled": False}
    )
    async with factory() as session:
        searcher = HybridSearcher(session, settings, FakeEmbedder(), None)
        scoped = await searcher.search(alice, "revenue", document_ids=[doc_b.id])
        assert {c.document_id for c in scoped.chunks} == {doc_b.id}
        dense_only = await searcher.search(alice, "revenue")
        assert all(c.keyword_rank is None for c in dense_only.chunks)
        assert {c.document_id for c in dense_only.chunks} == {doc_a, doc_b.id}


class QuotaExhaustedEmbedder(FakeEmbedder):
    async def embed_query(self, text: str) -> list[float]:
        raise RuntimeError("429 RESOURCE_EXHAUSTED: EmbedContentRequestsPerDay")


async def test_falls_back_to_keyword_search_when_query_embedding_fails(
    factory: async_sessionmaker[AsyncSession], migrated_database: Settings
) -> None:
    user, _ = await _seed(
        factory,
        "dave@example.com",
        ["Employees receive 24 days of paid annual leave.", "Revenue grew 18%."],
    )
    settings = migrated_database.model_copy(update={"rerank_enabled": False})
    async with factory() as session:
        searcher = HybridSearcher(session, settings, QuotaExhaustedEmbedder(), None)
        result = await searcher.search(user, "annual leave days")

    assert result.chunks[0].content.startswith("Employees receive 24 days")
    top = result.chunks[0]
    assert top.dense_rank is None and top.keyword_rank == 1
    # Scored like an unembedded chunk, so it still clears the relevance threshold
    assert top.score == pytest.approx(0.5)
