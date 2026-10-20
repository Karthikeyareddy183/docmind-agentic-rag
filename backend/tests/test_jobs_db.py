"""Postgres job queue semantics against a real database."""

import asyncio
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from docmind.config import Settings
from docmind.db.models import Job, JobStatus
from docmind.db.session import create_engine, create_session_factory
from docmind.jobs import queue

pytestmark = pytest.mark.db


@pytest.fixture
async def factory(clean_db: None, migrated_database: Settings) -> async_sessionmaker[AsyncSession]:
    engine = create_engine(migrated_database)
    yield create_session_factory(engine)
    await engine.dispose()


async def _enqueue(
    factory: async_sessionmaker[AsyncSession], n: int = 1, max_attempts: int = 3
) -> None:
    async with factory() as session:
        for i in range(n):
            queue.enqueue(session, "test", {"i": i}, max_attempts=max_attempts)
        await session.commit()


async def test_claim_marks_running_with_lease(factory: async_sessionmaker[AsyncSession]) -> None:
    await _enqueue(factory)
    job = await queue.claim(factory, "worker-a", lease_seconds=60)
    assert job is not None
    assert job.status == JobStatus.RUNNING
    assert job.attempts == 1
    assert job.locked_by == "worker-a"
    assert job.locked_until is not None and job.locked_until > datetime.now(UTC)
    assert await queue.claim(factory, "worker-b", lease_seconds=60) is None


async def test_concurrent_workers_never_share_a_job(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    await _enqueue(factory, n=5)
    claimed = await asyncio.gather(*(queue.claim(factory, f"w{i}", 60) for i in range(8)))
    ids = [j.id for j in claimed if j is not None]
    assert len(ids) == 5
    assert len(set(ids)) == 5


async def test_expired_lease_is_reclaimed(factory: async_sessionmaker[AsyncSession]) -> None:
    await _enqueue(factory)
    job = await queue.claim(factory, "dead-worker", lease_seconds=60)
    assert job is not None
    async with factory() as session:  # simulate the worker dying long ago
        await session.execute(
            update(Job)
            .where(Job.id == job.id)
            .values(locked_until=func.now() - timedelta(seconds=1))
        )
        await session.commit()
    again = await queue.claim(factory, "healthy-worker", lease_seconds=60)
    assert again is not None and again.id == job.id
    assert again.attempts == 2
    assert again.locked_by == "healthy-worker"


async def test_failure_retries_with_backoff_then_gives_up(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    await _enqueue(factory, max_attempts=2)
    job = await queue.claim(factory, "w", 60)
    assert job is not None
    assert await queue.fail(factory, job, "boom") is True
    async with factory() as session:
        stored = await session.scalar(select(Job).where(Job.id == job.id))
    assert stored is not None
    assert stored.status == JobStatus.QUEUED
    assert stored.run_after > datetime.now(UTC)  # backoff: not immediately claimable
    assert await queue.claim(factory, "w", 60) is None

    async with factory() as session:  # fast-forward the backoff
        await session.execute(update(Job).where(Job.id == job.id).values(run_after=func.now()))
        await session.commit()
    job = await queue.claim(factory, "w", 60)
    assert job is not None and job.attempts == 2
    assert await queue.fail(factory, job, "boom again") is False
    async with factory() as session:
        stored = await session.scalar(select(Job).where(Job.id == job.id))
    assert stored is not None
    assert stored.status == JobStatus.FAILED
    assert stored.last_error == "boom again"


async def test_non_retryable_failure_is_final(factory: async_sessionmaker[AsyncSession]) -> None:
    await _enqueue(factory)
    job = await queue.claim(factory, "w", 60)
    assert job is not None
    assert await queue.fail(factory, job, "bad file", retryable=False) is False


async def test_complete(factory: async_sessionmaker[AsyncSession]) -> None:
    await _enqueue(factory)
    job = await queue.claim(factory, "w", 60)
    assert job is not None
    await queue.complete(factory, job.id)
    async with factory() as session:
        stored = await session.scalar(select(Job).where(Job.id == job.id))
    assert stored is not None and stored.status == JobStatus.DONE


async def test_lanes_only_claim_their_kinds(factory: async_sessionmaker[AsyncSession]) -> None:
    async with factory() as session:
        queue.enqueue(session, queue.EMBED_DOCUMENT, {"i": 0})
        queue.enqueue(session, queue.INGEST_DOCUMENT, {"i": 1})
        await session.commit()
    ingest = await queue.claim(factory, "ingest-lane", 60, kinds=(queue.INGEST_DOCUMENT,))
    assert (
        ingest is not None and ingest.kind == queue.INGEST_DOCUMENT
    )  # skipped the older embed job
    assert await queue.claim(factory, "ingest-lane", 60, kinds=(queue.INGEST_DOCUMENT,)) is None
    embed = await queue.claim(factory, "embed-lane", 60, kinds=(queue.EMBED_DOCUMENT,))
    assert embed is not None and embed.kind == queue.EMBED_DOCUMENT


async def test_defer_requeues_without_consuming_an_attempt(
    factory: async_sessionmaker[AsyncSession],
) -> None:
    await _enqueue(factory)
    job = await queue.claim(factory, "w", 60)
    assert job is not None and job.attempts == 1
    await queue.defer(factory, job, 3600, "daily quota")
    async with factory() as session:
        stored = await session.scalar(select(Job).where(Job.id == job.id))
    assert stored is not None
    assert stored.status == JobStatus.QUEUED
    assert stored.attempts == 0
    assert stored.run_after > datetime.now(UTC) + timedelta(minutes=50)
    assert await queue.claim(factory, "w", 60) is None  # not before the quota resets
