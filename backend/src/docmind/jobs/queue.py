"""A small, durable job queue on Postgres.

Jobs are inserted in the same transaction as the data they refer to (no lost or phantom jobs),
claimed with ``FOR UPDATE SKIP LOCKED`` so many workers can poll safely, and leased: a job whose
worker died is reclaimed once its lease expires. Failed attempts retry with exponential backoff.
"""

import uuid
from collections.abc import Sequence
from datetime import timedelta
from typing import Any

from sqlalchemy import and_, func, or_, select, true, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from docmind.db.models import Job, JobStatus

INGEST_DOCUMENT = "ingest_document"
EMBED_DOCUMENT = "embed_document"


def enqueue(
    session: AsyncSession, kind: str, payload: dict[str, Any], max_attempts: int = 3
) -> Job:
    """Add a job to the session; it becomes visible when the caller commits."""
    job = Job(kind=kind, payload=payload, max_attempts=max_attempts)
    session.add(job)
    return job


async def claim(
    factory: async_sessionmaker[AsyncSession],
    worker_id: str,
    lease_seconds: int,
    kinds: Sequence[str] | None = None,
) -> Job | None:
    """Claim the oldest runnable job, optionally only of ``kinds`` (a worker "lane")."""
    # Database clock, not the app's: run_after defaults to the DB's now(), and app/DB clocks
    # can disagree by seconds, which made freshly queued jobs briefly unclaimable
    now = func.now()
    claimable = (
        select(Job.id)
        .where(
            or_(
                and_(Job.status == JobStatus.QUEUED, Job.run_after <= now),
                # A running job whose lease expired belongs to a dead worker
                and_(Job.status == JobStatus.RUNNING, Job.locked_until < now),
            ),
            Job.kind.in_(kinds) if kinds else true(),
        )
        .order_by(Job.run_after, Job.created_at)
        .limit(1)
        .with_for_update(skip_locked=True)
        .scalar_subquery()
    )
    async with factory() as session:
        job = await session.scalar(
            update(Job)
            .where(Job.id == claimable)
            .values(
                status=JobStatus.RUNNING,
                attempts=Job.attempts + 1,
                locked_by=worker_id,
                locked_until=now + timedelta(seconds=lease_seconds),
            )
            .returning(Job)
        )
        await session.commit()
        return job


async def extend_lease(
    factory: async_sessionmaker[AsyncSession], job_id: uuid.UUID, lease_seconds: int
) -> None:
    """Heartbeat for long jobs, so the lease does not expire while work is still progressing."""
    async with factory() as session:
        await session.execute(
            update(Job)
            .where(Job.id == job_id, Job.status == JobStatus.RUNNING)
            .values(locked_until=func.now() + timedelta(seconds=lease_seconds))
        )
        await session.commit()


async def defer(
    factory: async_sessionmaker[AsyncSession], job: Job, seconds: float, reason: str
) -> None:
    """Put a job back to run later without counting the attempt (e.g. provider quota reset)."""
    async with factory() as session:
        await session.execute(
            update(Job)
            .where(Job.id == job.id)
            .values(
                status=JobStatus.QUEUED,
                attempts=Job.attempts - 1,
                run_after=func.now() + timedelta(seconds=seconds),
                locked_until=None,
                last_error=reason[:4000],
            )
        )
        await session.commit()


async def complete(factory: async_sessionmaker[AsyncSession], job_id: uuid.UUID) -> None:
    async with factory() as session:
        await session.execute(
            update(Job)
            .where(Job.id == job_id)
            .values(status=JobStatus.DONE, locked_until=None, last_error=None)
        )
        await session.commit()


async def fail(
    factory: async_sessionmaker[AsyncSession], job: Job, error: str, retryable: bool = True
) -> bool:
    """Record a failure. Returns True if the job will be retried."""
    retry = retryable and job.attempts < job.max_attempts
    values: dict[str, Any] = {"last_error": error[:4000], "locked_until": None}
    if retry:
        backoff = timedelta(seconds=30 * 2 ** (job.attempts - 1))
        values |= {"status": JobStatus.QUEUED, "run_after": func.now() + backoff}
    else:
        values |= {"status": JobStatus.FAILED}
    async with factory() as session:
        await session.execute(update(Job).where(Job.id == job.id).values(**values))
        await session.commit()
    return retry
