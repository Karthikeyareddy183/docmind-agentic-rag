"""Job worker: polls the Postgres queue and runs ingestion.

Run standalone with ``python -m docmind.worker`` (scale by starting more processes), or inside the
API process when ``DOCMIND_RUN_WORKER_IN_API=true`` (single-box dev/demo).
"""

import asyncio
import contextlib
import logging
import os
import socket
import uuid

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from docmind.config import Settings
from docmind.db.models import Job
from docmind.documents.storage import Storage
from docmind.ingestion.pipeline import (
    EmbeddingDeferredError,
    IngestionError,
    embed_document,
    ingest_document,
    mark_failed,
)
from docmind.jobs import queue
from docmind.llm.gateway import LLMGateway

logger = logging.getLogger(__name__)


class Worker:
    def __init__(
        self,
        settings: Settings,
        factory: async_sessionmaker[AsyncSession],
        storage: Storage,
        kinds: tuple[str, ...] | None = None,
    ) -> None:
        self.settings = settings
        # Lane: which job kinds this worker takes (None = all). Separate lanes stop a long
        # background embedding from blocking other users' ingestion.
        self.kinds = kinds
        self.factory = factory
        self.storage = storage
        self.llm = LLMGateway(settings)
        self.worker_id = f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:6]}"
        self._stop = asyncio.Event()

    def stop(self) -> None:
        self._stop.set()

    async def run_once(self) -> bool:
        """Process at most one job. Returns True if a job was found."""
        job = await queue.claim(
            self.factory, self.worker_id, self.settings.job_lease_seconds, self.kinds
        )
        if job is None:
            return False
        log_extra = {"job_id": str(job.id), "kind": job.kind, "attempt": job.attempts}
        logger.info("Job started", extra=log_extra)
        document_id = uuid.UUID(job.payload["document_id"])
        if job.kind == queue.EMBED_DOCUMENT:
            return await self._run_embed(job, document_id, log_extra)
        if job.kind != queue.INGEST_DOCUMENT:
            await queue.fail(self.factory, job, f"Unknown job kind {job.kind!r}", retryable=False)
            return True

        try:
            await ingest_document(
                document_id,
                settings=self.settings,
                factory=self.factory,
                storage=self.storage,
                llm=self.llm,
            )
        except IngestionError as exc:
            await mark_failed(self.factory, document_id, str(exc))
            await queue.fail(self.factory, job, str(exc), retryable=False)
            logger.warning("Job failed permanently", extra={**log_extra, "error": str(exc)})
        except Exception as exc:
            will_retry = await queue.fail(self.factory, job, repr(exc))
            if not will_retry:
                await mark_failed(self.factory, document_id, "Processing failed. Please try again.")
            logger.exception("Job failed", extra={**log_extra, "will_retry": will_retry})
        else:
            await queue.complete(self.factory, job.id)
            logger.info("Job finished", extra=log_extra)
        return True

    async def _run_embed(
        self, job: Job, document_id: uuid.UUID, log_extra: dict[str, object]
    ) -> bool:
        """Stage 2. Failure never fails the document: it stays keyword-searchable."""
        try:
            await embed_document(
                document_id, settings=self.settings, factory=self.factory, job_id=job.id
            )
        except EmbeddingDeferredError as deferred:
            await queue.defer(self.factory, job, deferred.seconds, str(deferred))
            logger.info("Embedding job deferred", extra={**log_extra, "seconds": deferred.seconds})
        except Exception as exc:
            will_retry = await queue.fail(self.factory, job, repr(exc))
            logger.exception("Embedding job failed", extra={**log_extra, "will_retry": will_retry})
        else:
            await queue.complete(self.factory, job.id)
            logger.info("Embedding job finished", extra=log_extra)
        return True

    async def run(self) -> None:
        logger.info("Worker started", extra={"worker_id": self.worker_id, "kinds": self.kinds})
        while not self._stop.is_set():
            try:
                found = await self.run_once()
            except Exception:
                logger.exception("Worker loop error")
                found = False
            if not found:
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(self._stop.wait(), self.settings.worker_poll_seconds)
        logger.info("Worker stopped", extra={"worker_id": self.worker_id})
