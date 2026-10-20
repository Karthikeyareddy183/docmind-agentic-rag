"""FastAPI application factory."""

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from docmind import __version__
from docmind.api.errors import register_error_handlers
from docmind.api.middleware import RequestContextMiddleware
from docmind.api.routers import auth, chat, documents, health
from docmind.config import Settings, get_settings
from docmind.db.models import EMBEDDING_DIMENSION
from docmind.db.session import create_engine, create_session_factory
from docmind.documents.storage import LocalStorage
from docmind.jobs.queue import EMBED_DOCUMENT, INGEST_DOCUMENT
from docmind.jobs.worker import Worker
from docmind.llm.gateway import LLMGateway
from docmind.logging import configure_logging

API_PREFIX = "/api/v1"
# One worker per lane: fast ingestion is never queued behind slow background embedding
WORKER_LANES: tuple[tuple[str, ...], ...] = ((INGEST_DOCUMENT,), (EMBED_DOCUMENT,))
logger = logging.getLogger(__name__)


def _warm_models(settings: Settings) -> None:
    """Load embedding + reranker weights off the request path (first load takes a while on CPU)."""
    from docmind.retrieval.models import get_embedder, get_reranker

    try:
        get_embedder(
            settings.embedding_model, settings.embedding_dimension, settings.embedding_batch_size
        )
        if settings.rerank_enabled:
            get_reranker(settings.reranker_model)
        logger.info("Models ready")
    except Exception:
        logger.exception("Model warm-up failed")


def create_app(
    settings: Settings | None = None, *, start_background: bool | None = None
) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)
    background = settings.environment != "test" if start_background is None else start_background

    if settings.is_production and "change-me" in settings.jwt_secret.get_secret_value():
        raise RuntimeError("DOCMIND_JWT_SECRET must be set in production")
    if settings.embedding_dimension != EMBEDDING_DIMENSION:
        raise RuntimeError(
            f"DOCMIND_EMBEDDING_DIMENSION={settings.embedding_dimension} but the index column is "
            f"vector({EMBEDDING_DIMENSION}); change both together (migration + re-embed)"
        )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        engine = create_engine(settings)
        app.state.settings = settings
        app.state.engine = engine
        app.state.session_factory = create_session_factory(engine)
        app.state.storage = LocalStorage(settings.storage_dir)
        app.state.llm = LLMGateway(settings)

        tasks: list[asyncio.Task[None]] = []
        workers: list[Worker] = []
        if background:
            tasks.append(asyncio.create_task(asyncio.to_thread(_warm_models, settings)))
            if settings.run_worker_in_api:
                for lane in WORKER_LANES:
                    worker = Worker(settings, app.state.session_factory, app.state.storage, lane)
                    workers.append(worker)
                    tasks.append(asyncio.create_task(worker.run()))
        yield
        for worker in workers:
            worker.stop()
        for task in tasks:
            task.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await task
        await engine.dispose()

    app = FastAPI(
        title="DocMind API",
        version=__version__,
        description="Agentic document analysis: RAG over PDFs and scans with verifiable citations.",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID"],
    )
    app.add_middleware(RequestContextMiddleware)
    register_error_handlers(app)

    for router in (health.router, auth.router, documents.router, chat.router):
        app.include_router(router, prefix=API_PREFIX)

    return app
