"""Standalone ingestion worker: ``python -m docmind.worker``."""

import asyncio
import contextlib
import signal

from docmind.api.app import WORKER_LANES
from docmind.config import get_settings
from docmind.db.session import create_engine, create_session_factory
from docmind.documents.storage import LocalStorage
from docmind.jobs.worker import Worker
from docmind.logging import configure_logging


async def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    engine = create_engine(settings)
    factory = create_session_factory(engine)
    storage = LocalStorage(settings.storage_dir)
    workers = [Worker(settings, factory, storage, lane) for lane in WORKER_LANES]

    def stop_all() -> None:
        for w in workers:
            w.stop()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        # Windows has no add_signal_handler: Ctrl+C raises KeyboardInterrupt instead
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, stop_all)
    try:
        await asyncio.gather(*(w.run() for w in workers))
    finally:
        await engine.dispose()


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(main())
