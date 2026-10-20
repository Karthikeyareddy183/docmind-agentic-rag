"""Document use cases. All queries are scoped to the requesting user's ``owner_id``."""

import uuid
from pathlib import PurePath

import filetype
from fastapi import UploadFile
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from docmind.api.errors import NotFoundError, UnsupportedMediaTypeError
from docmind.db.models import Document, DocumentStatus
from docmind.documents.storage import Storage
from docmind.jobs.queue import INGEST_DOCUMENT, enqueue

# Detected from file bytes, never from the client-supplied Content-Type
ALLOWED_MIME_TYPES = {
    "application/pdf",
    "image/png",
    "image/jpeg",
    "image/webp",
    "image/tiff",
}

MAX_FILENAME_LENGTH = 255


def sanitize_filename(name: str | None) -> str:
    """Keep only the base name (no directories, no control chars) for display purposes."""
    base = PurePath((name or "").replace("\\", "/")).name
    cleaned = "".join(ch for ch in base if ch.isprintable()).strip()
    if cleaned in {"", ".", ".."}:
        cleaned = "upload"
    return cleaned[:MAX_FILENAME_LENGTH]


class DocumentService:
    def __init__(self, session: AsyncSession, storage: Storage) -> None:
        self.session = session
        self.storage = storage

    async def upload(
        self, owner_id: uuid.UUID, upload: UploadFile, max_bytes: int
    ) -> tuple[Document, bool]:
        """Store an upload. Returns ``(document, created)``; duplicates return the existing row."""
        stored = await self.storage.save(upload, max_bytes)
        try:
            kind = filetype.guess(stored.head)
            mime = kind.mime if kind else None
            if mime not in ALLOWED_MIME_TYPES:
                raise UnsupportedMediaTypeError(
                    "Unsupported file type. Allowed: PDF, PNG, JPEG, WEBP, TIFF"
                )

            existing = await self._find_by_hash(owner_id, stored.sha256)
            if existing is not None:
                await self.storage.delete(stored.key)
                return existing, False

            document = Document(
                id=uuid.uuid4(),
                owner_id=owner_id,
                filename=sanitize_filename(upload.filename),
                content_type=mime,
                size_bytes=stored.size_bytes,
                sha256=stored.sha256,
                storage_key=stored.key,
            )
            self.session.add(document)
            # Same transaction as the row: an upload can never exist without its ingestion job
            enqueue(self.session, INGEST_DOCUMENT, {"document_id": str(document.id)})
            try:
                await self.session.commit()
            except IntegrityError:
                # A concurrent upload of the same file won the unique-index race
                await self.session.rollback()
                existing = await self._find_by_hash(owner_id, stored.sha256)
                if existing is None:
                    raise
                await self.storage.delete(stored.key)
                return existing, False
        except BaseException:
            await self.storage.delete(stored.key)
            raise
        await self.session.refresh(document)
        return document, True

    async def _find_by_hash(self, owner_id: uuid.UUID, sha256: str) -> Document | None:
        return await self.session.scalar(
            select(Document).where(Document.owner_id == owner_id, Document.sha256 == sha256)
        )

    async def list(
        self, owner_id: uuid.UUID, limit: int, offset: int
    ) -> tuple[list[Document], int]:
        scoped = select(Document).where(Document.owner_id == owner_id)
        total = await self.session.scalar(select(func.count()).select_from(scoped.subquery()))
        rows = await self.session.scalars(
            scoped.order_by(Document.created_at.desc(), Document.id).limit(limit).offset(offset)
        )
        return list(rows), total or 0

    async def get(self, owner_id: uuid.UUID, document_id: uuid.UUID) -> Document:
        document = await self.session.scalar(
            select(Document).where(Document.id == document_id, Document.owner_id == owner_id)
        )
        if document is None:
            # Same response for "missing" and "someone else's" so ids can't be probed
            raise NotFoundError("Document not found")
        return document

    async def reprocess(self, owner_id: uuid.UUID, document_id: uuid.UUID) -> Document:
        document = await self.get(owner_id, document_id)
        document.status = DocumentStatus.UPLOADED
        document.progress = 0.0
        document.embedded_chunks = 0
        document.stage = None
        document.error = None
        enqueue(self.session, INGEST_DOCUMENT, {"document_id": str(document.id)})
        await self.session.commit()
        await self.session.refresh(document)
        return document

    async def delete(self, owner_id: uuid.UUID, document_id: uuid.UUID) -> None:
        document = await self.get(owner_id, document_id)
        await self.session.delete(document)
        await self.session.commit()
        await self.storage.delete(document.storage_key)
