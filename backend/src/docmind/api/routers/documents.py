import asyncio
import json
import uuid
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, File, Query, Request, Response, UploadFile, status
from fastapi.responses import FileResponse
from sqlalchemy import select
from sse_starlette.sse import EventSourceResponse

from docmind.api.deps import CurrentUser, DocumentServiceDep, SessionDep, SettingsDep, StorageDep
from docmind.db.models import Chunk, Document, DocumentStatus
from docmind.documents.schemas import ChunkOut, DocumentList, DocumentOut

router = APIRouter(prefix="/documents", tags=["documents"])

PROGRESS_POLL_SECONDS = 1.0
PROGRESS_MAX_SECONDS = 7200  # long documents embed in the background on rate-limited keys


@router.post(
    "",
    response_model=DocumentOut,
    status_code=status.HTTP_201_CREATED,
    responses={200: {"description": "Identical file already uploaded; existing document returned"}},
)
async def upload_document(
    user: CurrentUser,
    service: DocumentServiceDep,
    settings: SettingsDep,
    response: Response,
    file: Annotated[UploadFile, File(description="PDF or image (PNG, JPEG, WEBP, TIFF)")],
) -> Document:
    document, created = await service.upload(user.id, file, settings.max_upload_bytes)
    if not created:
        response.status_code = status.HTTP_200_OK
    return document


@router.get("", response_model=DocumentList)
async def list_documents(
    user: CurrentUser,
    service: DocumentServiceDep,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> DocumentList:
    items, total = await service.list(user.id, limit, offset)
    return DocumentList(items=[DocumentOut.model_validate(d) for d in items], total=total)


@router.get("/{document_id}", response_model=DocumentOut)
async def get_document(
    document_id: uuid.UUID, user: CurrentUser, service: DocumentServiceDep
) -> Document:
    return await service.get(user.id, document_id)


@router.post("/{document_id}/reprocess", response_model=DocumentOut)
async def reprocess_document(
    document_id: uuid.UUID, user: CurrentUser, service: DocumentServiceDep
) -> Document:
    return await service.reprocess(user.id, document_id)


@router.get("/{document_id}/chunks", response_model=list[ChunkOut])
async def list_chunks(
    document_id: uuid.UUID, user: CurrentUser, service: DocumentServiceDep, session: SessionDep
) -> list[Chunk]:
    await service.get(user.id, document_id)
    rows = await session.scalars(
        select(Chunk)
        .where(Chunk.document_id == document_id, Chunk.owner_id == user.id)
        .order_by(Chunk.ordinal)
    )
    return list(rows)


@router.get("/{document_id}/file")
async def download_file(
    document_id: uuid.UUID, user: CurrentUser, service: DocumentServiceDep, storage: StorageDep
) -> FileResponse:
    """The original upload, for the in-app viewer (PDF.js / image)."""
    document = await service.get(user.id, document_id)
    return FileResponse(
        storage.path(document.storage_key),
        media_type=document.content_type,
        filename=document.filename,
        content_disposition_type="inline",
    )


@router.get("/{document_id}/events")
async def document_events(
    document_id: uuid.UUID, request: Request, user: CurrentUser, service: DocumentServiceDep
) -> EventSourceResponse:
    """SSE stream of ingestion progress until the document is ready or failed."""
    await service.get(user.id, document_id)
    factory = request.app.state.session_factory

    async def events() -> AsyncIterator[dict[str, str]]:
        last = None
        for _ in range(int(PROGRESS_MAX_SECONDS / PROGRESS_POLL_SECONDS)):
            async with factory() as session:
                doc = await session.get(Document, document_id)
            if doc is None:
                yield {"event": "deleted", "data": "{}"}
                return
            snapshot = DocumentOut.model_validate(doc).model_dump(mode="json")
            if snapshot != last:
                yield {"event": "progress", "data": json.dumps(snapshot)}
                last = snapshot
            # Keep streaming after "ready" while background embedding is still catching up
            fully_indexed = doc.embedded_chunks >= doc.chunk_count
            if doc.status == DocumentStatus.FAILED or (
                doc.status == DocumentStatus.READY and fully_indexed
            ):
                return
            await asyncio.sleep(PROGRESS_POLL_SECONDS)

    return EventSourceResponse(events(), ping=15)


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: uuid.UUID, user: CurrentUser, service: DocumentServiceDep
) -> None:
    await service.delete(user.id, document_id)
