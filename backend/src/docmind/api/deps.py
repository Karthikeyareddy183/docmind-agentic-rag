"""FastAPI dependencies: settings, DB session, storage, and the authenticated user."""

import uuid
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, Request
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.ext.asyncio import AsyncSession

from docmind.api.errors import UnauthorizedError
from docmind.auth.security import decode_access_token
from docmind.config import Settings
from docmind.db.models import User
from docmind.db.session import session_scope
from docmind.documents.service import DocumentService
from docmind.documents.storage import Storage

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login", auto_error=False)


def get_settings(request: Request) -> Settings:
    settings: Settings = request.app.state.settings
    return settings


async def get_session(request: Request) -> AsyncIterator[AsyncSession]:
    async for session in session_scope(request.app.state.session_factory):
        yield session


def get_storage(request: Request) -> Storage:
    storage: Storage = request.app.state.storage
    return storage


SettingsDep = Annotated[Settings, Depends(get_settings)]
SessionDep = Annotated[AsyncSession, Depends(get_session)]
StorageDep = Annotated[Storage, Depends(get_storage)]


async def get_current_user(
    session: SessionDep,
    settings: SettingsDep,
    token: Annotated[str | None, Depends(oauth2_scheme)],
) -> User:
    user_id: uuid.UUID | None = decode_access_token(token, settings) if token else None
    user = await session.get(User, user_id) if user_id else None
    if user is None:
        raise UnauthorizedError("Invalid or expired credentials")
    return user


CurrentUser = Annotated[User, Depends(get_current_user)]


def get_document_service(session: SessionDep, storage: StorageDep) -> DocumentService:
    return DocumentService(session, storage)


DocumentServiceDep = Annotated[DocumentService, Depends(get_document_service)]
