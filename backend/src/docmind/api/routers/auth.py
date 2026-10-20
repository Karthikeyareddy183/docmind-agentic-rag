from typing import Annotated

from fastapi import APIRouter, Depends, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from docmind.api.deps import CurrentUser, SessionDep, SettingsDep
from docmind.api.errors import ConflictError, UnauthorizedError
from docmind.auth.schemas import RegisterRequest, TokenResponse, UserOut
from docmind.auth.security import create_access_token, hash_password, verify_password_or_dummy
from docmind.db.models import User

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=UserOut, status_code=status.HTTP_201_CREATED)
async def register(body: RegisterRequest, session: SessionDep) -> User:
    user = User(email=body.email.lower(), password_hash=hash_password(body.password))
    session.add(user)
    try:
        await session.commit()
    except IntegrityError:
        await session.rollback()
        raise ConflictError("An account with this email already exists") from None
    await session.refresh(user)
    return user


@router.post("/login", response_model=TokenResponse)
async def login(
    form: Annotated[OAuth2PasswordRequestForm, Depends()],
    session: SessionDep,
    settings: SettingsDep,
) -> TokenResponse:
    """OAuth2 password flow: ``username`` carries the email (works with Swagger's Authorize)."""
    user = await session.scalar(select(User).where(User.email == form.username.lower()))
    valid = verify_password_or_dummy(form.password, user.password_hash if user else None)
    if user is None or not valid:
        raise UnauthorizedError("Incorrect email or password")
    return TokenResponse(
        access_token=create_access_token(user.id, settings),
        expires_in=settings.access_token_ttl_minutes * 60,
    )


@router.get("/me", response_model=UserOut)
async def me(user: CurrentUser) -> User:
    return user
