import uuid
from datetime import UTC, datetime, timedelta

import jwt
from pydantic import SecretStr

from docmind.auth.security import (
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
    verify_password_or_dummy,
)
from docmind.config import Settings


def test_password_hash_roundtrip() -> None:
    hashed = hash_password("correct horse")
    assert hashed != "correct horse"
    assert verify_password("correct horse", hashed)
    assert not verify_password("wrong", hashed)


def test_unknown_user_never_verifies() -> None:
    assert not verify_password_or_dummy("anything", None)


def test_access_token_roundtrip(settings: Settings) -> None:
    user_id = uuid.uuid4()
    assert decode_access_token(create_access_token(user_id, settings), settings) == user_id


def test_tampered_token_rejected(settings: Settings) -> None:
    token = create_access_token(uuid.uuid4(), settings)
    other = settings.model_copy(
        update={"jwt_secret": SecretStr("a-different-secret-32-bytes-long!!")}
    )
    assert decode_access_token(token, other) is None
    assert decode_access_token(token[:-2] + "xx", settings) is None


def test_expired_token_rejected(settings: Settings) -> None:
    past = datetime.now(UTC) - timedelta(hours=2)
    token = jwt.encode(
        {
            "sub": str(uuid.uuid4()),
            "iat": past,
            "exp": past + timedelta(minutes=1),
            "type": "access",
        },
        settings.jwt_secret.get_secret_value(),
        algorithm=settings.jwt_algorithm,
    )
    assert decode_access_token(token, settings) is None


def test_wrong_token_type_rejected(settings: Settings) -> None:
    now = datetime.now(UTC)
    token = jwt.encode(
        {
            "sub": str(uuid.uuid4()),
            "iat": now,
            "exp": now + timedelta(minutes=5),
            "type": "refresh",
        },
        settings.jwt_secret.get_secret_value(),
        algorithm=settings.jwt_algorithm,
    )
    assert decode_access_token(token, settings) is None


def test_alg_none_rejected(settings: Settings) -> None:
    now = datetime.now(UTC)
    token = jwt.encode(
        {"sub": str(uuid.uuid4()), "iat": now, "exp": now + timedelta(minutes=5), "type": "access"},
        key=None,
        algorithm="none",
    )
    assert decode_access_token(token, settings) is None
