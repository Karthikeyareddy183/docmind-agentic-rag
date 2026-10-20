"""File storage behind a small interface so S3 can replace local disk in production."""

import contextlib
import hashlib
import uuid
from pathlib import Path
from typing import Protocol

import aiofiles
import aiofiles.os
from fastapi import UploadFile

from docmind.api.errors import PayloadTooLargeError

CHUNK_SIZE = 1024 * 1024


class StoredFile:
    def __init__(self, key: str, size_bytes: int, sha256: str, head: bytes) -> None:
        self.key = key
        self.size_bytes = size_bytes
        self.sha256 = sha256
        self.head = head  # first bytes, for content sniffing


class Storage(Protocol):
    async def save(self, upload: UploadFile, max_bytes: int) -> StoredFile: ...

    async def delete(self, key: str) -> None: ...

    def path(self, key: str) -> Path: ...


class LocalStorage:
    """Stores files under ``root`` with server-generated keys; client filenames never touch disk."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, key: str) -> Path:
        return self.root / key

    async def save(self, upload: UploadFile, max_bytes: int) -> StoredFile:
        """Stream the upload to disk, hashing as we go; abort once ``max_bytes`` is exceeded."""
        key = uuid.uuid4().hex
        target = self.path(key)
        digest = hashlib.sha256()
        size = 0
        head = b""
        try:
            async with aiofiles.open(target, "wb") as out:
                while chunk := await upload.read(CHUNK_SIZE):
                    size += len(chunk)
                    if size > max_bytes:
                        raise PayloadTooLargeError(
                            f"File exceeds the {max_bytes // (1024 * 1024)} MB limit"
                        )
                    if len(head) < 2048:
                        head += chunk[: 2048 - len(head)]
                    digest.update(chunk)
                    await out.write(chunk)
        except BaseException:
            await self.delete(key)
            raise
        return StoredFile(key=key, size_bytes=size, sha256=digest.hexdigest(), head=head)

    async def delete(self, key: str) -> None:
        with contextlib.suppress(FileNotFoundError):
            await aiofiles.os.remove(self.path(key))
