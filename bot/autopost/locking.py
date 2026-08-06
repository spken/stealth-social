"""Cancellation-safe application-wide autopost locking."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from filelock import FileLock, Timeout as FileLockTimeout
from sqlalchemy.engine import make_url
from sqlalchemy.exc import ArgumentError


class AutopostLockBusyError(TimeoutError):
    """Raised when another autopost invocation holds the global lock."""


def autopost_lock_path(
    database_url: str,
    *,
    explicit_path: Path | None = None,
) -> Path:
    """Derive the lock beside a file-backed SQLite database."""

    try:
        url = make_url(database_url)
    except ArgumentError as error:
        raise ValueError("database_url must be a valid SQLAlchemy URL") from error
    if not url.drivername.split("+", 1)[0] == "sqlite":
        raise ValueError("autopost lock requires a SQLite database")
    if explicit_path is not None:
        return Path(explicit_path).expanduser().resolve(strict=False)
    database = url.database
    if not database or database.casefold() == ":memory:":
        raise ValueError(
            "autopost lock requires a file-backed SQLite database unless "
            "explicit_path is provided"
        )
    database_path = Path(database).expanduser()
    if not database_path.is_absolute():
        database_path = Path.cwd() / database_path
    database_path = database_path.resolve(strict=False)
    return database_path.with_name(database_path.name + ".autopost.lock")


@asynccontextmanager
async def hold_autopost_lock(
    database_url: str,
    *,
    explicit_path: Path | None = None,
    timeout_seconds: float = 30.0,
    poll_seconds: float = 0.1,
) -> AsyncIterator[Path]:
    """Hold the global lock with only cancellable nonblocking attempts."""

    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    if poll_seconds <= 0:
        raise ValueError("poll_seconds must be positive")

    path = autopost_lock_path(database_url, explicit_path=explicit_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(path))
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_seconds
    acquired = False
    try:
        while not acquired:
            try:
                lock.acquire(timeout=0)
                acquired = True
            except FileLockTimeout as error:
                if loop.time() >= deadline:
                    raise AutopostLockBusyError(
                        "another autopost run holds the global lock"
                    ) from error
                await asyncio.sleep(poll_seconds)
        yield path
    finally:
        if acquired:
            lock.release()


__all__ = [
    "AutopostLockBusyError",
    "autopost_lock_path",
    "hold_autopost_lock",
]
