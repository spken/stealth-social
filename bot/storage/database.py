"""Async SQLite database lifecycle and session creation."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from sqlalchemy import event
from sqlalchemy.engine import make_url
from sqlalchemy.exc import ArgumentError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from bot.storage.models import Base
# Import content mappers before create_all; the schema is intentionally disposable.
from bot.storage import content_models as _content_models  # noqa: F401

AsyncSessionFactory = async_sessionmaker[AsyncSession]


def normalize_database_url(database_url: str) -> str:
    """Return an async SQLite SQLAlchemy URL.

    Configuration accepts the familiar synchronous ``sqlite:///`` form, while
    all application access uses the aiosqlite driver.
    """
    try:
        url = make_url(database_url)
    except ArgumentError as error:
        raise ValueError("database_url must be a valid SQLAlchemy URL") from error

    if url.drivername == "sqlite":
        url = url.set(drivername="sqlite+aiosqlite")
    elif url.drivername != "sqlite+aiosqlite":
        raise ValueError("database_url must use sqlite or sqlite+aiosqlite")
    return url.render_as_string(hide_password=False)


def create_async_sqlite_engine(
    database_url: str,
    *,
    echo: bool = False,
) -> AsyncEngine:
    """Create an async engine configured for SQLite safety and contention."""
    engine = create_async_engine(
        normalize_database_url(database_url),
        echo=echo,
        connect_args={"timeout": 30},
        pool_pre_ping=True,
    )

    @event.listens_for(engine.sync_engine, "connect")
    def configure_sqlite_connection(
        dbapi_connection: Any,
        connection_record: Any,
    ) -> None:
        del connection_record
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA busy_timeout=30000")
        finally:
            cursor.close()

    return engine


async def create_schema(engine: AsyncEngine) -> None:
    """Create the complete disposable application schema."""
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)


class Database:
    """Own the async engine and non-expiring session factory."""

    def __init__(self, database_url: str, *, echo: bool = False) -> None:
        self.url = normalize_database_url(database_url)
        self.engine = create_async_sqlite_engine(self.url, echo=echo)
        self.session_factory: AsyncSessionFactory = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False,
            autoflush=False,
        )

    async def initialize(self) -> None:
        """Idempotently create the database schema."""
        await create_schema(self.engine)

    def session(self) -> AsyncSession:
        """Create a session suitable for ``async with`` use."""
        return self.session_factory()

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[AsyncSession]:
        """Yield a session inside a committing transaction."""
        async with self.session_factory() as session:
            async with session.begin():
                yield session

    async def close(self) -> None:
        """Dispose pooled connections."""
        await self.engine.dispose()

    async def __aenter__(self) -> Database:
        try:
            await self.initialize()
        except BaseException:
            await self.close()
            raise
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()
