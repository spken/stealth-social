"""Async SQLite database lifecycle and session creation."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from sqlalchemy import event, inspect
from sqlalchemy.engine import Connection, make_url
from sqlalchemy.exc import ArgumentError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from bot.storage.models import Base

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


_ADDITIVE_COLUMN_MIGRATIONS = (
    (
        "social_actions",
        "external_dispatch_started_at",
        'ALTER TABLE "social_actions" '
        'ADD COLUMN "external_dispatch_started_at" DATETIME',
    ),
    (
        "account_states",
        "execution_owner",
        'ALTER TABLE "account_states" '
        'ADD COLUMN "execution_owner" VARCHAR(255)',
    ),
    (
        "account_states",
        "execution_expires_at",
        'ALTER TABLE "account_states" '
        'ADD COLUMN "execution_expires_at" DATETIME',
    ),
)

_LEGACY_PROCESSING_DISPATCH_BACKFILL = (
    'UPDATE "social_actions" '
    'SET "external_dispatch_started_at" = '
    'COALESCE("updated_at", "created_at", CURRENT_TIMESTAMP) '
    'WHERE "status" = \'processing\' '
    'AND "external_dispatch_started_at" IS NULL'
)

_ADDITIVE_INDEX_MIGRATIONS = (
    (
        "social_actions",
        'CREATE INDEX IF NOT EXISTS "ix_social_actions_claim_recovery" '
        'ON "social_actions" '
        '("status", "claim_expires_at", "external_dispatch_started_at")',
    ),
    (
        "account_states",
        'CREATE INDEX IF NOT EXISTS "ix_account_states_execution_lease" '
        'ON "account_states" ("execution_expires_at", "execution_owner")',
    ),
)


def _migrate_additive_runtime_schema(connection: Connection) -> None:
    """Add runtime lease columns and indexes to an existing SQLite schema."""
    if connection.dialect.name != "sqlite":
        return

    inspector = inspect(connection)
    columns_by_table: dict[str, set[str] | None] = {}
    for table_name, column_name, ddl in _ADDITIVE_COLUMN_MIGRATIONS:
        if table_name not in columns_by_table:
            columns_by_table[table_name] = (
                {
                    str(column["name"])
                    for column in inspector.get_columns(table_name)
                }
                if inspector.has_table(table_name)
                else None
            )
        existing_columns = columns_by_table[table_name]
        if existing_columns is not None and column_name not in existing_columns:
            connection.exec_driver_sql(ddl)
            existing_columns.add(column_name)
            if (
                table_name == "social_actions"
                and column_name == "external_dispatch_started_at"
            ):
                connection.exec_driver_sql(
                    _LEGACY_PROCESSING_DISPATCH_BACKFILL
                )

    for table_name, ddl in _ADDITIVE_INDEX_MIGRATIONS:
        if columns_by_table.get(table_name) is not None:
            connection.exec_driver_sql(ddl)


async def create_schema(engine: AsyncEngine) -> None:
    """Create or additively upgrade the application schema."""
    async with engine.begin() as connection:
        await connection.exec_driver_sql("BEGIN IMMEDIATE")
        await connection.run_sync(_migrate_additive_runtime_schema)
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
