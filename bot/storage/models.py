"""SQLAlchemy persistence models for social actions and account safety state."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
)
from sqlalchemy.engine.interfaces import Dialect
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.types import TypeDecorator


def utc_now() -> datetime:
    """Return an aware UTC timestamp for ORM defaults."""
    return datetime.now(UTC)


class UTCDateTime(TypeDecorator[datetime]):
    """Persist aware datetimes and restore SQLite's naive values as UTC."""

    impl = DateTime
    cache_ok = True

    def load_dialect_impl(self, dialect: Dialect) -> Any:
        return dialect.type_descriptor(DateTime(timezone=True))

    def process_bind_param(
        self,
        value: datetime | None,
        dialect: Dialect,
    ) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("database timestamps must be timezone-aware")
        return value.astimezone(UTC)

    def process_result_value(
        self,
        value: datetime | None,
        dialect: Dialect,
    ) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)


class Base(DeclarativeBase):
    """Base for all application-owned tables."""


class SocialActionRecord(Base):
    """Persistent representation of ``bot.models.SocialAction``."""

    __tablename__ = "social_actions"
    __table_args__ = (
        CheckConstraint("attempts >= 0", name="ck_social_actions_attempts_nonnegative"),
        CheckConstraint("max_attempts > 0", name="ck_social_actions_max_attempts_positive"),
        Index(
            "ix_social_actions_due",
            "status",
            "scheduled_at",
            "retry_available_at",
            "claim_expires_at",
        ),
        Index(
            "ix_social_actions_duplicate",
            "platform",
            "account_name",
            "target_scope",
            "fingerprint",
            "status",
            "published_at",
        ),
        Index(
            "ix_social_actions_rate_usage",
            "platform",
            "account_name",
            "status",
            "published_at",
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    action_type: Mapped[str] = mapped_column(String(32), nullable=False)
    platform: Mapped[str] = mapped_column(String(16), nullable=False)
    account_name: Mapped[str] = mapped_column(String(255), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    title: Mapped[str | None] = mapped_column(Text)
    subreddit: Mapped[str | None] = mapped_column(String(255))
    target_url: Mapped[str | None] = mapped_column(Text)
    parent_post_id: Mapped[str | None] = mapped_column(String(255))
    parent_comment_id: Mapped[str | None] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime(), nullable=False, default=utc_now
    )
    scheduled_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="draft")
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=3)
    last_error: Mapped[str | None] = mapped_column(Text)
    external_content_id: Mapped[str | None] = mapped_column(String(512))
    external_content_url: Mapped[str | None] = mapped_column(Text)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSON, nullable=False, default=dict
    )
    fingerprint: Mapped[str] = mapped_column(String(128), nullable=False)

    target_scope: Mapped[str] = mapped_column(String(1024), nullable=False)
    claim_owner: Mapped[str | None] = mapped_column(String(255))
    claim_expires_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    retry_available_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    published_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    error_page_url: Mapped[str | None] = mapped_column(Text)
    error_screenshot_path: Mapped[str | None] = mapped_column(Text)
    updated_at: Mapped[datetime] = mapped_column(
        UTCDateTime(), nullable=False, default=utc_now, onupdate=utc_now
    )

    events: Mapped[list[ActionEventRecord]] = relationship(
        back_populates="action",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class ActionEventRecord(Base):
    """Append-only state transition and execution diagnostic record."""

    __tablename__ = "action_events"
    __table_args__ = (
        Index("ix_action_events_action_created", "action_id", "created_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    action_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("social_actions.id", ondelete="CASCADE"),
        nullable=False,
    )
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    from_status: Mapped[str | None] = mapped_column(String(32))
    to_status: Mapped[str | None] = mapped_column(String(32))
    message: Mapped[str | None] = mapped_column(Text)
    page_url: Mapped[str | None] = mapped_column(Text)
    screenshot_path: Mapped[str | None] = mapped_column(Text)
    context_json: Mapped[dict[str, Any]] = mapped_column(
        "context", JSON, nullable=False, default=dict
    )
    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime(), nullable=False, default=utc_now
    )

    action: Mapped[SocialActionRecord] = relationship(back_populates="events")


class AccountStateRecord(Base):
    """Per-account pause and failure-threshold state."""

    __tablename__ = "account_states"
    __table_args__ = (
        CheckConstraint(
            "consecutive_failures >= 0",
            name="ck_account_states_consecutive_failures_nonnegative",
        ),
        CheckConstraint(
            "total_failures >= 0",
            name="ck_account_states_total_failures_nonnegative",
        ),
        CheckConstraint(
            "failure_threshold IS NULL OR failure_threshold > 0",
            name="ck_account_states_failure_threshold_positive",
        ),
        Index("ix_account_states_paused", "paused", "paused_until"),
    )

    platform: Mapped[str] = mapped_column(String(16), primary_key=True)
    account_name: Mapped[str] = mapped_column(String(255), primary_key=True)
    paused: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    paused_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    paused_until: Mapped[datetime | None] = mapped_column(UTCDateTime())
    pause_reason: Mapped[str | None] = mapped_column(Text)
    consecutive_failures: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0
    )
    total_failures: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    failure_threshold: Mapped[int | None] = mapped_column(Integer)
    last_failure_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    last_success_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    last_error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime(), nullable=False, default=utc_now
    )
    updated_at: Mapped[datetime] = mapped_column(
        UTCDateTime(), nullable=False, default=utc_now, onupdate=utc_now
    )
