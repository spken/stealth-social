"""Stable machine-readable outcomes for autopost runs."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, field_validator

from bot.models import ActionStatus, Platform


class AutopostOutcome(StrEnum):
    PUBLISHED = "published"
    SKIPPED_RECENT_SUCCESS = "skipped_recent_success"
    TEMPORARY_FAILURE = "temporary_failure"
    ATTENTION_REQUIRED = "attention_required"
    CONFIGURATION_ERROR = "configuration_error"


class AutopostResult(BaseModel):
    """The bounded JSON result emitted by one autopost invocation."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    campaign_id: str
    outcome: AutopostOutcome
    platform: Platform | None = None
    account: str | None = None
    topic: str | None = None
    request_id: UUID | None = None
    candidate_id: UUID | None = None
    action_id: UUID | None = None
    action_status: ActionStatus | None = None
    published_url: str | None = None
    retry_at: datetime | None = None
    attention_reason: str | None = None

    @field_validator("retry_at")
    @classmethod
    def normalize_retry_at(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)

    @property
    def exit_code(self) -> int:
        return {
            AutopostOutcome.PUBLISHED: 0,
            AutopostOutcome.SKIPPED_RECENT_SUCCESS: 0,
            AutopostOutcome.CONFIGURATION_ERROR: 2,
            AutopostOutcome.ATTENTION_REQUIRED: 3,
            AutopostOutcome.TEMPORARY_FAILURE: 75,
        }[self.outcome]


__all__ = ["AutopostOutcome", "AutopostResult"]
