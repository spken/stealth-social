"""Immutable models shared by browser collection, storage, and selection."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Any
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

from bot.models import Platform

NonEmptyString = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]


def utc_now() -> datetime:
    return datetime.now(UTC)


class _ExampleModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        revalidate_instances="always",
        str_strip_whitespace=True,
        validate_default=True,
    )


class ExampleType(StrEnum):
    X_POST = "x_post"
    X_REPLY = "x_reply"
    REDDIT_POST = "reddit_post"
    REDDIT_COMMENT = "reddit_comment"


class CollectionRunStatus(StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"


class PromptInjectionSeverity(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class PromptInjectionCategory(StrEnum):
    AUTHORITY_OVERRIDE = "authority_override"
    ROLE_IMPERSONATION = "role_impersonation"
    DELIMITER_INJECTION = "delimiter_injection"
    PROMPT_EXTRACTION = "prompt_extraction"
    TOOL_COMMAND_REQUEST = "tool_command_request"
    TASK_REDIRECTION = "task_redirection"


class PromptInjectionFinding(_ExampleModel):
    severity: PromptInjectionSeverity
    category: PromptInjectionCategory
    evidence: NonEmptyString
    start_offset: int = Field(ge=0)
    end_offset: int = Field(ge=0)


class PromptSafetyResult(_ExampleModel):
    findings: tuple[PromptInjectionFinding, ...] = ()

    @property
    def highest_severity(self) -> PromptInjectionSeverity | None:
        order = {
            PromptInjectionSeverity.LOW: 1,
            PromptInjectionSeverity.MEDIUM: 2,
            PromptInjectionSeverity.HIGH: 3,
        }
        return max(self.findings, key=lambda item: order[item.severity]).severity if self.findings else None

    @property
    def is_high_risk(self) -> bool:
        return self.highest_severity is PromptInjectionSeverity.HIGH


class ExampleCollectionRequest(_ExampleModel):
    id: UUID = Field(default_factory=uuid4)
    platform: Platform
    account_name: NonEmptyString
    sources: tuple[NonEmptyString, ...] = ()
    accounts: tuple[NonEmptyString, ...] = ()
    queries: tuple[NonEmptyString, ...] = ()
    subreddits: tuple[NonEmptyString, ...] = ()
    post_urls: tuple[NonEmptyString, ...] = ()
    sort: str = "top"
    time_filter: str = "month"
    maximum_items_per_source: int = Field(default=25, ge=1, le=500)
    maximum_comments_per_post: int = Field(default=20, ge=0, le=200)
    minimum_score: int = 0
    include_own_content: bool = True
    since: datetime | None = None
    until: datetime | None = None

    @field_validator("since", "until")
    @classmethod
    def normalize_timestamp(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)


class ExampleCollectionRun(_ExampleModel):
    id: UUID = Field(default_factory=uuid4)
    platform: Platform
    started_at: datetime = Field(default_factory=utc_now)
    finished_at: datetime | None = None
    status: CollectionRunStatus = CollectionRunStatus.RUNNING
    request_snapshot: dict[str, Any] = Field(default_factory=dict)
    collected_count: int = Field(default=0, ge=0)
    rejected_count: int = Field(default=0, ge=0)
    duplicate_count: int = Field(default=0, ge=0)
    disabled_count: int = Field(default=0, ge=0)
    error_type: str | None = None
    error_message: str | None = None
    retry_after_seconds: float | None = Field(default=None, ge=0)

    @field_validator("started_at", "finished_at")
    @classmethod
    def normalize_timestamp(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)


class CollectionRunResult(_ExampleModel):
    status: CollectionRunStatus = CollectionRunStatus.COMPLETED
    finished_at: datetime = Field(default_factory=utc_now)
    collected_count: int = Field(default=0, ge=0)
    rejected_count: int = Field(default=0, ge=0)
    duplicate_count: int = Field(default=0, ge=0)
    disabled_count: int = Field(default=0, ge=0)
    error_type: str | None = None
    error_message: str | None = None
    retry_after_seconds: float | None = Field(default=None, ge=0)

    @field_validator("finished_at")
    @classmethod
    def normalize_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)


class ExampleUpsertReport(_ExampleModel):
    inserted_count: int = Field(default=0, ge=0)
    refreshed_count: int = Field(default=0, ge=0)
    duplicate_count: int = Field(default=0, ge=0)
    rejected_count: int = Field(default=0, ge=0)


class CollectedExample(_ExampleModel):
    """Transient browser extraction; raw usernames must never be placed here."""

    platform: Platform
    content_type: ExampleType
    external_id: NonEmptyString | None = None
    source_url: NonEmptyString
    author_identifier: NonEmptyString | None = None
    title: str | None = None
    body: str = ""
    parent_text: str | None = None
    subreddit: str | None = None
    published_at: datetime | None = None
    collected_at: datetime = Field(default_factory=utc_now)
    expires_at: datetime | None = None
    content_hash: str | None = None
    engagement_score: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)
    topic_tags: tuple[NonEmptyString, ...] = ()
    is_own_content: bool = False
    generated: bool = False
    injection_findings: tuple[dict[str, Any], ...] = ()

    @field_validator("published_at", "collected_at", "expires_at")
    @classmethod
    def normalize_timestamp(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return value
        if value.tzinfo is None or value.utcoffset() is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)


class ContentExample(_ExampleModel):
    id: UUID = Field(default_factory=uuid4)
    collection_run_id: UUID | None = None
    platform: Platform
    content_type: ExampleType
    external_id: NonEmptyString | None = None
    source_url: NonEmptyString
    author_identifier: NonEmptyString | None = None
    title: str | None = None
    body: str = ""
    parent_text: str | None = None
    subreddit: str | None = None
    published_at: datetime | None = None
    collected_at: datetime = Field(default_factory=utc_now)
    expires_at: datetime | None = None
    content_hash: NonEmptyString
    engagement_score: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)
    topic_tags: tuple[NonEmptyString, ...] = ()
    is_own_content: bool = False
    generated: bool = False
    is_active: bool = True
    is_quarantined: bool = False
    injection_findings: tuple[dict[str, Any], ...] = ()

    @field_validator("published_at", "collected_at", "expires_at")
    @classmethod
    def normalize_timestamp(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return value
        if value.tzinfo is None or value.utcoffset() is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)


class ExampleListFilters(_ExampleModel):
    platform: Platform | None = None
    content_type: ExampleType | None = None
    subreddit: str | None = None
    active_only: bool = True
    limit: int = Field(default=100, ge=1, le=1000)
    offset: int = Field(default=0, ge=0)


class ExampleSelectionFilters(_ExampleModel):
    platform: Platform
    content_type: ExampleType | None = None
    subreddit: str | None = None
    allowed_subreddits: tuple[NonEmptyString, ...] = ()
    compatible_types: tuple[ExampleType, ...] = ()
    topic: str | None = None
    keywords: tuple[NonEmptyString, ...] = ()
    target_text: str | None = None
    audience: str | None = None
    tone: str | None = None
    preferred_example_ids: tuple[UUID, ...] = ()
    maximum_context_examples: int = Field(default=8, ge=0, le=50)
    maximum_example_characters: int = Field(default=12000, ge=0, le=100000)
    allow_generated: bool = False
    useful_window_days: int = Field(default=90, ge=1, le=3650)


class SelectedExample(_ExampleModel):
    example_id: UUID
    score: float = Field(ge=0, le=10)
    component_scores: dict[str, float] = Field(default_factory=dict)
    selection_reason: NonEmptyString
    example: ContentExample = Field(exclude=True)


class TargetContextRequest(_ExampleModel):
    platform: Platform
    account_name: NonEmptyString
    target_url: NonEmptyString | None = None
    target_id: NonEmptyString | None = None
    target_kind: str
    subreddit: str | None = None


class TargetContext(_ExampleModel):
    platform: Platform
    canonical_url: NonEmptyString
    external_id: NonEmptyString
    title: str | None = None
    body: str = ""
    parent_text: str | None = None
    discussion_comments: tuple[str, ...] = ()
    subreddit: str | None = None
    parent_post_id: str | None = None
    parent_comment_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    retrieved_at: datetime = Field(default_factory=utc_now)
    injection_findings: tuple[dict[str, Any], ...] = ()

    @field_validator("retrieved_at")
    @classmethod
    def normalize_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)


class ExampleCollectionError(RuntimeError):
    """Safe public-collection failure with bounded diagnostic metadata."""

    def __init__(self, message: str, *, retry_after_seconds: float | None = None) -> None:
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


class ExampleTargetUnavailableError(ExampleCollectionError):
    """The requested public target cannot be resolved."""


class ExampleRateLimitedError(ExampleCollectionError):
    """A public source reported a rate limit."""


class ExampleChallengeError(ExampleCollectionError):
    """A challenge or login wall prevents safe public collection."""


__all__ = [
    "CollectedExample",
    "CollectionRunResult",
    "CollectionRunStatus",
    "ContentExample",
    "ExampleCollectionError",
    "ExampleCollectionRequest",
    "ExampleCollectionRun",
    "ExampleListFilters",
    "ExampleRateLimitedError",
    "ExampleSelectionFilters",
    "ExampleTargetUnavailableError",
    "ExampleChallengeError",
    "ExampleType",
    "ExampleUpsertReport",
    "PromptInjectionCategory",
    "PromptInjectionFinding",
    "PromptInjectionSeverity",
    "PromptSafetyResult",
    "SelectedExample",
    "TargetContext",
    "TargetContextRequest",
]
