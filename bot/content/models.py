"""Immutable content-generation domain models."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Any, Self
from uuid import UUID, uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

from bot.examples.models import ExampleSelectionFilters, SelectedExample, TargetContext
from bot.models import Platform

NonEmptyString = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]


def utc_now() -> datetime:
    return datetime.now(UTC)


class _ContentModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        revalidate_instances="always",
        str_strip_whitespace=True,
        validate_default=True,
    )


class GenerationType(StrEnum):
    X_POST = "x_post"
    X_REPLY = "x_reply"
    REDDIT_POST = "reddit_post"
    REDDIT_COMMENT = "reddit_comment"
    REDDIT_REPLY = "reddit_reply"


class RankingMode(StrEnum):
    HEURISTIC = "heuristic"
    OLLAMA = "ollama"


class ContentPurpose(StrEnum):
    EDUCATIONAL = "educational"
    PRODUCT_UPDATE = "product_update"
    BUILDER_UPDATE = "builder_update"
    PROMOTIONAL = "promotional"
    ORGANIC_DISCUSSION = "organic_discussion"
    CUSTOMER_SUPPORT = "customer_support"


class GenerationStatus(StrEnum):
    SCHEDULED = "scheduled"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class CandidateApprovalStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    SUPERSEDED = "superseded"


class ValidationSeverity(StrEnum):
    ERROR = "error"
    WARNING = "warning"


class AccountContext(_ContentModel):
    account_name: NonEmptyString
    platform: Platform
    identity: str | None = None
    products: tuple[NonEmptyString, ...] = ()
    verified_facts: tuple[NonEmptyString, ...] = ()
    forbidden_claims: tuple[NonEmptyString, ...] = ()
    required_disclosures: tuple[NonEmptyString, ...] = ()


class FactRequirement(_ContentModel):
    statement: NonEmptyString
    required_terms: tuple[NonEmptyString, ...] = ()


class ContentRequest(_ContentModel):
    id: UUID = Field(default_factory=uuid4)
    generation_type: GenerationType
    platform: Platform
    account_name: NonEmptyString
    content_purpose: ContentPurpose | None = None
    topic: NonEmptyString | None = None
    goal: NonEmptyString | None = None
    product_context: NonEmptyString | None = None
    project_context: NonEmptyString | None = None
    target_audience: NonEmptyString | None = None
    tone: NonEmptyString | None = None
    desired_length: NonEmptyString | None = None
    call_to_action: NonEmptyString | None = None
    subreddit: NonEmptyString | None = None
    target_url: NonEmptyString | None = None
    parent_url: NonEmptyString | None = None
    source_post_text: str | None = None
    source_comment_text: str | None = None
    required_facts: tuple[FactRequirement, ...] = ()
    forbidden_claims: tuple[NonEmptyString, ...] = ()
    forbidden_phrases: tuple[NonEmptyString, ...] = ()
    keywords: tuple[NonEmptyString, ...] = ()
    additional_instructions: NonEmptyString | None = None
    example_selection_filters: ExampleSelectionFilters | None = None
    candidate_count: int = Field(default=3, ge=1, le=10)
    profile_name: NonEmptyString | None = None
    campaign_id: NonEmptyString | None = None
    desired_generation_time: datetime | None = None
    unattended_approval_requested: bool = False
    account_context: AccountContext
    target_context: TargetContext | None = None
    selected_examples: tuple[SelectedExample, ...] = Field(default=(), exclude=True)
    strategy_names: tuple[NonEmptyString, ...] = ()
    resolved_parameters: dict[str, Any] = Field(default_factory=dict)

    @field_validator("desired_generation_time")
    @classmethod
    def normalize_timestamp(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def validate_request(self) -> Self:
        expected_platform = (
            Platform.X
            if self.generation_type in {GenerationType.X_POST, GenerationType.X_REPLY}
            else Platform.REDDIT
        )
        if self.platform is not expected_platform:
            raise ValueError("platform does not match generation_type")
        if self.account_context.platform is not self.platform:
            raise ValueError("account context platform does not match request platform")
        if self.account_context.account_name != self.account_name:
            raise ValueError("account context does not match request account")
        if self.target_context is not None and self.target_context.platform is not self.platform:
            raise ValueError("target context platform does not match request platform")
        if self.content_purpose is None:
            default = (
                ContentPurpose.EDUCATIONAL
                if self.generation_type in {GenerationType.X_POST, GenerationType.REDDIT_POST}
                else ContentPurpose.ORGANIC_DISCUSSION
            )
            object.__setattr__(self, "content_purpose", default)
        if self.generation_type is GenerationType.REDDIT_POST and not self.subreddit:
            raise ValueError("subreddit is required for reddit_post generation")
        if self.generation_type in {
            GenerationType.X_REPLY,
            GenerationType.REDDIT_COMMENT,
            GenerationType.REDDIT_REPLY,
        } and not self.target_url and not self.target_context:
            raise ValueError("reply and comment generation requires target context")
        return self


class CandidateDraft(_ContentModel):
    title: NonEmptyString | None = None
    body: NonEmptyString
    strategy: NonEmptyString
    used_example_ids: tuple[UUID, ...] = ()


class StructuredCandidateResponse(_ContentModel):
    title: NonEmptyString | None = None
    body: NonEmptyString
    strategy: NonEmptyString
    used_example_ids: tuple[UUID, ...] = ()


class StructuredGenerationResponse(_ContentModel):
    candidates: tuple[StructuredCandidateResponse, ...] = Field(min_length=1)


class GenerationResult(_ContentModel):
    candidates: tuple[CandidateDraft, ...] = Field(min_length=1)
    model_name: NonEmptyString
    resolved_parameters: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)
    latency_seconds: float = Field(default=0.0, ge=0)
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("created_at")
    @classmethod
    def normalize_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)


class ValidationFinding(_ContentModel):
    code: NonEmptyString
    severity: ValidationSeverity
    message: NonEmptyString
    field: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ValidationResult(_ContentModel):
    findings: tuple[ValidationFinding, ...] = ()

    @property
    def has_errors(self) -> bool:
        return any(item.severity is ValidationSeverity.ERROR for item in self.findings)

    @property
    def has_warnings(self) -> bool:
        return any(item.severity is ValidationSeverity.WARNING for item in self.findings)


class RankingResult(_ContentModel):
    score: float = Field(ge=0, le=10)
    explanation: NonEmptyString
    mode: RankingMode
    components: dict[str, float] = Field(default_factory=dict)


class SanitizedFailure(_ContentModel):
    error_type: NonEmptyString
    message: NonEmptyString
    retryable: bool = False
    retry_at: datetime | None = None
    response_sha256: str | None = None
    response_length: int | None = Field(default=None, ge=0)
    response_excerpt: str | None = None

    @field_validator("retry_at")
    @classmethod
    def normalize_timestamp(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)


class StoredGenerationRequest(_ContentModel):
    id: UUID
    generation_type: GenerationType
    platform: Platform
    account_name: NonEmptyString
    campaign_id: NonEmptyString | None = None
    status: GenerationStatus
    request_snapshot: dict[str, Any] = Field(default_factory=dict)
    resolved_profile: dict[str, Any] = Field(default_factory=dict)
    target_snapshot: dict[str, Any] | None = None
    selected_examples: tuple[SelectedExample, ...] = Field(default=(), exclude=True)
    prompt_version: NonEmptyString
    schema_version: NonEmptyString
    desired_generation_time: datetime | None = None
    claim_owner: str | None = None
    claim_expires_at: datetime | None = None
    attempt_count: int = Field(default=0, ge=0)
    next_retry_at: datetime | None = None
    completed_at: datetime | None = None
    failure: SanitizedFailure | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    unattended_approval_requested: bool = False
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    @field_validator(
        "desired_generation_time",
        "claim_expires_at",
        "next_retry_at",
        "completed_at",
        "created_at",
        "updated_at",
    )
    @classmethod
    def normalize_timestamp(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)


class PreparedGenerationData(_ContentModel):
    target_context: TargetContext | None = None
    selected_examples: tuple[SelectedExample, ...] = ()
    request_snapshot: dict[str, Any] = Field(default_factory=dict)
    resolved_profile: dict[str, Any] = Field(default_factory=dict)
    prompt_version: NonEmptyString = "social-content-v1"
    schema_version: NonEmptyString = "social-content-schema-v1"


class CandidateDecision(_ContentModel):
    method: NonEmptyString
    decided_at: datetime = Field(default_factory=utc_now)
    note: str | None = None

    @field_validator("decided_at")
    @classmethod
    def normalize_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)


class StoredCandidate(_ContentModel):
    id: UUID
    request_id: UUID
    ordinal: int = Field(ge=1)
    revision_of_candidate_id: UUID | None = None
    draft: CandidateDraft
    model_name: NonEmptyString
    generation_parameters: dict[str, Any] = Field(default_factory=dict)
    generated_at: datetime = Field(default_factory=utc_now)
    validation: ValidationResult = Field(default_factory=ValidationResult)
    ranking: RankingResult | None = None
    approval_status: CandidateApprovalStatus = CandidateApprovalStatus.PENDING
    decision: CandidateDecision | None = None
    social_action_id: UUID | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("generated_at")
    @classmethod
    def normalize_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)


class DiscoveredTopic(_ContentModel):
    id: UUID = Field(default_factory=uuid4)
    platform: Platform
    label: NonEmptyString
    keywords: tuple[NonEmptyString, ...]
    supporting_example_ids: tuple[UUID, ...]
    support_count: int = Field(ge=0)
    distinct_source_count: int = Field(ge=0)
    median_recency: datetime | None = None
    discovered_at: datetime = Field(default_factory=utc_now)
    expires_at: datetime | None = None
    is_active: bool = True
    score: float = Field(default=0.0, ge=0, le=10)

    @field_validator("median_recency", "discovered_at", "expires_at")
    @classmethod
    def normalize_timestamp(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)


class ValidatedCandidate(_ContentModel):
    id: UUID
    ordinal: int = Field(ge=1)
    draft: CandidateDraft
    validation: ValidationResult


class RankedCandidate(_ContentModel):
    candidate: ValidatedCandidate
    ranking: RankingResult


__all__ = [
    "AccountContext",
    "CandidateApprovalStatus",
    "CandidateDecision",
    "CandidateDraft",
    "ContentPurpose",
    "ContentRequest",
    "DiscoveredTopic",
    "FactRequirement",
    "GenerationResult",
    "GenerationStatus",
    "GenerationType",
    "PreparedGenerationData",
    "RankedCandidate",
    "RankingMode",
    "RankingResult",
    "SanitizedFailure",
    "StoredCandidate",
    "StoredGenerationRequest",
    "StructuredCandidateResponse",
    "StructuredGenerationResponse",
    "ValidatedCandidate",
    "ValidationFinding",
    "ValidationResult",
    "ValidationSeverity",
]
