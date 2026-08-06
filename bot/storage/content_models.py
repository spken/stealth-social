"""SQLAlchemy records for the disposable content-generation schema."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
    text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from bot.storage.models import Base, UTCDateTime, utc_now


class ExampleCollectionRunRecord(Base):
    """One bounded browser collection attempt."""

    __tablename__ = "example_collection_runs"
    __table_args__ = (
        CheckConstraint(
            "collected_count >= 0 AND rejected_count >= 0 AND "
            "duplicate_count >= 0 AND disabled_count >= 0",
            name="ck_collection_runs_counts_nonnegative",
        ),
        Index("ix_collection_runs_platform_status", "platform", "status", "started_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    platform: Mapped[str] = mapped_column(String(16), nullable=False)
    started_at: Mapped[datetime] = mapped_column(
        UTCDateTime(), nullable=False, default=utc_now
    )
    finished_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="running")
    request_json: Mapped[dict[str, Any]] = mapped_column(
        "request", JSON, nullable=False, default=dict
    )
    collected_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    rejected_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    duplicate_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    disabled_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    error_type: Mapped[str | None] = mapped_column(String(128))
    error_message: Mapped[str | None] = mapped_column(Text)
    retry_after_seconds: Mapped[float | None] = mapped_column()


class ContentExampleRecord(Base):
    """Normalized public example; authored text is never silently rewritten."""

    __tablename__ = "content_examples"
    __table_args__ = (
        UniqueConstraint(
            "platform", "content_type", "content_hash",
            name="uq_content_examples_semantic_hash",
        ),
        Index(
            "uq_content_examples_external_id",
            "platform", "external_id",
            unique=True,
            sqlite_where=text("external_id IS NOT NULL"),
        ),
        Index(
            "ix_content_examples_active_selection",
            "platform", "is_active", "is_quarantined", "expires_at", "content_type",
        ),
        Index("ix_content_examples_source", "source_url", "collected_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    collection_run_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("example_collection_runs.id", ondelete="SET NULL")
    )
    platform: Mapped[str] = mapped_column(String(16), nullable=False)
    content_type: Mapped[str] = mapped_column(String(32), nullable=False)
    external_id: Mapped[str | None] = mapped_column(String(255))
    source_url: Mapped[str] = mapped_column(Text, nullable=False)
    author_identifier: Mapped[str | None] = mapped_column(String(128))
    title: Mapped[str | None] = mapped_column(Text)
    body: Mapped[str] = mapped_column(Text, nullable=False, default="")
    parent_text: Mapped[str | None] = mapped_column(Text)
    subreddit: Mapped[str | None] = mapped_column(String(255))
    published_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    collected_at: Mapped[datetime] = mapped_column(
        UTCDateTime(), nullable=False, default=utc_now
    )
    expires_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    engagement_score: Mapped[float] = mapped_column(nullable=False, default=0.0)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSON, nullable=False, default=dict
    )
    topic_tags_json: Mapped[list[str]] = mapped_column(
        "topic_tags", JSON, nullable=False, default=list
    )
    is_own_content: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    generated: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    is_quarantined: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    injection_findings_json: Mapped[list[dict[str, Any]]] = mapped_column(
        "injection_findings", JSON, nullable=False, default=list
    )


class GenerationRequestRecord(Base):
    """Generation request and scheduled-generation queue row."""

    __tablename__ = "generation_requests"
    __table_args__ = (
        CheckConstraint("attempt_count >= 0", name="ck_generation_requests_attempts_nonnegative"),
        CheckConstraint(
            "(claim_owner IS NULL AND claim_expires_at IS NULL) OR "
            "(claim_owner IS NOT NULL AND claim_expires_at IS NOT NULL)",
            name="ck_generation_requests_claim_lease_complete",
        ),
        Index(
            "ix_generation_requests_due_claims",
            "status", "desired_generation_time", "next_retry_at", "claim_expires_at",
        ),
        Index("ix_generation_requests_account_status", "platform", "account_name", "status"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    generation_type: Mapped[str] = mapped_column(String(32), nullable=False)
    platform: Mapped[str] = mapped_column(String(16), nullable=False)
    account_name: Mapped[str] = mapped_column(String(255), nullable=False)
    campaign_id: Mapped[str | None] = mapped_column(String(255))
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    request_json: Mapped[dict[str, Any]] = mapped_column(
        "request_snapshot", JSON, nullable=False, default=dict
    )
    resolved_profile_json: Mapped[dict[str, Any]] = mapped_column(
        "resolved_profile", JSON, nullable=False, default=dict
    )
    target_snapshot_json: Mapped[dict[str, Any] | None] = mapped_column(
        "target_snapshot", JSON
    )
    selected_examples_json: Mapped[list[dict[str, Any]]] = mapped_column(
        "selected_examples", JSON, nullable=False, default=list
    )
    prompt_version: Mapped[str] = mapped_column(String(128), nullable=False)
    schema_version: Mapped[str] = mapped_column(String(128), nullable=False)
    desired_generation_time: Mapped[datetime | None] = mapped_column(UTCDateTime())
    claim_owner: Mapped[str | None] = mapped_column(String(255))
    claim_expires_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    attempt_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    next_retry_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    completed_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    failure_json: Mapped[dict[str, Any] | None] = mapped_column("failure", JSON)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSON, nullable=False, default=dict
    )
    unattended_approval_requested: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )
    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime(), nullable=False, default=utc_now
    )
    updated_at: Mapped[datetime] = mapped_column(
        UTCDateTime(), nullable=False, default=utc_now, onupdate=utc_now
    )


class GeneratedCandidateRecord(Base):
    """Immutable generated candidate and its approval/promotion evidence."""

    __tablename__ = "generated_candidates"
    __table_args__ = (
        UniqueConstraint(
            "request_id", "ordinal", name="uq_generated_candidates_request_ordinal"
        ),
        CheckConstraint("ordinal > 0", name="ck_generated_candidates_ordinal_positive"),
        CheckConstraint(
            "ranking_score IS NULL OR (ranking_score >= 0 AND ranking_score <= 10)",
            name="ck_generated_candidates_rank_bounded",
        ),
        Index(
            "ix_generated_candidates_request_rank",
            "request_id", "ranking_score", "ordinal",
        ),
        Index("ix_generated_candidates_request_status", "request_id", "approval_status"),
        Index(
            "uq_generated_candidates_approved_request",
            "request_id",
            unique=True,
            sqlite_where=text("approval_status = 'approved'"),
        ),
        Index(
            "uq_generated_candidates_social_action",
            "social_action_id",
            unique=True,
            sqlite_where=text("social_action_id IS NOT NULL"),
        ),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    request_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("generation_requests.id", ondelete="CASCADE"), nullable=False
    )
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    revision_of_candidate_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("generated_candidates.id", ondelete="SET NULL")
    )
    title: Mapped[str | None] = mapped_column(Text)
    body: Mapped[str] = mapped_column(Text, nullable=False)
    strategy: Mapped[str] = mapped_column(Text, nullable=False)
    model_name: Mapped[str] = mapped_column(String(255), nullable=False)
    generation_parameters_json: Mapped[dict[str, Any]] = mapped_column(
        "generation_parameters", JSON, nullable=False, default=dict
    )
    used_example_ids_json: Mapped[list[str]] = mapped_column(
        "used_example_ids", JSON, nullable=False, default=list
    )
    generated_at: Mapped[datetime] = mapped_column(
        UTCDateTime(), nullable=False, default=utc_now
    )
    validation_json: Mapped[dict[str, Any]] = mapped_column(
        "validation", JSON, nullable=False, default=dict
    )
    ranking_score: Mapped[float | None] = mapped_column()
    ranking_explanation: Mapped[str | None] = mapped_column(Text)
    ranking_json: Mapped[dict[str, Any] | None] = mapped_column("ranking", JSON)
    approval_status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="pending"
    )
    decision_json: Mapped[dict[str, Any] | None] = mapped_column("decision", JSON)
    social_action_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("social_actions.id", ondelete="SET NULL")
    )
    metadata_json: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSON, nullable=False, default=dict
    )


class DiscoveredTopicRecord(Base):
    """Immutable topic-discovery snapshot for one platform."""

    __tablename__ = "discovered_topics"
    __table_args__ = (
        CheckConstraint(
            "support_count >= 0 AND distinct_source_count >= 0",
            name="ck_discovered_topics_counts_nonnegative",
        ),
        CheckConstraint(
            "score >= 0 AND score <= 10",
            name="ck_discovered_topics_score_bounded",
        ),
        Index("ix_discovered_topics_active_platform", "platform", "is_active", "discovered_at"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    platform: Mapped[str] = mapped_column(String(16), nullable=False)
    label: Mapped[str] = mapped_column(String(255), nullable=False)
    keywords_json: Mapped[list[str]] = mapped_column(
        "keywords", JSON, nullable=False, default=list
    )
    supporting_example_ids_json: Mapped[list[str]] = mapped_column(
        "supporting_example_ids", JSON, nullable=False, default=list
    )
    support_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    distinct_source_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    median_recency: Mapped[datetime | None] = mapped_column(UTCDateTime())
    discovered_at: Mapped[datetime] = mapped_column(
        UTCDateTime(), nullable=False, default=utc_now
    )
    expires_at: Mapped[datetime | None] = mapped_column(UTCDateTime())
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    score: Mapped[float] = mapped_column(nullable=False, default=0.0)


__all__ = [
    "ContentExampleRecord",
    "DiscoveredTopicRecord",
    "ExampleCollectionRunRecord",
    "GeneratedCandidateRecord",
    "GenerationRequestRecord",
]
