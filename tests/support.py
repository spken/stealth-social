"""Deterministic builders shared by the autopost tests."""

from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from bot.config import Settings
from bot.content.models import (
    CandidateApprovalStatus,
    CandidateDecision,
    CandidateDraft,
    GenerationStatus,
    GenerationType,
    RankingMode,
    RankingResult,
    StoredCandidate,
    StoredGenerationRequest,
    ValidationResult,
)
from bot.models import ActionStatus, ActionType, Platform, SocialAction


def settings_values() -> dict[str, Any]:
    """Return an independent, valid baseline configuration mapping."""

    return {
        "database_url": "sqlite+aiosqlite:///:memory:",
        "dry_run": False,
        "manual_approval": True,
        "global_pause": False,
        "randomized_delay": {"minimum_seconds": 0, "maximum_seconds": 0},
        "automation": {
            "allow_scheduled_generation": True,
            "allow_unattended_approval": True,
            "allow_unattended_publishing": True,
        },
        "accounts": {
            "x": {
                "main": {
                    "session_profile": "x-main-test",
                    "identity": "Test Builder",
                    "verified_facts": ["The test project is local."],
                }
            },
            "reddit": {
                "main": {
                    "session_profile": "reddit-main-test",
                    "allowed_subreddits": ["SideProject"],
                    "identity": "Test Builder",
                    "community_rules": {
                        "SideProject": {"allow_promotional_content": False}
                    },
                }
            },
        },
        "autopost_campaigns": {
            "daily-x": {
                "platform": "x",
                "account": "main",
                "topics": ["First topic", "Second topic"],
                "minimum_interval_hours": 20,
                "purpose": "educational",
            },
            "weekly-reddit": {
                "platform": "reddit",
                "account": "main",
                "subreddit": "SideProject",
                "topics": ["Reddit topic one", "Reddit topic two"],
                "minimum_interval_hours": 144,
                "purpose": "builder-update",
            },
        },
    }


def make_settings(**top_level_updates: Any) -> Settings:
    """Build settings from a fresh baseline with shallow top-level overrides."""

    values = deepcopy(settings_values())
    values.update(top_level_updates)
    return Settings.model_validate(values)


def make_generation_request(
    *,
    campaign_id: str = "daily-x",
    topic: str = "First topic",
    created_at: datetime | None = None,
    status: GenerationStatus = GenerationStatus.COMPLETED,
    claim_owner: str | None = None,
    claim_expires_at: datetime | None = None,
    failure: Any = None,
    attempt_count: int = 0,
) -> StoredGenerationRequest:
    """Build one immutable persisted generation-request snapshot."""

    instant = created_at or datetime(2030, 1, 1, tzinfo=UTC)
    return StoredGenerationRequest(
        id=uuid4(),
        generation_type=GenerationType.X_POST,
        platform=Platform.X,
        account_name="main",
        campaign_id=campaign_id,
        status=status,
        request_snapshot={"topic": topic},
        prompt_version="test-prompt-v1",
        schema_version="test-schema-v1",
        claim_owner=claim_owner,
        claim_expires_at=claim_expires_at,
        attempt_count=attempt_count,
        failure=failure,
        created_at=instant,
        updated_at=instant,
    )


def make_candidate(
    request_id,
    *,
    action_id=None,
    approval_status: CandidateApprovalStatus = CandidateApprovalStatus.PENDING,
) -> StoredCandidate:
    """Build a safe, deterministic generated candidate."""

    return StoredCandidate(
        id=uuid4(),
        request_id=request_id,
        ordinal=1,
        draft=CandidateDraft(body="Safe generated body", strategy="test"),
        model_name="test-model",
        validation=ValidationResult(),
        ranking=RankingResult(
            score=9,
            explanation="deterministic test score",
            mode=RankingMode.HEURISTIC,
        ),
        approval_status=approval_status,
        decision=(CandidateDecision(method="test") if action_id else None),
        social_action_id=action_id,
    )


def make_action(
    *,
    action_id=None,
    status: ActionStatus = ActionStatus.DRAFT,
    created_at: datetime | None = None,
    scheduled_at: datetime | None = None,
    retry_available_at: datetime | None = None,
    published_at: datetime | None = None,
    claim_owner: str | None = None,
    claim_expires_at: datetime | None = None,
    external_dispatch_started_at: datetime | None = None,
) -> SocialAction:
    """Build one X action with lifecycle timestamps for repository tests."""

    instant = created_at or datetime(2030, 1, 1, tzinfo=UTC)
    return SocialAction(
        id=action_id or uuid4(),
        action_type=ActionType.X_POST,
        platform=Platform.X,
        account_name="main",
        content="Safe generated body",
        status=status,
        created_at=instant,
        scheduled_at=scheduled_at,
        claim_owner=claim_owner,
        claim_expires_at=claim_expires_at,
        retry_available_at=retry_available_at,
        published_at=published_at,
        external_dispatch_started_at=external_dispatch_started_at,
    )
