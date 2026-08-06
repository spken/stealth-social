"""Shared domain models for social actions."""

from __future__ import annotations

import hashlib
import json
import unicodedata
from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Any, Self
from uuid import UUID, uuid4
from urllib.parse import urlsplit, urlunsplit

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    PositiveInt,
    StringConstraints,
    field_validator,
    model_validator,
)

NonEmptyString = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]


class ActionType(StrEnum):
    """Supported social action kinds."""

    X_POST = "x_post"
    REDDIT_POST = "reddit_post"
    REDDIT_COMMENT = "reddit_comment"
    REDDIT_REPLY = "reddit_reply"


class Platform(StrEnum):
    """Supported social platforms."""

    X = "x"
    REDDIT = "reddit"


class ActionStatus(StrEnum):
    """Lifecycle states for a social action."""

    DRAFT = "draft"
    PENDING_APPROVAL = "pending_approval"
    SCHEDULED = "scheduled"
    PROCESSING = "processing"
    PUBLISHED = "published"
    FAILED = "failed"
    CANCELLED = "cancelled"


def normalize_target_url(value: str) -> str:
    """Canonicalize an absolute HTTP(S) URL used as an action target."""
    try:
        parsed = urlsplit(value.strip())
        port = parsed.port
    except ValueError as exc:
        raise ValueError("target_url must be a valid absolute HTTP(S) URL") from exc

    scheme = parsed.scheme.casefold()
    hostname = parsed.hostname
    if scheme not in {"http", "https"} or not hostname:
        raise ValueError("target_url must be a valid absolute HTTP(S) URL")

    userinfo, separator, _ = parsed.netloc.rpartition("@")
    normalized_hostname = hostname.casefold()
    authority = f"{userinfo}@" if separator else ""
    authority += (
        f"[{normalized_hostname}]" if ":" in normalized_hostname else normalized_hostname
    )
    is_default_port = (scheme == "http" and port == 80) or (
        scheme == "https" and port == 443
    )
    if port is not None and not is_default_port:
        authority += f":{port}"

    path = parsed.path.rstrip("/") or "/"
    return urlunsplit((scheme, authority, path, parsed.query, ""))


class _DomainModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        revalidate_instances="always",
        str_strip_whitespace=True,
        validate_default=True,
    )


class SocialAction(_DomainModel):
    """A validated, platform-neutral unit of social publishing work."""

    id: UUID = Field(default_factory=uuid4)
    action_type: ActionType
    platform: Platform
    account_name: NonEmptyString
    content: str = ""
    title: str | None = None
    subreddit: str | None = None
    target_url: str | None = None
    parent_post_id: str | None = None
    parent_comment_id: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    scheduled_at: datetime | None = None
    claim_owner: str | None = None
    claim_expires_at: datetime | None = None
    external_dispatch_started_at: datetime | None = None
    retry_available_at: datetime | None = None
    published_at: datetime | None = None
    status: ActionStatus = ActionStatus.DRAFT
    attempts: NonNegativeInt = 0
    max_attempts: PositiveInt = 3
    last_error: str | None = None
    external_content_id: str | None = None
    external_content_url: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    fingerprint: str = ""

    @field_validator(
        "title",
        "subreddit",
        "target_url",
        "parent_post_id",
        "parent_comment_id",
        "last_error",
        "external_content_id",
        "external_content_url",
    )
    @classmethod
    def normalize_optional_strings(cls, value: str | None) -> str | None:
        """Represent blank optional fields consistently as absent."""
        return value or None

    @field_validator(
        "created_at",
        "scheduled_at",
        "claim_expires_at",
        "external_dispatch_started_at",
        "retry_available_at",
        "published_at",
    )
    @classmethod
    def normalize_timestamp(cls, value: datetime | None) -> datetime | None:
        """Interpret naive timestamps as UTC and convert aware values to UTC."""
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)

    @field_validator("fingerprint")
    @classmethod
    def validate_supplied_fingerprint(cls, value: str) -> str:
        normalized = value.casefold()
        if normalized and (
            len(normalized) != 64
            or any(character not in "0123456789abcdef" for character in normalized)
        ):
            raise ValueError("fingerprint must be a 64-character SHA-256 hex digest")
        return normalized

    @model_validator(mode="after")
    def validate_action(self) -> Self:
        expected_platform = (
            Platform.X if self.action_type is ActionType.X_POST else Platform.REDDIT
        )
        if self.platform is not expected_platform:
            raise ValueError(
                f"platform must be '{expected_platform.value}' when action_type is "
                f"'{self.action_type.value}'"
            )
        if self.target_url is not None:
            normalize_target_url(self.target_url)

        if self.action_type is ActionType.X_POST:
            self._validate_x_post()
        elif self.action_type is ActionType.REDDIT_POST:
            self._validate_reddit_post()
        elif self.action_type is ActionType.REDDIT_COMMENT:
            self._validate_reddit_comment()
        else:
            self._validate_reddit_reply()

        expected_fingerprint = normalized_content_fingerprint(self)
        if self.fingerprint and self.fingerprint != expected_fingerprint:
            raise ValueError(
                "fingerprint does not match the action's normalized semantic content"
            )
        object.__setattr__(self, "fingerprint", expected_fingerprint)
        return self

    def _validate_x_post(self) -> None:
        if not self.content:
            raise ValueError("content is required for action_type 'x_post'")
        if len(self.content) > 280:
            raise ValueError("content must be at most 280 characters for action_type 'x_post'")
        self._reject_fields(
            "x_post",
            "title",
            "subreddit",
            "parent_comment_id",
        )

    def _validate_reddit_post(self) -> None:
        if not self.subreddit:
            raise ValueError("subreddit is required for action_type 'reddit_post'")
        if not self.title:
            raise ValueError("title is required for action_type 'reddit_post'")
        if bool(self.content) == bool(self.target_url):
            raise ValueError(
                "reddit_post requires exactly one of content (text post) or "
                "target_url (link post)"
            )
        self._reject_fields("reddit_post", "parent_post_id", "parent_comment_id")

    def _validate_reddit_comment(self) -> None:
        if not self.content:
            raise ValueError("content is required for action_type 'reddit_comment'")
        if not self.parent_post_id and not self.target_url:
            raise ValueError(
                "reddit_comment requires parent_post_id or target_url to identify "
                "the target post"
            )
        self._reject_fields("reddit_comment", "title", "parent_comment_id")

    def _validate_reddit_reply(self) -> None:
        if not self.content:
            raise ValueError("content is required for action_type 'reddit_reply'")
        if not self.parent_comment_id and not self.target_url:
            raise ValueError(
                "reddit_reply requires parent_comment_id or target_url to identify "
                "the target comment"
            )
        self._reject_fields("reddit_reply", "title")

    def _reject_fields(self, action_name: str, *field_names: str) -> None:
        populated_fields = [
            field_name for field_name in field_names if getattr(self, field_name) is not None
        ]
        if populated_fields:
            joined_fields = ", ".join(populated_fields)
            raise ValueError(f"{action_name} does not accept fields: {joined_fields}")


def canonical_target_scope(action: SocialAction) -> str:
    """Return the canonical semantic destination for duplicate detection."""
    action_type = ActionType(action.action_type)

    if action_type is ActionType.X_POST:
        if action.parent_post_id:
            return f"x:post:id:{_normalize_fingerprint_part(action.parent_post_id)}"
        if action.target_url:
            return f"x:post:url:{normalize_target_url(action.target_url)}"
        return "x:feed"

    if action_type is ActionType.REDDIT_POST:
        if not action.subreddit:
            raise ValueError("reddit posts require subreddit for target scope")
        return f"reddit:subreddit:{_normalize_fingerprint_part(action.subreddit)}"

    if action_type is ActionType.REDDIT_COMMENT:
        if action.parent_post_id:
            target = f"id:{_normalize_fingerprint_part(action.parent_post_id)}"
        elif action.target_url:
            target = f"url:{normalize_target_url(action.target_url)}"
        else:
            raise ValueError("reddit comments require a parent post target")
        return f"reddit:post:{target}"

    if action_type is ActionType.REDDIT_REPLY:
        if action.parent_comment_id:
            target = f"id:{_normalize_fingerprint_part(action.parent_comment_id)}"
        elif action.target_url:
            target = f"url:{normalize_target_url(action.target_url)}"
        else:
            raise ValueError("reddit replies require a parent comment target")
        return f"reddit:comment:{target}"

    raise ValueError(f"unsupported action type: {action_type}")


class ActionResult(_DomainModel):
    """Outcome returned by a platform adapter for one action."""

    action_id: UUID
    success: bool
    external_content_id: str | None = None
    external_content_url: str | None = None
    message: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


def normalized_content_fingerprint(action: SocialAction) -> str:
    """Hash normalized semantic fields in an action-specific scope."""
    action_type = ActionType(action.action_type)
    common_parts = (
        _normalize_fingerprint_part(action_type.value),
        _normalize_fingerprint_part(action.account_name),
        canonical_target_scope(action),
    )

    if action_type is ActionType.X_POST:
        semantic_parts = (
            *common_parts,
            _normalize_fingerprint_part(action.content),
        )
    elif action_type is ActionType.REDDIT_POST:
        content_kind = "text" if action.content else "link"
        content_or_url = (
            _normalize_fingerprint_part(action.content)
            if action.content
            else normalize_target_url(action.target_url or "")
        )
        semantic_parts = (
            *common_parts,
            _normalize_fingerprint_part(action.title or ""),
            content_kind,
            content_or_url,
        )
    else:
        semantic_parts = (
            *common_parts,
            _normalize_fingerprint_part(action.content),
        )

    canonical_payload = json.dumps(
        semantic_parts,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical_payload).hexdigest()


def _normalize_fingerprint_part(value: str) -> str:
    normalized_unicode = unicodedata.normalize("NFKC", value)
    return " ".join(normalized_unicode.split()).casefold()
