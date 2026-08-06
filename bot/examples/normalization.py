"""Deterministic normalization and pseudonymous identity helpers."""

from __future__ import annotations

import hashlib
import json
import unicodedata
from datetime import UTC, datetime
from typing import overload
from uuid import UUID, uuid4

from bot.examples.models import (
    CollectedExample,
    ContentExample,
    ExampleType,
    PromptInjectionSeverity,
)
from bot.models import Platform


def normalize_authored_text(value: str) -> str:
    """Normalize inline whitespace while retaining authored paragraph breaks."""

    lines = [
        " ".join(line.split())
        for line in value.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    ]
    paragraphs: list[str] = []
    pending_blank = False
    for line in lines:
        if line:
            if pending_blank and paragraphs:
                paragraphs.append("")
            paragraphs.append(line)
            pending_blank = False
        else:
            pending_blank = True
    return "\n".join(paragraphs).strip()


def _hash_payload(parts: tuple[str, ...]) -> str:
    payload = json.dumps(parts, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


@overload
def content_hash(example_or_platform: CollectedExample) -> str: ...


@overload
def content_hash(
    example_or_platform: Platform,
    content_type: ExampleType,
    title: str,
    body: str,
    parent_text: str | None = None,
) -> str: ...


def content_hash(
    example_or_platform: CollectedExample | Platform,
    content_type: ExampleType | None = None,
    title: str = "",
    body: str = "",
    parent_text: str | None = None,
) -> str:
    if isinstance(example_or_platform, CollectedExample):
        example = example_or_platform
        platform = example.platform
        content_type = example.content_type
        title = example.title or ""
        body = example.body
        parent_text = example.parent_text
    elif content_type is not None:
        platform = example_or_platform
    else:
        raise TypeError("content_hash requires an example or content_type")
    assert content_type is not None
    return _hash_payload(
        (
            platform.value,
            content_type.value,
            normalize_authored_text(title).casefold(),
            normalize_authored_text(body).casefold(),
            normalize_authored_text(parent_text or "").casefold(),
        )
    )


def author_identifier(platform: Platform, username: str) -> str:
    """Return a stable platform-scoped pseudonym without retaining the username."""

    normalized = unicodedata.normalize("NFKC", username).strip().casefold()
    if not normalized:
        raise ValueError("username must not be blank")
    return _hash_payload((f"{platform.value}:{normalized}",))


def normalize_example(
    example: CollectedExample,
    *,
    collection_run_id: UUID | None = None,
    expires_at: datetime | None = None,
) -> ContentExample:
    """Create a stored example while preserving title/body separately."""

    title = normalize_authored_text(example.title or "") or None
    body = normalize_authored_text(example.body)
    parent_text = normalize_authored_text(example.parent_text or "") or None
    authored_size = sum(
        not character.isspace() for character in f"{title or ''}\n{body}"
    )
    if authored_size < 20:
        raise ValueError("example content is too short after normalization")
    expiry = expires_at or example.expires_at
    if expiry is not None:
        if expiry.tzinfo is None or expiry.utcoffset() is None:
            expiry = expiry.replace(tzinfo=UTC)
        else:
            expiry = expiry.astimezone(UTC)
    high_risk = any(
        isinstance(finding, dict)
        and finding.get("severity") == PromptInjectionSeverity.HIGH.value
        for finding in example.injection_findings
    )
    return ContentExample(
        platform=example.platform,
        content_type=example.content_type,
        external_id=example.external_id,
        source_url=example.source_url,
        author_identifier=example.author_identifier,
        title=title,
        body=body,
        parent_text=parent_text,
        subreddit=example.subreddit,
        published_at=example.published_at,
        collected_at=example.collected_at,
        expires_at=expiry,
        content_hash=content_hash(example.platform, example.content_type, title or "", body, parent_text),
        engagement_score=example.engagement_score,
        metadata=dict(example.metadata),
        topic_tags=example.topic_tags,
        is_own_content=example.is_own_content,
        generated=example.generated,
        is_active=not high_risk,
        is_quarantined=high_risk,
        injection_findings=example.injection_findings,
        id=uuid4(),
        collection_run_id=collection_run_id,
    )


__all__ = [
    "author_identifier",
    "content_hash",
    "normalize_authored_text",
    "normalize_example",
]
