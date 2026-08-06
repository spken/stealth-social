"""Deterministic topic discovery from stored public examples."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

from bot.config import Settings
from bot.content.models import (
    AccountContext,
    ContentRequest,
    GenerationType,
    DiscoveredTopic,
)
from bot.content.service import GenerationPipelineResult, GenerationService
from bot.examples.models import ContentExample, ExampleListFilters, ExampleSelectionFilters
from bot.examples.normalization import normalize_authored_text
from bot.models import Platform
from bot.storage.content_repository import ContentRepository

_STOPWORDS = frozenset(
    "a an and are as at be been being by can could did do does for from had has have how "
    "i if in into is it its may might more most must no not of on or our should so than "
    "that the their them then there these they this to too was we were what when where "
    "which who why will with would you your"
    .split()
)
_BLOCKED_TOPIC_TERMS = frozenset(
    {"always", "best", "guaranteed", "guarantee", "predict", "prediction", "viral"}
)
_TOKEN = re.compile(r"(?<![@#])[a-z][a-z0-9_-]{2,}", re.IGNORECASE)
_URL = re.compile(r"https?://[^\s]+", re.IGNORECASE)


class TopicDiscoveryService:
    """Group recurring themes using only persisted, normalized example metadata."""

    def __init__(
        self,
        settings: Settings,
        content_repository: ContentRepository,
        generation_service: GenerationService,
    ) -> None:
        self._settings = settings
        self._content = content_repository
        self._generation = generation_service

    async def discover(
        self,
        platform: Platform,
        *,
        since: datetime,
        minimum_examples: int = 3,
        minimum_sources: int = 2,
    ) -> list[DiscoveredTopic]:
        if minimum_examples < 1 or minimum_sources < 1:
            raise ValueError("minimum examples and sources must be positive")
        cutoff = _aware(since)
        examples = await self._content.list_examples(
            ExampleListFilters(platform=platform, active_only=True, limit=1000)
        )
        recent = [
            example
            for example in examples
            if example.collected_at >= cutoff
            and not example.is_quarantined
            and not example.generated
        ]
        keyword_sets = {
            example.id: _keywords(example)
            for example in recent
        }
        components = _connected_components(recent, keyword_sets)
        now = datetime.now(UTC)
        topics: list[DiscoveredTopic] = []
        for component in components:
            supported = _cap_source_contribution(component)
            source_keys = {_topic_source_key(example) for example in supported}
            if len(supported) < minimum_examples or len(source_keys) < minimum_sources:
                continue
            labels = _labels(supported, keyword_sets)
            if not labels:
                continue
            stamps = sorted(
                (example.published_at or example.collected_at).astimezone(UTC)
                for example in supported
            )
            median = stamps[len(stamps) // 2]
            recency = max(0.0, min(1.0, 1.0 - (now - median).total_seconds() / 90 / 86400))
            engagement_tiebreak = min(
                0.5,
                sum(max(example.engagement_score, 0.0) for example in supported)
                / max(1, len(supported) * 10_000),
            )
            topics.append(
                DiscoveredTopic(
                    platform=platform,
                    label=" ".join(labels),
                    keywords=tuple(labels),
                    supporting_example_ids=tuple(
                        example.id for example in supported
                    ),
                    support_count=len(supported),
                    distinct_source_count=len(source_keys),
                    median_recency=median,
                    discovered_at=now,
                    expires_at=now + timedelta(days=7),
                    score=min(10.0, len(supported) + len(source_keys) * 0.5 + recency + engagement_tiebreak),
                )
            )
        topics.sort(key=lambda topic: (-topic.score, topic.label, str(topic.id)))
        return await self._content.replace_topics(
            platform,
            topics,
            discovered_at=now,
        )

    async def list(self, platform: Platform | None = None) -> list[DiscoveredTopic]:
        return await self._content.list_topics(platform, active_only=True)

    async def create_generation_request(
        self,
        topic_id: UUID,
        action_type: GenerationType | str,
        account: str,
        overrides: Mapping[str, Any] | None = None,
    ) -> GenerationPipelineResult:
        topic = await self._content.get_topic(topic_id)
        if topic is None or not topic.is_active:
            raise ValueError(f"active topic {topic_id} was not found")
        now = datetime.now(UTC)
        if topic.expires_at is not None and topic.expires_at <= now:
            raise ValueError(f"topic {topic_id} has expired")
        generation_type = GenerationType(action_type)
        expected_platform = _platform_for(generation_type)
        if topic.platform is not expected_platform:
            raise ValueError("generation type does not match topic platform")
        account_settings = (
            self._settings.accounts.x.get(account)
            if expected_platform is Platform.X
            else self._settings.accounts.reddit.get(account)
        )
        if account_settings is None or not account_settings.enabled:
            raise ValueError(f"no enabled {expected_platform.value} account named {account!r}")

        values = dict(overrides or {})
        values.update(
            {
                "generation_type": generation_type,
                "platform": expected_platform,
                "account_name": account,
                "topic": values.get("topic") or topic.label,
                "keywords": tuple(
                    dict.fromkeys((*values.get("keywords", ()), *topic.keywords))
                ),
                "account_context": _account_context(
                    expected_platform, account, account_settings
                ),
            }
        )
        raw_filters = values.get("example_selection_filters")
        filters = ExampleSelectionFilters.model_validate(
            raw_filters or {"platform": expected_platform}
        )
        values["example_selection_filters"] = filters.model_copy(
            update={
                "platform": expected_platform,
                "preferred_example_ids": topic.supporting_example_ids,
            }
        )
        request = ContentRequest.model_validate(values)
        return await self._generation.create(request)


def _aware(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _keywords(example: ContentExample) -> frozenset[str]:
    text = _URL.sub(" ", normalize_authored_text(f"{example.title or ''}\n{example.body}"))
    counts = Counter(
        token.casefold()
        for token in _TOKEN.findall(text)
        if token.casefold() not in _STOPWORDS and not token.isnumeric()
        and token.casefold() not in _BLOCKED_TOPIC_TERMS
    )
    tokens = {
        token
        for token, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:12]
    }
    tags = {
        " ".join(tag.split()).casefold()
        for tag in example.topic_tags
        if " ".join(tag.split())
        and not _BLOCKED_TOPIC_TERMS.intersection(" ".join(tag.split()).casefold().split())
    }
    return frozenset(tokens | tags)


def _connected_components(
    examples: list[ContentExample],
    keyword_sets: dict[UUID, frozenset[str]],
) -> list[list[ContentExample]]:
    remaining = {example.id: example for example in examples}
    components: list[list[ContentExample]] = []
    while remaining:
        seed_id = min(remaining, key=str)
        stack = [seed_id]
        component: list[ContentExample] = []
        while stack:
            current_id = stack.pop()
            current = remaining.pop(current_id, None)
            if current is None:
                continue
            component.append(current)
            current_words = keyword_sets[current.id]
            for other_id, other in tuple(remaining.items()):
                other_words = keyword_sets[other_id]
                union = current_words | other_words
                overlap = len(current_words & other_words) / len(union) if union else 0.0
                if overlap >= 0.35:
                    stack.append(other_id)
        components.append(component)
    return components


def _cap_source_contribution(examples: list[ContentExample]) -> list[ContentExample]:
    counts: Counter[str] = Counter()
    selected: list[ContentExample] = []
    ordered = sorted(
        examples,
        key=lambda example: (
            -(example.published_at or example.collected_at).timestamp(),
            str(example.id),
        ),
    )
    for example in ordered:
        source = _topic_source_key(example)
        if counts[source] >= 2:
            continue
        counts[source] += 1
        selected.append(example)
    return selected


def _topic_source_key(example: ContentExample) -> str:
    """Use one stable identity per example for diversity accounting."""

    return example.author_identifier or example.source_url


def _labels(
    examples: list[ContentExample],
    keyword_sets: dict[UUID, frozenset[str]],
) -> tuple[str, ...]:
    document_frequency = Counter(
        token for example in examples for token in keyword_sets[example.id]
    )
    return tuple(
        token
        for token, _ in sorted(
            document_frequency.items(), key=lambda item: (-item[1], item[0])
        )[:3]
    )


def _platform_for(generation_type: GenerationType) -> Platform:
    return (
        Platform.X
        if generation_type in {GenerationType.X_POST, GenerationType.X_REPLY}
        else Platform.REDDIT
    )


def _account_context(platform: Platform, account: str, settings: Any) -> AccountContext:
    return AccountContext(
        account_name=account,
        platform=platform,
        identity=settings.identity,
        products=tuple(settings.products),
        verified_facts=tuple(settings.verified_facts),
        forbidden_claims=tuple(settings.forbidden_claims),
        required_disclosures=tuple(settings.required_disclosures),
    )


__all__ = ["TopicDiscoveryService"]
