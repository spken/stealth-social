"""Transparent relevance and diversity selection for public examples."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from datetime import UTC, datetime
from difflib import SequenceMatcher

from bot.content.models import ContentRequest, GenerationType
from bot.examples.models import ContentExample, ExampleType, SelectedExample
from bot.examples.normalization import normalize_authored_text

COMPATIBLE_TYPES: dict[GenerationType, set[ExampleType]] = {
    GenerationType.X_POST: {ExampleType.X_POST},
    GenerationType.X_REPLY: {ExampleType.X_REPLY, ExampleType.X_POST},
    GenerationType.REDDIT_POST: {ExampleType.REDDIT_POST},
    GenerationType.REDDIT_COMMENT: {ExampleType.REDDIT_COMMENT, ExampleType.REDDIT_POST},
    GenerationType.REDDIT_REPLY: {ExampleType.REDDIT_COMMENT},
}

_STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the their "
    "this to was were with you your we our i me my".split()
)
_TOKEN = re.compile(r"[a-z0-9_]{3,}")


class NoRelevantExamplesFound(LookupError):
    """Internal diagnostic used when compatibility/scoring yields no examples."""


# TODO: Add optional local embedding retrieval through Ollama.
class ExampleSelector:
    """Select examples with a reproducible score breakdown."""

    def select(
        self,
        request: ContentRequest,
        pool: Sequence[ContentExample],
    ) -> tuple[SelectedExample, ...]:
        filters = request.example_selection_filters
        compatible = COMPATIBLE_TYPES[request.generation_type]
        target_text = " ".join(
            item
            for item in (
                request.source_post_text,
                request.source_comment_text,
                request.target_context.title if request.target_context else None,
                request.target_context.body if request.target_context else None,
                request.target_context.parent_text if request.target_context else None,
            )
            if item
        )
        topic_tokens = _tokens(" ".join(
            item for item in (request.topic, *request.keywords) if item
        ))
        target_tokens = _tokens(target_text)
        scored: list[tuple[float, ContentExample, dict[str, float], str]] = []
        now = datetime.now(UTC)
        for example in pool:
            if not example.is_active or example.is_quarantined:
                continue
            if example.platform is not request.platform:
                continue
            if example.content_type not in compatible:
                continue
            if example.generated and not (filters and filters.allow_generated):
                continue
            if filters and filters.content_type is not None and example.content_type is not filters.content_type:
                continue
            if (
                filters
                and filters.subreddit
                and (
                    example.subreddit is None
                    or example.subreddit.casefold() != filters.subreddit.casefold()
                )
            ):
                continue
            if filters and filters.allowed_subreddits:
                if (
                    example.subreddit is None
                    or example.subreddit.casefold()
                    not in {item.casefold() for item in filters.allowed_subreddits}
                ):
                    continue
            text = f"{example.title or ''}\n{example.body}"
            content_context = 1.0
            if example.content_type in compatible:
                content_context += 1.0
            if (
                request.subreddit
                and example.subreddit is not None
                and example.subreddit.casefold() == request.subreddit.casefold()
            ):
                content_context += 1.0
            topic_overlap = _overlap(topic_tokens, _tokens(text)) * 2.0
            target_overlap = _overlap(target_tokens, _tokens(text)) * 2.0
            recency = _recency_score(example, now, filters.useful_window_days if filters else 90)
            own = 1.0 if example.is_own_content else 0.0
            engagement = min(
                0.5,
                math.log1p(max(example.engagement_score, 0.0))
                / math.log1p(10_000.0)
                * 0.5,
            )
            tag_overlap = _tag_overlap(
                example,
                request.target_audience,
                request.tone,
            )
            components = {
                "content_community_context": min(content_context, 3.0),
                "topic_keyword_overlap": min(topic_overlap, 2.0),
                "target_text_similarity": min(target_overlap, 2.0),
                "recency": recency,
                "own_successful_content": own,
                "capped_engagement": engagement,
                "audience_tone_overlap": tag_overlap,
            }
            raw_score = sum(components.values())
            reason = ", ".join(
                name.replace("_", " ")
                for name, value in sorted(
                    components.items(), key=lambda item: (-item[1], item[0])
                )[:2]
                if value > 0
            ) or "compatible public example"
            scored.append((raw_score, example, components, reason))

        if not scored:
            raise NoRelevantExamplesFound
        preferred = set(filters.preferred_example_ids) if filters else set()
        scored.sort(
            key=lambda item: (-item[0], 0 if item[1].id in preferred else 1, str(item[1].id))
        )
        selected: list[SelectedExample] = []
        author_counts: dict[str, int] = {}
        source_counts: dict[str, int] = {}
        total_characters = 0
        max_examples = filters.maximum_context_examples if filters else request.candidate_count * 2
        max_characters = filters.maximum_example_characters if filters else 12000
        for raw_score, example, components, reason in scored:
            if len(selected) >= max_examples:
                break
            author = example.author_identifier or "unknown"
            if author_counts.get(author, 0) >= 2:
                continue
            source = example.source_url
            if source_counts.get(source, 0) >= 2:
                continue
            text_size = len(example.title or "") + len(example.body) + len(example.parent_text or "")
            if total_characters + text_size > max_characters:
                continue
            similarity = max(
                (_sequence_similarity(example, item.example) for item in selected),
                default=0.0,
            )
            final_score = max(0.0, min(10.0, raw_score - 0.75 * similarity))
            selected.append(
                SelectedExample(
                    example_id=example.id,
                    score=final_score,
                    component_scores={**components, "diversity_penalty": 0.75 * similarity},
                    selection_reason=reason,
                    example=example,
                )
            )
            author_counts[author] = author_counts.get(author, 0) + 1
            source_counts[source] = source_counts.get(source, 0) + 1
            total_characters += text_size
        return tuple(selected)


def _tokens(value: str) -> set[str]:
    return {
        token
        for token in _TOKEN.findall(normalize_authored_text(value).casefold())
        if token not in _STOPWORDS and not token.isnumeric()
    }


def _overlap(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _recency_score(example: ContentExample, now: datetime, useful_days: int) -> float:
    stamp = example.published_at or example.collected_at
    age = max(0.0, (now - stamp.astimezone(UTC)).total_seconds() / 86400.0)
    return max(0.0, min(1.0, 1.0 - age / useful_days))


def _tag_overlap(example: ContentExample, audience: str | None, tone: str | None) -> float:
    requested = _tokens(" ".join(item for item in (audience, tone) if item))
    tags = _tokens(" ".join(example.topic_tags))
    return min(0.5, _overlap(requested, tags) * 0.5)


def _sequence_similarity(left: ContentExample, right: ContentExample) -> float:
    left_text = normalize_authored_text(f"{left.title or ''}\n{left.body}").casefold()
    right_text = normalize_authored_text(f"{right.title or ''}\n{right.body}").casefold()
    return SequenceMatcher(None, left_text, right_text).ratio()


__all__ = ["COMPATIBLE_TYPES", "ExampleSelector", "NoRelevantExamplesFound"]
