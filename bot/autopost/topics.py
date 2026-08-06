"""Pure campaign-topic rotation helpers."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Sequence

from bot.content.models import StoredGenerationRequest


def select_campaign_topic(
    topics: Sequence[str],
    requests: Sequence[StoredGenerationRequest],
) -> str:
    """Choose the least-recently attempted configured topic."""

    if not topics:
        raise ValueError("topics cannot be empty")

    never = datetime.min.replace(tzinfo=UTC)
    last_used: dict[str, datetime] = {}
    current = {topic.strip().casefold() for topic in topics}
    for request in requests:
        raw_topic = request.request_snapshot.get("topic")
        if not isinstance(raw_topic, str):
            continue
        key = raw_topic.strip().casefold()
        if key not in current:
            continue
        previous = last_used.get(key)
        if previous is None or request.created_at > previous:
            last_used[key] = request.created_at
    return min(
        enumerate(topics),
        key=lambda item: (
            last_used.get(item[1].strip().casefold(), never),
            item[0],
        ),
    )[1]
