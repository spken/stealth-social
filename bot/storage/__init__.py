"""Persistence integration package."""

from bot.storage.content_models import (
    ContentExampleRecord,
    DiscoveredTopicRecord,
    ExampleCollectionRunRecord,
    GeneratedCandidateRecord,
    GenerationRequestRecord,
)
from bot.storage.content_repository import ContentRepository

__all__ = [
    "ContentExampleRecord",
    "ContentRepository",
    "DiscoveredTopicRecord",
    "ExampleCollectionRunRecord",
    "GeneratedCandidateRecord",
    "GenerationRequestRecord",
]
