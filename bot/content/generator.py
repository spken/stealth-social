"""Content generation contract."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from bot.models import ContentRequest, GeneratedContent


@runtime_checkable
class ContentGenerator(Protocol):
    """Asynchronous content generator interface."""

    async def generate(self, request: ContentRequest) -> GeneratedContent:
        """Generate content for ``request``."""
        ...


# TODO: Integrate locally hosted LLM content generation.
