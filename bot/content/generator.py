"""Content generation contract."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from bot.content.models import ContentRequest, GenerationResult


@runtime_checkable
class ContentGenerator(Protocol):
    """Asynchronous content generator interface."""

    async def generate(self, request: ContentRequest) -> GenerationResult:
        """Generate content for ``request``."""
        ...
