"""Deterministic content generator for local development."""

from __future__ import annotations

from bot.models import ContentRequest, GeneratedContent


class MockContentGenerator:
    """Return configured or request-provided content without external services."""

    def __init__(self, content: str | None = None, title: str | None = None) -> None:
        if content is not None and not content.strip():
            raise ValueError("mock content must not be blank")
        if title is not None and not title.strip():
            raise ValueError("mock title must not be blank")
        self._content = content
        self._title = title

    async def generate(self, request: ContentRequest) -> GeneratedContent:
        """Resolve content deterministically, preferring explicitly supplied text."""
        content = self._content or request.content or request.prompt
        if content is None:
            raise ValueError("content request did not provide content or a prompt")
        title = self._title if self._title is not None else request.title
        metadata = dict(request.metadata)
        metadata["generator"] = "mock"
        return GeneratedContent(content=content, title=title, metadata=metadata)
