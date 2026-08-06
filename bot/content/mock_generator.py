"""Deterministic content generator for local development."""

from __future__ import annotations

from bot.content.models import CandidateDraft, ContentRequest, GenerationResult


class MockContentGenerator:
    """Return configured or request-provided content without external services."""

    def __init__(self, content: str | None = None, title: str | None = None) -> None:
        if content is not None and not content.strip():
            raise ValueError("mock content must not be blank")
        if title is not None and not title.strip():
            raise ValueError("mock title must not be blank")
        self._content = content
        self._title = title

    async def generate(self, request: ContentRequest) -> GenerationResult:
        """Return a deterministic batch without external services."""
        count = request.candidate_count
        base = self._content or request.topic or request.goal or "a useful idea"
        body = f"A practical note about {base}, with a clear takeaway for the community."
        strategies = request.strategy_names
        if len(strategies) != count:
            strategies = tuple(
                f"mock strategy {index}" for index in range(1, count + 1)
            )
        title = self._title
        if title is None and request.generation_type.value == "reddit_post":
            title = f"A practical note about {base}"
        drafts = tuple(
            CandidateDraft(
                title=title,
                body=body if count == 1 else f"{body} ({index})",
                strategy=strategies[index - 1],
                used_example_ids=tuple(
                    item.example_id for item in request.selected_examples
                ),
            )
            for index in range(1, count + 1)
        )
        return GenerationResult(
            candidates=drafts,
            model_name="mock",
            resolved_parameters=dict(request.resolved_parameters),
            latency_seconds=0.0,
            prompt_tokens=0,
            completion_tokens=0,
            metadata={"generator": "mock"},
        )
