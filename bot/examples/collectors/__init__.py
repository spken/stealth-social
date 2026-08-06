"""Browser-visible, public-content collectors."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from bot.examples.models import (
    CollectedExample,
    ExampleCollectionRequest,
    TargetContext,
    TargetContextRequest,
)
from bot.examples.collectors.reddit import RedditExampleCollector
from bot.examples.collectors.x import XExampleCollector


@runtime_checkable
class ExampleCollector(Protocol):
    async def collect(self, request: ExampleCollectionRequest) -> list[CollectedExample]:
        raise NotImplementedError


@runtime_checkable
class TargetContextResolver(Protocol):
    async def resolve_target(self, request: TargetContextRequest) -> TargetContext:
        raise NotImplementedError


__all__ = [
    "ExampleCollector",
    "RedditExampleCollector",
    "TargetContextResolver",
    "XExampleCollector",
]
