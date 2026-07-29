"""Shared protocol and failures for social platform adapters."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from bot.models import ActionResult, SocialAction


@runtime_checkable
class SocialPlatform(Protocol):
    """Adapter capable of executing a validated social action."""

    async def execute(self, action: SocialAction) -> ActionResult:
        """Execute ``action`` and return its platform result."""
        ...


class PlatformError(RuntimeError):
    """Base platform failure with an explicit retry decision."""

    retryable = False

    def __init__(self, message: str, *, retryable: bool | None = None) -> None:
        super().__init__(message)
        self.retryable = type(self).retryable if retryable is None else retryable


class PlatformAuthenticationError(PlatformError):
    """An account session is missing, expired, or unauthorized."""


class PlatformActionRejectedError(PlatformError):
    """The platform rejected an action that should not be retried unchanged."""


class PlatformRateLimitError(PlatformError):
    """The platform refused an action because a rate limit was reached."""

    retryable = True

    def __init__(
        self,
        message: str,
        *,
        retry_after_seconds: float | None = None,
    ) -> None:
        if retry_after_seconds is not None and retry_after_seconds < 0:
            raise ValueError("retry_after_seconds must be non-negative")
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


class PlatformUnavailableError(PlatformError):
    """A transient platform or browser failure prevented execution."""

    retryable = True
