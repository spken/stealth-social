"""Browser integration package."""
from bot.browser.sessions import (
    BrowserSession,
    FailureCapture,
    SessionStatus,
    classify_auth_state,
)
from bot.browser.manager import (
    BrowserAccountNotConfiguredError,
    BrowserManager,
    BrowserManagerClosedError,
    BrowserManagerError,
    InvalidSessionProfileError,
)

__all__ = [
    "BrowserAccountNotConfiguredError",
    "BrowserManager",
    "BrowserManagerClosedError",
    "BrowserManagerError",
    "BrowserSession",
    "FailureCapture",
    "InvalidSessionProfileError",
    "SessionStatus",
    "classify_auth_state",
]
