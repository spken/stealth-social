"""Social platform adapter package."""

from bot.platforms.reddit import (
    BrowserInteractionError,
    RedditAdapter,
    RedditChallengeError,
    RedditTargetNotFoundError,
    SubredditNotAllowedError,
    SubredditUnavailableError,
)
from bot.platforms.x import XAdapter, XChallengeError, XTargetNotFoundError

__all__ = [
    "BrowserInteractionError",
    "RedditAdapter",
    "RedditChallengeError",
    "RedditTargetNotFoundError",
    "SubredditNotAllowedError",
    "SubredditUnavailableError",
    "XAdapter",
    "XChallengeError",
    "XTargetNotFoundError",
]
