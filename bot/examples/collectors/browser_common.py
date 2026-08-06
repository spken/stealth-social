"""Shared safety boundaries for browser-only public collection."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable, Mapping
from typing import Any
from urllib.parse import unquote, urlsplit, urlunsplit

from playwright.async_api import Locator, Page, Response

from bot.examples.models import (
    ExampleChallengeError,
    ExampleCollectionError,
    ExampleRateLimitedError,
)
from bot.examples.normalization import normalize_authored_text
from bot.models import Platform

_X_HOSTS = frozenset({"x.com", "www.x.com", "twitter.com", "www.twitter.com"})
_REDDIT_HOSTS = frozenset({"reddit.com", "www.reddit.com", "old.reddit.com"})
_X_STATUS = re.compile(r"^/(?P<user>[A-Za-z0-9_]{1,15})/status/(?P<id>[0-9]+)$")
_REDDIT_PERMALINK = re.compile(
    r"^/r/(?P<subreddit>[A-Za-z0-9_]{2,21})/comments/(?P<post>[A-Za-z0-9]+)/"
    r"(?P<slug>[^/]+)(?:/(?P<comment>[A-Za-z0-9]+))?/?$",
    re.IGNORECASE,
)
_RETRY_AFTER = re.compile(r"^[0-9]+(?:\.[0-9]+)?$")


def validate_public_url(value: str, platform: Platform, *, target_kind: str | None = None) -> str:
    """Validate and canonicalize an exact platform-owned public URL."""

    try:
        parsed = urlsplit(value.strip())
        port = parsed.port
    except (AttributeError, ValueError) as error:
        raise ValueError("public URL is malformed") from error
    host = (parsed.hostname or "").casefold()
    allowed = _X_HOSTS if platform is Platform.X else _REDDIT_HOSTS
    if parsed.scheme.casefold() != "https" or host not in allowed:
        raise ValueError("public URL must use HTTPS on an approved platform host")
    if parsed.username or parsed.password:
        raise ValueError("public URL must not contain credentials")
    if port not in {None, 443}:
        raise ValueError("public URL must use the default HTTPS port")
    if parsed.fragment:
        raise ValueError("public URL must not contain a fragment")
    path = unquote(parsed.path).rstrip("/")
    if platform is Platform.X:
        match = _X_STATUS.fullmatch(path)
        if match is None:
            raise ValueError("X public URL must identify a numeric status")
        return f"https://x.com/{match.group('user')}/status/{match.group('id')}"
    match = _REDDIT_PERMALINK.fullmatch(path)
    if match is None:
        raise ValueError("Reddit public URL must be a post or comment permalink")
    if target_kind == "post" and match.group("comment") is not None:
        raise ValueError("Reddit post URL must not identify a comment")
    if target_kind == "comment" and match.group("comment") is None:
        raise ValueError("Reddit comment URL must identify a comment")
    suffix = f"/{match.group('comment')}" if match.group("comment") else ""
    return (
        f"https://old.reddit.com/r/{match.group('subreddit')}/comments/"
        f"{match.group('post')}/{match.group('slug')}{suffix}/"
    )


async def navigate_public_page(page: Page, url: str, platform: Platform) -> Response | None:
    """Navigate once and apply the common stop policy immediately."""

    response = await page.goto(url, wait_until="domcontentloaded")
    await raise_for_collection_page(page, response, platform)
    return response


async def raise_for_collection_page(
    page: Page,
    response: Response | None = None,
    platform: Platform | None = None,
) -> None:
    """Stop collection on status, challenge, login-wall, or unavailable signals."""

    status = response.status if response is not None else None
    retry_after = _retry_after(response.headers if response is not None else {})
    safe_location = _safe_location(page.url)
    if status == 429:
        raise ExampleRateLimitedError(
            f"public collection was rate limited at {safe_location}",
            retry_after_seconds=retry_after,
        )
    if status in {401, 403}:
        raise ExampleChallengeError(
            f"public collection access was denied at {safe_location}",
            retry_after_seconds=retry_after,
        )
    if status == 404 or (status is not None and status >= 500):
        raise ExampleCollectionError(f"public source unavailable at {safe_location}")

    platform = platform or Platform.X
    if platform is Platform.X:
        challenge_selector = (
            'iframe[src*="captcha"], iframe[src*="challenge"], '
            '[data-testid*="challenge"], [data-testid="loginButton"], '
            'input[autocomplete="username"]'
        )
        unavailable_selector = '[data-testid="empty_state_header"], [data-testid="error-detail"]'
    else:
        challenge_selector = (
            'iframe[src*="captcha"], iframe[src*="challenge"], '
            '.g-recaptcha, form[action*="/login"], form[action*="/challenge"]'
        )
        unavailable_selector = '.error-page, .listing-page .infobar, .thing.deleted'
    if await _has_visible(page, challenge_selector):
        raise ExampleChallengeError(
            f"public collection stopped at a challenge or login wall at {safe_location}"
        )
    if await _has_visible(page, unavailable_selector):
        raise ExampleCollectionError(f"public target is unavailable at {safe_location}")


async def bounded_scroll(
    page: Page,
    *,
    item_ids: Callable[[], Awaitable[set[str]]],
    maximum_items: int,
    maximum_scrolls: int = 8,
    stop_predicate: Callable[[], Awaitable[bool]] | None = None,
) -> None:
    """Scroll a finite number of times until IDs stop changing."""

    if maximum_items < 1 or maximum_scrolls < 0:
        raise ValueError("bounded scroll limits must be positive")
    previous: set[str] = set()
    unchanged = 0
    for _ in range(maximum_scrolls + 1):
        current = await item_ids()
        if len(current) >= maximum_items:
            return
        if stop_predicate is not None and await stop_predicate():
            return
        if current == previous:
            unchanged += 1
            if unchanged >= 2:
                return
        else:
            unchanged = 0
        previous = current
        await page.mouse.wheel(0, 1200)
        await asyncio.sleep(0.75)


async def safe_text(locator: Locator, *, maximum_length: int = 20000) -> str:
    try:
        value = await locator.inner_text(timeout=3000)
    except Exception:
        return ""
    return normalize_authored_text(value)[:maximum_length]


async def safe_attribute(
    locator: Locator,
    name: str,
    *,
    maximum_length: int = 2000,
) -> str | None:
    try:
        value = await locator.get_attribute(name, timeout=3000)
    except Exception:
        return None
    if value is None:
        return None
    return value[:maximum_length]


async def _has_visible(page: Page, selector: str) -> bool:
    locator = page.locator(selector)
    for index in range(min(await locator.count(), 30)):
        try:
            if await locator.nth(index).is_visible():
                return True
        except Exception:
            continue
    return False


def _retry_after(headers: Mapping[str, str]) -> float | None:
    value = headers.get("retry-after")
    if value is None or not _RETRY_AFTER.fullmatch(value.strip()):
        return None
    return min(float(value), 86400.0)


def _safe_location(value: str) -> str:
    try:
        parsed = urlsplit(value)
    except ValueError:
        return "unknown-host"
    return urlunsplit((parsed.scheme.casefold(), parsed.hostname or "", parsed.path or "/", "", ""))[:500]


__all__ = [
    "bounded_scroll",
    "navigate_public_page",
    "raise_for_collection_page",
    "safe_attribute",
    "safe_text",
    "validate_public_url",
]
