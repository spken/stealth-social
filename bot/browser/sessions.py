"""In-memory browser session and authentication state types."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from urllib.parse import urlsplit

from playwright.async_api import BrowserContext, Error as PlaywrightError, Page

from bot.models import Platform


class SessionStatus(StrEnum):
    """Observable state of a live browser session."""

    UNKNOWN = "unknown"
    AUTHENTICATED = "authenticated"
    AUTH_REQUIRED = "auth_required"
    CHALLENGE_REQUIRED = "challenge_required"
    CLOSED = "closed"


@dataclass(frozen=True, slots=True)
class BrowserSession:
    """One serialized lease of an in-memory persistent browser context."""

    platform: Platform
    account_name: str
    profile_name: str
    profile_directory: Path
    context: BrowserContext
    page: Page
    status: SessionStatus


@dataclass(frozen=True, slots=True)
class FailureCapture:
    """Best-effort diagnostics collected after an action failure."""

    path: Path | None
    page_url: str | None


@dataclass(frozen=True, slots=True)
class _AuthSignals:
    hosts: tuple[str, ...]
    auth_paths: tuple[str, ...]
    challenge_path_prefixes: tuple[str, ...]
    login_selector: str
    challenge_selector: str


_MAX_SIGNAL_CONTROLS = 20

_X_LOGIN_SELECTOR = (
    'input[autocomplete="username"], '
    '[data-testid="LoginForm_Login_Button"], '
    '[data-testid="loginButton"]'
)
_X_CHALLENGE_SELECTOR = (
    'iframe[src*="captcha"], iframe[src*="challenge"], '
    '[data-testid*="challenge"], input[name="challenge_response"], '
    'form[action*="/account/access"], form[action*="/verify"]'
)
_REDDIT_LOGIN_SELECTOR = (
    "form#login_login-main, "
    'form.login-form input[name="user"], '
    'form[action*="/login"] input[name="password"]'
)
_REDDIT_CHALLENGE_SELECTOR = (
    'iframe[src*="captcha"], iframe[src*="challenge"], '
    ".g-recaptcha, [data-sitekey], "
    'form[action*="/challenge"], form[action*="/verify"]'
)

_X_AUTH_SIGNALS = _AuthSignals(
    hosts=("x.com", "twitter.com"),
    auth_paths=("/login", "/i/flow/login"),
    challenge_path_prefixes=(
        "/account/access",
        "/i/flow/challenge",
        "/i/flow/consent",
        "/i/flow/email_verification",
        "/i/flow/phone_verification",
        "/i/flow/two-factor-authentication",
        "/i/flow/verify",
    ),
    login_selector=_X_LOGIN_SELECTOR,
    challenge_selector=_X_CHALLENGE_SELECTOR,
)

_REDDIT_AUTH_SIGNALS = _AuthSignals(
    hosts=("reddit.com",),
    auth_paths=("/account/login", "/login"),
    challenge_path_prefixes=(
        "/challenge",
        "/checkpoint",
        "/verify",
    ),
    login_selector=_REDDIT_LOGIN_SELECTOR,
    challenge_selector=_REDDIT_CHALLENGE_SELECTOR,
)

_AUTH_SIGNALS = {
    Platform.X: _X_AUTH_SIGNALS,
    Platform.REDDIT: _REDDIT_AUTH_SIGNALS,
}


def _matches_host(url: str, expected_hosts: tuple[str, ...]) -> bool:
    try:
        parsed = urlsplit(url)
        host = (parsed.hostname or "").casefold()
    except (AttributeError, ValueError):
        return False
    if parsed.scheme.casefold() not in {"http", "https"}:
        return False
    return any(
        host == expected or host.endswith(f".{expected}")
        for expected in expected_hosts
    )


def _path_has_prefix(path: str, prefixes: tuple[str, ...]) -> bool:
    return any(
        path == prefix or path.startswith(f"{prefix}/")
        for prefix in prefixes
    )


def _classify_url(url: str, signals: _AuthSignals) -> SessionStatus:
    if not _matches_host(url, signals.hosts):
        return SessionStatus.UNKNOWN

    try:
        path = urlsplit(url).path.casefold()
    except (AttributeError, ValueError):
        return SessionStatus.UNKNOWN
    normalized_path = (path or "/").rstrip("/") or "/"
    if _path_has_prefix(
        normalized_path,
        signals.challenge_path_prefixes,
    ):
        return SessionStatus.CHALLENGE_REQUIRED
    if normalized_path in signals.auth_paths:
        return SessionStatus.AUTH_REQUIRED
    return SessionStatus.UNKNOWN


async def _has_visible(page: Page, selector: str) -> bool:
    locator = page.locator(selector)
    for index in range(min(await locator.count(), _MAX_SIGNAL_CONTROLS)):
        if await locator.nth(index).is_visible():
            return True
    return False


async def classify_auth_state(
    page: Page,
    platform: Platform | str,
) -> SessionStatus:
    """Classify platform-owned authentication and challenge surfaces.

    URL signals are matched against parsed, platform-owned paths only. DOM
    signals are limited to visible platform controls, so page text, user
    content, query parameters, and fragments cannot determine auth state.
    """

    try:
        resolved_platform = Platform(platform)
    except (TypeError, ValueError):
        return SessionStatus.UNKNOWN

    if page.is_closed():
        return SessionStatus.CLOSED

    signals = _AUTH_SIGNALS[resolved_platform]
    page_url = page.url
    url_status = _classify_url(page_url, signals)
    if url_status is not SessionStatus.UNKNOWN:
        return url_status
    if not _matches_host(page_url, signals.hosts):
        return SessionStatus.UNKNOWN

    try:
        challenge_visible = await _has_visible(
            page,
            signals.challenge_selector,
        )
        login_visible = await _has_visible(page, signals.login_selector)
    except PlaywrightError:
        return (
            SessionStatus.CLOSED
            if page.is_closed()
            else SessionStatus.UNKNOWN
        )

    if page.is_closed():
        return SessionStatus.CLOSED
    if not _matches_host(page.url, signals.hosts):
        return SessionStatus.UNKNOWN
    if challenge_visible:
        return SessionStatus.CHALLENGE_REQUIRED
    if login_visible:
        return SessionStatus.AUTH_REQUIRED
    return SessionStatus.AUTHENTICATED
