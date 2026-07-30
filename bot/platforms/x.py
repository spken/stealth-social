"""Browser-backed X publishing through the first-party web interface."""

from __future__ import annotations

import re
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal
from urllib.parse import urljoin, urlsplit, urlunsplit
from uuid import UUID, uuid4

import structlog
from playwright.async_api import (
    Error as PlaywrightError,
    Locator,
    Page,
    Response,
    TimeoutError as PlaywrightTimeoutError,
)

from bot.browser.manager import (
    BrowserAccountNotConfiguredError,
    BrowserManager,
    BrowserManagerError,
)
from bot.browser.sessions import FailureCapture
from bot.config import Settings
from bot.models import ActionResult, ActionType, Platform, SocialAction
from bot.platforms.base import (
    PlatformActionRejectedError,
    PlatformAuthenticationError,
    PlatformError,
    PlatformRateLimitError,
    PlatformUnavailableError,
)
from bot.platforms.reddit import BrowserInteractionError

logger = structlog.get_logger(__name__)

_X_ORIGIN = "https://x.com"
_COMPOSE_URL = f"{_X_ORIGIN}/compose/post"
_X_TARGET_HOSTS = frozenset(
    {"x.com", "www.x.com", "twitter.com", "www.twitter.com"}
)
_X_CREATE_TWEET_HOSTS = _X_TARGET_HOSTS | frozenset(
    {"api.x.com", "api.twitter.com"}
)
_MAX_POST_LENGTH = 280
_NAVIGATION_TIMEOUT_MS = 30_000
_INTERACTION_TIMEOUT_MS = 15_000
_CREATED_CONTENT_TIMEOUT_MS = 8_000

_STATUS_PATH_RE = re.compile(
    r"^/(?:i/web|[a-z0-9_]{1,15})/status/"
    r"(?P<post_id>[1-9][0-9]{0,29})/?$",
    re.IGNORECASE,
)
_POST_ID_RE = re.compile(r"^[1-9][0-9]{0,29}$")

_PLATFORM_UI_ROOT_SELECTOR = "#react-root"
_REPLY_DIALOG_SCOPE_SELECTORS = (
    '[role="dialog"]',
    '[data-testid="sheetDialog"]',
)
_BASELINE_COMPOSER_MARKER_ATTRIBUTE = (
    "data-social-bot-existing-x-composer"
)
_BASELINE_REPLY_SCOPE_MARKER_ATTRIBUTE = (
    "data-social-bot-existing-x-reply-scope"
)
_COMPOSER_DIALOG_ANCESTOR_SELECTOR = (
    "xpath=ancestor::*"
    "[@role='dialog' or @data-testid='sheetDialog'][1]"
)
_FORM_ANCESTOR_SELECTOR = "xpath=ancestor::form[1]"

# Composer selectors. The semantic locator is a fallback for X deployments
# where the stable test id is temporarily absent.
_COMPOSER_SELECTORS = (
    '[data-testid="tweetTextarea_0"]',
    '[role="textbox"][contenteditable="true"]',
)
_SUBMIT_SELECTORS = (
    '[data-testid="tweetButton"]',
    '[data-testid="tweetButtonInline"]',
)
_REPLY_TRIGGER_SELECTOR = '[data-testid="reply"]'
_TWEET_ARTICLE_SELECTOR = 'article[data-testid="tweet"]'
_STATUS_ANCHOR_SELECTOR = 'a[href*="/status/"]'
_CREATED_STATUS_ANCHOR_SELECTOR = (
    '[data-testid="toast"] a[href*="/status/"], '
    '[role="alert"] a[href*="/status/"]'
)
_TARGET_ARTICLE_SELECTOR_TEMPLATE = (
    _TWEET_ARTICLE_SELECTOR
    + ':has(a[href$="/status/{post_id}"]), '
    + _TWEET_ARTICLE_SELECTOR
    + ':has(a[href*="/status/{post_id}?"])'
)
_TARGET_REPLY_SCOPE_SELECTOR_TEMPLATE = (
    '[data-testid="cellInnerDiv"]:has('
    + _TARGET_ARTICLE_SELECTOR_TEMPLATE
    + "), "
    + _TARGET_ARTICLE_SELECTOR_TEMPLATE
)
_LOGIN_CONTROL_SELECTORS = (
    'input[autocomplete="username"]',
    '[data-testid="LoginForm_Login_Button"]',
    '[data-testid="loginButton"]',
)
_LOGIN_SELECTOR = ", ".join(
    f"{_PLATFORM_UI_ROOT_SELECTOR} {selector}"
    for selector in _LOGIN_CONTROL_SELECTORS
)
_CHALLENGE_CONTROL_SELECTORS = (
    'iframe[src*="captcha"]',
    'iframe[src*="challenge"]',
    '[data-testid*="challenge"]',
    'input[name="challenge_response"]',
    'form[action*="/account/access"]',
    'form[action*="/verify"]',
)
_CHALLENGE_SELECTOR = ", ".join(
    f"{_PLATFORM_UI_ROOT_SELECTOR} {selector}"
    for selector in _CHALLENGE_CONTROL_SELECTORS
)
_STATE_MESSAGE_CONTAINER_SELECTORS = (
    '[data-testid="error-detail"]',
    '[data-testid="emptyState"]',
    '[role="alert"]',
    '[data-testid="toast"]',
)
_STATE_MESSAGE_SELECTOR = ", ".join(
    f"{_PLATFORM_UI_ROOT_SELECTOR} {selector}"
    for selector in _STATE_MESSAGE_CONTAINER_SELECTORS
)

_AUTH_PATH_PREFIXES = ("/login", "/i/flow/login")
_CHALLENGE_PATH_PREFIXES = (
    "/account/access",
    "/i/flow/challenge",
    "/i/flow/consent",
    "/i/flow/email_verification",
    "/i/flow/two-factor-authentication",
)
_AUTHENTICATION_SIGNALS = (
    "log in to x",
    "sign in to x",
    "you must log in",
    "session has expired",
    "could not authenticate",
    "authentication required",
)
_CHALLENGE_SIGNALS = (
    "verify your identity",
    "verify your account",
    "additional verification",
    "security check",
    "account access",
    "unusual activity",
    "complete the captcha",
    "prove you're human",
    "prove you are human",
)
_RATE_LIMIT_SIGNALS = (
    "rate limit exceeded",
    "too many requests",
    "daily limit",
    "try again later",
)
_NOT_FOUND_SIGNALS = (
    "this page doesn’t exist",
    "this page doesn't exist",
    "post not found",
    "this post is unavailable",
    "nothing to see here",
)
_TRANSIENT_SIGNALS = (
    "something went wrong",
    "try reloading",
    "temporarily unavailable",
    "service unavailable",
)

_AUTHENTICATION_CODES = frozenset(
    {
        "32",
        "89",
        "99",
        "135",
        "215",
        "220",
        "AUTHENTICATION_ERROR",
        "UNAUTHENTICATED",
        "UNAUTHORIZED",
    }
)
_CHALLENGE_CODES = frozenset(
    {
        "226",
        "326",
        "ACCOUNT_LOCKED",
        "CHALLENGE_REQUIRED",
        "VERIFICATION_REQUIRED",
    }
)
_RATE_LIMIT_CODES = frozenset(
    {"88", "185", "RATE_LIMIT", "RATE_LIMITED", "TOO_MANY_REQUESTS"}
)
_TARGET_NOT_FOUND_CODES = frozenset(
    {"34", "144", "NOT_FOUND", "STATUS_NOT_FOUND", "TWEET_NOT_FOUND"}
)
_ACTION_REJECTED_CODES = frozenset(
    {
        "170",
        "186",
        "187",
        "324",
        "385",
        "DUPLICATE_STATUS",
        "INVALID_CONTENT",
        "REPLY_RESTRICTED",
    }
)
_TRANSIENT_CODES = frozenset(
    {"130", "131", "INTERNAL_ERROR", "SERVICE_UNAVAILABLE"}
)
_REPLY_PARENT_ID_FIELDS = (
    "in_reply_to_status_id_str",
    "in_reply_to_status_id",
    "in_reply_to_tweet_id",
    "in_reply_to_tweet_id_str",
    "inReplyToStatusId",
    "inReplyToStatusIdStr",
    "inReplyToTweetId",
)

_PagePurpose = Literal["compose", "target"]


class XTargetNotFoundError(PlatformActionRejectedError):
    """The requested X post could not be resolved."""


class XChallengeError(PlatformAuthenticationError):
    """X requires a human challenge or account verification."""


@dataclass(frozen=True, slots=True)
class _ReplyTarget:
    post_id: str
    navigation_url: str


@dataclass(frozen=True, slots=True)
class _ReplyComposer:
    composer: Locator
    scope: Locator


@dataclass(frozen=True, slots=True)
class _PublishedPost:
    post_id: str
    url: str
    response_status: int | None


class XAdapter:
    """Publish X posts for one configured, browser-backed account."""

    def __init__(
        self,
        browser_manager: BrowserManager,
        settings: Settings,
        account_name: str,
    ) -> None:
        normalized_account_name = account_name.strip()
        if not normalized_account_name:
            raise PlatformActionRejectedError("X account name is required")

        account = settings.accounts.x.get(normalized_account_name)
        if account is None:
            raise PlatformActionRejectedError(
                f"X account '{normalized_account_name}' is not configured"
            )
        if not account.enabled:
            raise PlatformActionRejectedError(
                f"X account '{normalized_account_name}' is disabled"
            )

        try:
            browser_manager.resolve_profile(Platform.X, normalized_account_name)
        except BrowserAccountNotConfiguredError as error:
            raise PlatformActionRejectedError(
                f"X account '{normalized_account_name}' has no browser profile"
            ) from error

        self._browser_manager = browser_manager
        self._account_name = normalized_account_name
        self._log = logger.bind(
            platform=Platform.X.value,
            account_name=normalized_account_name,
        )

    @property
    def account_name(self) -> str:
        """Return the configured account owned by this adapter."""

        return self._account_name

    async def execute(self, action: SocialAction) -> ActionResult:
        """Execute a validated X post action."""

        if action.platform is not Platform.X:
            raise PlatformActionRejectedError(
                "XAdapter only accepts actions for platform 'x'"
            )
        if action.action_type is not ActionType.X_POST:
            raise PlatformActionRejectedError(
                f"Unsupported X action type '{action.action_type.value}'"
            )
        if action.account_name != self._account_name:
            raise PlatformActionRejectedError(
                "X action account does not match this adapter"
            )
        if (
            action.title is not None
            or action.subreddit is not None
            or action.parent_comment_id is not None
        ):
            raise PlatformActionRejectedError(
                "X post contains fields that are not supported"
            )

        content = _validate_post_content(action.content)
        reply_target = _resolve_reply_target(
            target_url=action.target_url,
            parent_post_id=action.parent_post_id,
        )
        return await self._publish_post(
            content,
            reply_target=reply_target,
            action_id=action.id,
        )

    async def publish_post(
        self,
        content: str,
        reply_to: str | None = None,
        *,
        action_id: UUID | None = None,
    ) -> ActionResult:
        """Publish a text post, optionally replying to an X status URL or id."""

        normalized_content = _validate_post_content(content)
        reply_target = _resolve_direct_reply_target(reply_to)
        return await self._publish_post(
            normalized_content,
            reply_target=reply_target,
            action_id=action_id or uuid4(),
        )

    # TODO: Add X media upload support.

    async def _publish_post(
        self,
        content: str,
        *,
        reply_target: _ReplyTarget | None,
        action_id: UUID,
    ) -> ActionResult:
        async def operation(page: Page) -> ActionResult:
            purpose: _PagePurpose = "target" if reply_target is not None else "compose"
            navigation_url = (
                reply_target.navigation_url
                if reply_target is not None
                else _COMPOSE_URL
            )
            await self._navigate(page, navigation_url, purpose=purpose)

            required_submit_scope: Locator | None = None
            if reply_target is not None:
                reply_composer = await self._open_reply_composer(
                    page,
                    reply_target,
                )
                composer = reply_composer.composer
                required_submit_scope = reply_composer.scope
            else:
                composer = await self._wait_for_composer(page, purpose=purpose)

            await self._fill_composer(composer, content)
            submit = await self._submit_locator(
                page,
                composer,
                purpose=purpose,
                required_scope=required_submit_scope,
            )
            published = await self._submit_post(
                page,
                submit,
                reply_target=reply_target,
                purpose=purpose,
            )

            metadata: dict[str, Any] = {
                "is_reply": reply_target is not None,
            }
            if reply_target is not None:
                metadata["reply_to_post_id"] = reply_target.post_id
            if published.response_status is not None:
                metadata["response_status"] = published.response_status

            return ActionResult(
                action_id=action_id,
                success=True,
                external_content_id=published.post_id,
                external_content_url=published.url,
                message=(
                    "X reply published"
                    if reply_target is not None
                    else "X post published"
                ),
                metadata=metadata,
            )

        return await self._run_operation(action_id, operation)

    async def _run_operation(
        self,
        action_id: UUID,
        operation: Callable[[Page], Awaitable[ActionResult]],
    ) -> ActionResult:
        self._log.info(
            "x_action_started",
            action_id=str(action_id),
            action_type=ActionType.X_POST.value,
        )
        try:
            async with self._browser_manager.get_page(
                Platform.X,
                self._account_name,
            ) as page:
                try:
                    result = await operation(page)
                except BrowserInteractionError as error:
                    capture = await self._capture_failure(page, action_id)
                    error.attach_capture(capture)
                    raise
                except PlatformError:
                    await self._capture_failure(page, action_id)
                    raise
                except PlaywrightError as error:
                    capture = await self._capture_failure(page, action_id)
                    raise BrowserInteractionError(
                        "X browser interaction failed",
                        current_url=_safe_page_url(
                            capture.page_url if capture is not None else page.url
                        ),
                        screenshot_path=(
                            capture.path if capture is not None else None
                        ),
                    ) from error
        except PlatformError:
            raise
        except BrowserManagerError as error:
            raise BrowserInteractionError(
                "X browser session could not be acquired"
            ) from error

        self._log.info(
            "x_action_succeeded",
            action_id=str(action_id),
            action_type=ActionType.X_POST.value,
            external_content_id=result.external_content_id,
        )
        return result

    async def _capture_failure(
        self,
        page: Page,
        action_id: UUID,
    ) -> FailureCapture | None:
        try:
            return await self._browser_manager.capture_failure(page, action_id)
        except Exception as error:
            self._log.warning(
                "x_failure_capture_failed",
                action_id=str(action_id),
                error_type=type(error).__name__,
            )
            return None

    async def _navigate(
        self,
        page: Page,
        url: str,
        *,
        purpose: _PagePurpose,
    ) -> Response | None:
        response = await page.goto(
            url,
            wait_until="domcontentloaded",
            timeout=_NAVIGATION_TIMEOUT_MS,
        )
        await self._raise_for_page(page, response, purpose=purpose)
        return response

    async def _raise_for_page(
        self,
        page: Page,
        response: Response | None,
        *,
        purpose: _PagePurpose,
    ) -> None:
        parsed_url = _split_safe_url(page.url)
        if (
            parsed_url is None
            or parsed_url.hostname is None
            or parsed_url.hostname.casefold() not in _X_TARGET_HOSTS
        ):
            raise PlatformAuthenticationError(
                "X redirected the browser to an unexpected origin"
            )

        path = parsed_url.path.casefold()
        state_text = await _visible_state_text(page)

        if (
            _path_has_prefix(path, _CHALLENGE_PATH_PREFIXES)
            or await _has_visible(page, _CHALLENGE_SELECTOR)
            or _contains_signal(state_text, _CHALLENGE_SIGNALS)
        ):
            raise XChallengeError(
                "X requires a human challenge or account verification"
            )
        if (
            _path_has_prefix(path, _AUTH_PATH_PREFIXES)
            or await _has_visible(page, _LOGIN_SELECTOR)
            or _contains_signal(state_text, _AUTHENTICATION_SIGNALS)
        ):
            raise PlatformAuthenticationError(
                "X account session is not authenticated"
            )
        if _contains_signal(state_text, _RATE_LIMIT_SIGNALS) or (
            response is not None and response.status == 429
        ):
            raise PlatformRateLimitError(
                "X rate limit reached",
                retry_after_seconds=_retry_after_seconds(
                    response.headers if response is not None else {}
                ),
            )
        if purpose == "target" and (
            _contains_signal(state_text, _NOT_FOUND_SIGNALS)
            or (response is not None and response.status == 404)
        ):
            raise XTargetNotFoundError("X reply target was not found")
        if _contains_signal(state_text, _TRANSIENT_SIGNALS) or (
            response is not None and response.status >= 500
        ):
            raise PlatformUnavailableError("X is temporarily unavailable")

        if response is None:
            return
        if response.status in {401, 407}:
            raise PlatformAuthenticationError(
                "X account session is not authenticated"
            )
        if response.status == 403:
            raise PlatformActionRejectedError(
                "X rejected access to the requested page"
            )
        if response.status == 404:
            raise PlatformUnavailableError("X publishing page was not found")
        if response.status >= 400:
            raise PlatformActionRejectedError(
                "X rejected access to the requested page"
            )

    async def _open_reply_composer(
        self,
        page: Page,
        target: _ReplyTarget,
    ) -> _ReplyComposer:
        current_target = _try_status_url(page.url)
        if (
            current_target is not None
            and current_target.post_id != target.post_id
        ):
            raise XTargetNotFoundError(
                "X redirected to a different reply target"
            )

        target_article = page.locator(
            _TARGET_ARTICLE_SELECTOR_TEMPLATE.format(post_id=target.post_id)
        ).first
        try:
            await target_article.wait_for(
                state="visible",
                timeout=_INTERACTION_TIMEOUT_MS,
            )
        except PlaywrightTimeoutError as error:
            await self._raise_for_page(
                page,
                None,
                purpose="target",
            )
            raise XTargetNotFoundError("X reply target was not found") from error

        if not await _article_contains_status(target_article, target.post_id):
            raise XTargetNotFoundError("X reply target was not found")

        reply_trigger = target_article.locator(_REPLY_TRIGGER_SELECTOR).first
        try:
            await reply_trigger.wait_for(
                state="visible",
                timeout=_INTERACTION_TIMEOUT_MS,
            )
        except PlaywrightTimeoutError as error:
            await self._raise_for_page(
                page,
                None,
                purpose="target",
            )
            raise PlatformActionRejectedError(
                "X reply target does not accept replies"
            ) from error
        if not await reply_trigger.is_enabled():
            raise PlatformActionRejectedError(
                "X reply target does not accept replies"
            )

        baseline_marker = uuid4().hex
        await _mark_visible_elements(
            page,
            _COMPOSER_SELECTORS,
            attribute=_BASELINE_COMPOSER_MARKER_ATTRIBUTE,
            marker=baseline_marker,
        )
        await _mark_visible_elements(
            page,
            _REPLY_DIALOG_SCOPE_SELECTORS,
            attribute=_BASELINE_REPLY_SCOPE_MARKER_ATTRIBUTE,
            marker=baseline_marker,
        )

        await reply_trigger.click(timeout=_INTERACTION_TIMEOUT_MS)
        return await self._wait_for_reply_composer(
            page,
            target,
            baseline_marker=baseline_marker,
        )

    async def _wait_for_reply_composer(
        self,
        page: Page,
        target: _ReplyTarget,
        *,
        baseline_marker: str,
    ) -> _ReplyComposer:
        visible_composer_selectors = _visible_selectors(
            _COMPOSER_SELECTORS
        )
        new_composer_selectors = _visible_unmarked_selectors(
            _COMPOSER_SELECTORS,
            attribute=_BASELINE_COMPOSER_MARKER_ATTRIBUTE,
            marker=baseline_marker,
        )
        target_scope_selector = (
            _TARGET_REPLY_SCOPE_SELECTOR_TEMPLATE.format(
                post_id=target.post_id
            )
        )
        scope_selectors = [
            (
                f'{selector}:not(['
                f'{_BASELINE_REPLY_SCOPE_MARKER_ATTRIBUTE}="{baseline_marker}"'
                f"]):has({visible_composer_selectors})"
            )
            for selector in _REPLY_DIALOG_SCOPE_SELECTORS
        ]
        scope_selectors.append(
            f":is({target_scope_selector})"
            f":has({new_composer_selectors})"
        )
        scope = page.locator(", ".join(scope_selectors)).first

        try:
            await scope.wait_for(
                state="visible",
                timeout=_INTERACTION_TIMEOUT_MS,
            )
            composer = scope.locator(new_composer_selectors).first
            if not await composer.count() or not await composer.is_visible():
                composer = scope.locator(
                    visible_composer_selectors
                ).first
            await composer.wait_for(
                state="visible",
                timeout=_INTERACTION_TIMEOUT_MS,
            )
        except PlaywrightTimeoutError as error:
            await self._raise_for_page(page, None, purpose="target")
            raise BrowserInteractionError(
                "X reply composer did not open in the requested target scope",
                current_url=_safe_page_url(page.url),
            ) from error

        return _ReplyComposer(composer=composer, scope=scope)

    async def _wait_for_composer(
        self,
        page: Page,
        *,
        purpose: _PagePurpose,
    ) -> Locator:
        visible_selectors = _visible_selectors(_COMPOSER_SELECTORS)
        composer = page.locator(visible_selectors).first
        try:
            await composer.wait_for(
                state="visible",
                timeout=_INTERACTION_TIMEOUT_MS,
            )
        except PlaywrightTimeoutError as error:
            await self._raise_for_page(page, None, purpose=purpose)
            raise BrowserInteractionError(
                "X post composer did not become available",
                current_url=_safe_page_url(page.url),
            ) from error
        return composer

    async def _fill_composer(self, composer: Locator, content: str) -> None:
        await composer.fill(content, timeout=_INTERACTION_TIMEOUT_MS)
        rendered_value = await composer.evaluate(
            """element => {
                if (typeof element.value === "string") {
                    return element.value;
                }
                return element.innerText;
            }"""
        )
        if not isinstance(rendered_value, str) or _normalize_line_endings(
            rendered_value
        ) != _normalize_line_endings(content):
            raise BrowserInteractionError(
                "X composer content could not be verified"
            )

    async def _submit_locator(
        self,
        page: Page,
        composer: Locator,
        *,
        purpose: _PagePurpose,
        required_scope: Locator | None = None,
    ) -> Locator:
        scope: Locator | Page
        if required_scope is not None:
            scope = required_scope
        else:
            scope = page
            dialog = composer.locator(
                _COMPOSER_DIALOG_ANCESTOR_SELECTOR
            )
            if await dialog.count() and await dialog.first.is_visible():
                scope = dialog.first
            else:
                form = composer.locator(_FORM_ANCESTOR_SELECTOR)
                if await form.count() and await form.first.is_visible():
                    scope = form.first

        visible_selectors = _visible_selectors(_SUBMIT_SELECTORS)
        submit = scope.locator(visible_selectors).first
        try:
            await submit.wait_for(
                state="visible",
                timeout=_INTERACTION_TIMEOUT_MS,
            )
            element = await submit.element_handle()
            if element is None:
                raise BrowserInteractionError(
                    "X submit control could not be resolved"
                )
            await page.wait_for_function(
                """button =>
                    !button.disabled &&
                    button.getAttribute("aria-disabled") !== "true";
                """,
                arg=element,
                timeout=_INTERACTION_TIMEOUT_MS,
            )
        except PlaywrightTimeoutError as error:
            await self._raise_for_page(page, None, purpose=purpose)
            raise PlatformActionRejectedError(
                "X submission control remained disabled"
            ) from error
        return submit

    async def _submit_post(
        self,
        page: Page,
        submit: Locator,
        *,
        reply_target: _ReplyTarget | None,
        purpose: _PagePurpose,
    ) -> _PublishedPost:
        baseline_ids = await _baseline_status_ids(page)
        if reply_target is not None:
            baseline_ids.add(reply_target.post_id)

        dispatch_started = False
        submission_error: Exception | None = None
        response: Response | None = None
        try:
            async with page.expect_response(
                _is_create_tweet_response,
                timeout=_NAVIGATION_TIMEOUT_MS,
            ) as response_info:
                dispatch_started = True
                await submit.click(timeout=_INTERACTION_TIMEOUT_MS)
            response = await response_info.value
        except Exception as error:
            if not dispatch_started:
                raise
            submission_error = error

        try:
            tombstone = False
            if response is not None:
                payload = await _response_payload(response)
                created_id, tombstone = _extract_created_post(payload)
                reply_parent_id, reply_parent_exposed = (
                    _extract_created_reply_parent(payload)
                )
                _raise_for_create_response(
                    response,
                    payload,
                    reply_target=reply_target,
                    created_id=created_id,
                )
                if created_id is not None:
                    if created_id in baseline_ids:
                        raise BrowserInteractionError(
                            "X submission response did not identify a new post",
                            current_url=_safe_page_url(page.url),
                            retryable=False,
                        )
                    if (
                        reply_target is not None
                        and reply_parent_exposed
                        and reply_parent_id != reply_target.post_id
                    ):
                        raise BrowserInteractionError(
                            "X returned a post that was not bound to the "
                            "requested reply target",
                            current_url=_safe_page_url(page.url),
                            retryable=False,
                        )
                    return _PublishedPost(
                        post_id=created_id,
                        url=_canonical_status_url(created_id),
                        response_status=response.status,
                    )

            if tombstone:
                raise PlatformActionRejectedError(
                    "X did not return a publishable post"
                )

            if reply_target is None:
                discovered = await _wait_for_created_status(
                    page,
                    baseline_ids,
                )
                if discovered is not None:
                    return _PublishedPost(
                        post_id=discovered.post_id,
                        url=discovered.navigation_url,
                        response_status=(
                            response.status if response is not None else None
                        ),
                    )

            try:
                await self._raise_for_page(page, None, purpose=purpose)
            except (PlatformRateLimitError, PlatformUnavailableError) as error:
                raise BrowserInteractionError(
                    "X submission outcome could not be confirmed",
                    current_url=_safe_page_url(page.url),
                    retryable=False,
                ) from error

            if submission_error is not None:
                raise BrowserInteractionError(
                    "X submission outcome could not be confirmed",
                    current_url=_safe_page_url(page.url),
                    retryable=False,
                ) from submission_error
            raise BrowserInteractionError(
                "X submission response did not identify the published post",
                current_url=_safe_page_url(page.url),
                retryable=False,
            )
        except BrowserInteractionError:
            raise
        except PlatformError:
            raise
        except Exception as error:
            raise BrowserInteractionError(
                "X submission outcome could not be confirmed",
                current_url=_safe_page_url(page.url),
                retryable=False,
            ) from error


def _validate_post_content(content: str) -> str:
    if not isinstance(content, str) or not content.strip():
        raise PlatformActionRejectedError("X post content is required")
    if len(content) > _MAX_POST_LENGTH:
        raise PlatformActionRejectedError(
            f"X post content must be at most {_MAX_POST_LENGTH} characters"
        )
    return content


def _resolve_direct_reply_target(reply_to: str | None) -> _ReplyTarget | None:
    if reply_to is None:
        return None
    if not isinstance(reply_to, str) or not reply_to.strip():
        raise PlatformActionRejectedError(
            "X reply target must be a status URL or post id"
        )

    normalized = reply_to.strip()
    if _POST_ID_RE.fullmatch(normalized):
        return _ReplyTarget(
            post_id=normalized,
            navigation_url=_canonical_status_url(normalized),
        )
    try:
        return _parse_status_url(normalized)
    except ValueError as error:
        raise PlatformActionRejectedError(
            "X reply target must be an x.com or twitter.com status URL"
        ) from error


def _resolve_reply_target(
    *,
    target_url: str | None,
    parent_post_id: str | None,
) -> _ReplyTarget | None:
    url_target: _ReplyTarget | None = None
    if target_url is not None:
        try:
            url_target = _parse_status_url(target_url)
        except ValueError as error:
            raise PlatformActionRejectedError(
                "X target_url must be an x.com or twitter.com status URL"
            ) from error

    id_target: _ReplyTarget | None = None
    if parent_post_id is not None:
        normalized_id = parent_post_id.strip()
        if not _POST_ID_RE.fullmatch(normalized_id):
            raise PlatformActionRejectedError(
                "X parent_post_id must be a numeric X post id"
            )
        id_target = _ReplyTarget(
            post_id=normalized_id,
            navigation_url=_canonical_status_url(normalized_id),
        )

    if url_target is not None and id_target is not None:
        if url_target.post_id != id_target.post_id:
            raise PlatformActionRejectedError(
                "X target_url and parent_post_id identify different posts"
            )
        return url_target
    return url_target or id_target


def _parse_status_url(value: str) -> _ReplyTarget:
    parsed = _split_safe_url(value)
    if parsed is None:
        raise ValueError("invalid status URL")
    if parsed.hostname.casefold() not in _X_TARGET_HOSTS:
        raise ValueError("unsupported X status host")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("status URL must not contain user information")
    if parsed.port is not None:
        raise ValueError("status URL must not contain a port")

    match = _STATUS_PATH_RE.fullmatch(parsed.path)
    if match is None:
        raise ValueError("URL is not an X status URL")
    post_id = match.group("post_id")
    normalized_path = parsed.path.rstrip("/")
    return _ReplyTarget(
        post_id=post_id,
        navigation_url=urlunsplit(("https", "x.com", normalized_path, "", "")),
    )


def _try_status_url(value: str | None) -> _ReplyTarget | None:
    if not value:
        return None
    try:
        return _parse_status_url(value)
    except ValueError:
        return None


def _canonical_status_url(post_id: str) -> str:
    return f"{_X_ORIGIN}/i/web/status/{post_id}"


async def _article_contains_status(article: Locator, post_id: str) -> bool:
    links = article.locator(_STATUS_ANCHOR_SELECTOR)
    for index in range(await links.count()):
        href = await links.nth(index).get_attribute("href")
        target = _try_status_url(urljoin(_X_ORIGIN, href or ""))
        if target is not None and target.post_id == post_id:
            return True
    return False


async def _baseline_status_ids(page: Page) -> set[str]:
    post_ids: set[str] = set()
    current = _try_status_url(page.url)
    if current is not None:
        post_ids.add(current.post_id)

    links = page.locator(_CREATED_STATUS_ANCHOR_SELECTOR)
    for index in range(min(await links.count(), 50)):
        href = await links.nth(index).get_attribute("href")
        target = _try_status_url(urljoin(page.url, href or ""))
        if target is not None:
            post_ids.add(target.post_id)
    return post_ids


async def _wait_for_created_status(
    page: Page,
    baseline_ids: set[str],
) -> _ReplyTarget | None:
    try:
        handle = await page.wait_for_function(
            """({selector, baseline}) => {
                const extract = value => {
                    try {
                        const url = new URL(value, window.location.href);
                        const match = url.pathname.match(
                            /\\/status\\/([1-9][0-9]{0,29})\\/?$/
                        );
                        if (!match) return null;
                        return {id: match[1], href: url.href};
                    } catch (_) {
                        return null;
                    }
                };
                const current = extract(window.location.href);
                if (current && !baseline.includes(current.id)) return current;
                for (const anchor of document.querySelectorAll(selector)) {
                    const candidate = extract(anchor.href);
                    if (candidate && !baseline.includes(candidate.id)) {
                        return candidate;
                    }
                }
                return false;
            }""",
            arg={
                "selector": _CREATED_STATUS_ANCHOR_SELECTOR,
                "baseline": sorted(baseline_ids),
            },
            timeout=_CREATED_CONTENT_TIMEOUT_MS,
        )
    except PlaywrightTimeoutError:
        return None

    discovered = await handle.json_value()
    if not isinstance(discovered, Mapping):
        return None
    post_id = _post_id(discovered.get("id"))
    href = discovered.get("href")
    if post_id is None or not isinstance(href, str):
        return None
    target = _try_status_url(urljoin(page.url, href))
    if target is None or target.post_id != post_id:
        return None
    return target


def _is_create_tweet_response(response: Response) -> bool:
    try:
        parsed = urlsplit(response.url)
        port = parsed.port
        method = response.request.method.upper()
    except (AttributeError, ValueError):
        return False

    hostname = parsed.hostname.casefold() if parsed.hostname else ""
    return (
        method == "POST"
        and parsed.scheme.casefold() == "https"
        and parsed.username is None
        and parsed.password is None
        and port in {None, 443}
        and hostname in _X_CREATE_TWEET_HOSTS
        and parsed.path.rstrip("/").endswith("/CreateTweet")
    )


async def _response_payload(response: Response) -> Mapping[str, Any]:
    try:
        payload = await response.json()
    except (PlaywrightError, TypeError, ValueError):
        return {}
    return payload if isinstance(payload, Mapping) else {}


def _extract_created_post(
    payload: Mapping[str, Any],
) -> tuple[str | None, bool]:
    result = _create_tweet_result(payload)
    return _post_id_from_result(result, depth=0)


def _create_tweet_result(
    payload: Mapping[str, Any],
) -> Mapping[str, Any]:
    data = _as_mapping(payload.get("data"))
    create_tweet = _as_mapping(
        data.get("create_tweet") or data.get("createTweet")
    )
    tweet_results = _as_mapping(
        create_tweet.get("tweet_results") or create_tweet.get("tweetResults")
    )
    return _as_mapping(tweet_results.get("result"))


def _extract_created_reply_parent(
    payload: Mapping[str, Any],
) -> tuple[str | None, bool]:
    return _reply_parent_id_from_result(
        _create_tweet_result(payload),
        depth=0,
    )


def _post_id_from_result(
    result: Mapping[str, Any],
    *,
    depth: int,
) -> tuple[str | None, bool]:
    if not result or depth > 5:
        return None, False

    typename = str(result.get("__typename") or "").casefold()
    if any(
        marker in typename
        for marker in ("tombstone", "unavailable", "withheld")
    ):
        return None, True

    direct_id = _post_id(result.get("rest_id"))
    if direct_id is not None:
        return direct_id, False

    legacy = _as_mapping(result.get("legacy"))
    legacy_id = _post_id(legacy.get("id_str"))
    if legacy_id is not None:
        return legacy_id, False

    tombstone_seen = False
    for key in ("tweet", "result"):
        nested_id, nested_tombstone = _post_id_from_result(
            _as_mapping(result.get(key)),
            depth=depth + 1,
        )
        if nested_id is not None:
            return nested_id, nested_tombstone
        tombstone_seen = tombstone_seen or nested_tombstone

    nested_results = _as_mapping(
        result.get("tweet_results") or result.get("tweetResults")
    )
    nested_id, nested_tombstone = _post_id_from_result(
        _as_mapping(nested_results.get("result")),
        depth=depth + 1,
    )
    return nested_id, tombstone_seen or nested_tombstone


def _reply_parent_id_from_result(
    result: Mapping[str, Any],
    *,
    depth: int,
) -> tuple[str | None, bool]:
    if not result or depth > 5:
        return None, False

    legacy = _as_mapping(result.get("legacy"))
    for container in (legacy, result):
        for field in _REPLY_PARENT_ID_FIELDS:
            if field in container:
                return _post_id(container.get(field)), True

    for key in ("tweet", "result"):
        parent_id, exposed = _reply_parent_id_from_result(
            _as_mapping(result.get(key)),
            depth=depth + 1,
        )
        if exposed:
            return parent_id, True

    nested_results = _as_mapping(
        result.get("tweet_results") or result.get("tweetResults")
    )
    return _reply_parent_id_from_result(
        _as_mapping(nested_results.get("result")),
        depth=depth + 1,
    )


def _raise_for_create_response(
    response: Response,
    payload: Mapping[str, Any],
    *,
    reply_target: _ReplyTarget | None,
    created_id: str | None,
) -> None:
    errors = _graphql_errors(payload)
    codes = {code for code, _ in errors if code}
    messages = " ".join(message for _, message in errors).casefold()

    if created_id is not None and response.status < 400:
        return
    if codes & _CHALLENGE_CODES or _contains_signal(
        messages, _CHALLENGE_SIGNALS
    ):
        raise XChallengeError(
            "X requires a human challenge or account verification"
        )
    if codes & _AUTHENTICATION_CODES or _contains_signal(
        messages, _AUTHENTICATION_SIGNALS
    ):
        raise PlatformAuthenticationError(
            "X account session is not authenticated"
        )
    if codes & _RATE_LIMIT_CODES or _contains_signal(
        messages, _RATE_LIMIT_SIGNALS
    ) or response.status == 429:
        raise PlatformRateLimitError(
            "X rate limit reached",
            retry_after_seconds=_retry_after_seconds(response.headers),
        )
    if reply_target is not None and (
        codes & _TARGET_NOT_FOUND_CODES
        or _contains_signal(messages, _NOT_FOUND_SIGNALS)
        or response.status == 404
    ):
        raise XTargetNotFoundError("X reply target was not found")
    if codes & _ACTION_REJECTED_CODES:
        raise PlatformActionRejectedError("X rejected the post content")
    if codes & _TRANSIENT_CODES or _contains_signal(
        messages, _TRANSIENT_SIGNALS
    ):
        raise BrowserInteractionError(
            "X submission outcome could not be confirmed",
            retryable=False,
        )

    if response.status in {401, 407}:
        raise PlatformAuthenticationError(
            "X account session is not authenticated"
        )
    if response.status >= 500:
        raise BrowserInteractionError(
            "X submission outcome could not be confirmed",
            retryable=False,
        )
    if response.status == 403:
        raise PlatformActionRejectedError("X rejected the action")
    if response.status >= 400:
        raise PlatformActionRejectedError("X rejected the action")
    if errors:
        raise PlatformActionRejectedError("X rejected the action")


def _graphql_errors(
    payload: Mapping[str, Any],
) -> list[tuple[str, str]]:
    raw_errors = payload.get("errors")
    if not isinstance(raw_errors, Sequence) or isinstance(
        raw_errors, (str, bytes, bytearray)
    ):
        return []

    errors: list[tuple[str, str]] = []
    for raw_error in raw_errors:
        error = _as_mapping(raw_error)
        extensions = _as_mapping(error.get("extensions"))
        code = _error_code(
            error.get("code")
            or extensions.get("code")
            or extensions.get("name")
        )
        message_value = error.get("message")
        message = message_value if isinstance(message_value, str) else ""
        errors.append((code, message))
    return errors


def _error_code(value: object) -> str:
    if isinstance(value, bool) or value is None:
        return ""
    if isinstance(value, (int, str)):
        return str(value).strip().upper()
    return ""


def _post_id(value: object) -> str | None:
    if isinstance(value, bool):
        return None
    normalized = str(value) if isinstance(value, int) else value
    if isinstance(normalized, str) and _POST_ID_RE.fullmatch(normalized):
        return normalized
    return None


def _as_mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _retry_after_seconds(headers: Mapping[str, str]) -> float | None:
    retry_after = headers.get("retry-after")
    if retry_after is not None:
        try:
            return max(0.0, float(retry_after))
        except ValueError:
            pass

    reset_at = headers.get("x-rate-limit-reset")
    if reset_at is not None:
        try:
            return max(0.0, float(reset_at) - time.time())
        except ValueError:
            pass
    return None


def _split_safe_url(value: str):
    try:
        parsed = urlsplit(value.strip())
        _ = parsed.port
    except (AttributeError, ValueError):
        return None
    if parsed.scheme.casefold() not in {"http", "https"} or not parsed.hostname:
        return None
    return parsed


def _safe_page_url(value: str | None) -> str | None:
    if not value:
        return None
    parsed = _split_safe_url(value)
    if parsed is None:
        return None
    hostname = parsed.hostname.casefold() if parsed.hostname else ""
    path = parsed.path if hostname in _X_TARGET_HOSTS else "/"
    return urlunsplit((parsed.scheme, hostname, path, "", ""))


def _path_has_prefix(path: str, prefixes: Sequence[str]) -> bool:
    return any(path == prefix or path.startswith(f"{prefix}/") for prefix in prefixes)


async def _visible_state_text(page: Page) -> str:
    messages: list[str] = []
    locator = page.locator(_STATE_MESSAGE_SELECTOR)
    for index in range(min(await locator.count(), 20)):
        candidate = locator.nth(index)
        if await candidate.is_visible():
            messages.append(
                await candidate.inner_text(timeout=_INTERACTION_TIMEOUT_MS)
            )
    return " ".join(messages).casefold()


def _visible_selectors(selectors: Sequence[str]) -> str:
    return ", ".join(f"{selector}:visible" for selector in selectors)


def _visible_unmarked_selectors(
    selectors: Sequence[str],
    *,
    attribute: str,
    marker: str,
) -> str:
    return ", ".join(
        f'{selector}:not([{attribute}="{marker}"]):visible'
        for selector in selectors
    )


async def _mark_visible_elements(
    page: Page,
    selectors: Sequence[str],
    *,
    attribute: str,
    marker: str,
) -> None:
    await page.locator(_visible_selectors(selectors)).evaluate_all(
        """(elements, marker) => {
            for (const element of elements) {
                element.setAttribute(marker.attribute, marker.value);
            }
        }""",
        {"attribute": attribute, "value": marker},
    )


async def _has_visible(page: Page, selector: str) -> bool:
    locator = page.locator(selector)
    for index in range(min(await locator.count(), 20)):
        if await locator.nth(index).is_visible():
            return True
    return False


def _contains_signal(text: str, signals: Sequence[str]) -> bool:
    return any(signal in text for signal in signals)


def _normalize_line_endings(value: str) -> str:
    return value.replace("\r\n", "\n").replace("\r", "\n")
