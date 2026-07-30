"""Browser-backed Reddit publishing through the stable old Reddit forms."""

from __future__ import annotations

import html
import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote, unquote, urljoin, urlsplit, urlunsplit
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
from bot.models import (
    ActionResult,
    ActionType,
    Platform,
    SocialAction,
    normalize_target_url,
)
from bot.platforms.base import (
    PlatformActionRejectedError,
    PlatformAuthenticationError,
    PlatformError,
    PlatformRateLimitError,
    PlatformUnavailableError,
)

logger = structlog.get_logger(__name__)

_OLD_REDDIT_ORIGIN = "https://old.reddit.com"
_CANONICAL_REDDIT_ORIGIN = "https://www.reddit.com"
_REDDIT_TARGET_HOSTS = frozenset({"reddit.com", "www.reddit.com", "old.reddit.com"})
_NAVIGATION_TIMEOUT_MS = 30_000
_INTERACTION_TIMEOUT_MS = 15_000
_CREATED_CONTENT_TIMEOUT_MS = 5_000

_FULLNAME_RE = re.compile(r"^(?P<kind>t[13])_(?P<id>[a-z0-9]+)$", re.IGNORECASE)
_SUBREDDIT_RE = re.compile(r"^[a-z0-9_]{2,21}$", re.IGNORECASE)
_COMMENTS_PATH_RE = re.compile(
    r"^/(?:r/(?P<subreddit>[a-z0-9_]{2,21})/)?comments/"
    r"(?P<post_id>[a-z0-9]+)"
    r"(?:/(?P<slug>[^/]+))?"
    r"(?:/(?P<comment_id>[a-z0-9]+))?/?$",
    re.IGNORECASE,
)
_RETRY_AFTER_RE = re.compile(
    r"(?:try again in|wait)\s+(?P<amount>\d+(?:\.\d+)?)\s*"
    r"(?P<unit>second|minute|hour)s?",
    re.IGNORECASE,
)

# Submit-page selectors.
_SUBMIT_FORM_SELECTOR = 'form#newlink, form[action$="/api/submit"]'
_POST_TITLE_SELECTOR = 'input[name="title"]'
_POST_URL_SELECTOR = 'input[name="url"]'
_POST_TEXT_SELECTOR = 'textarea[name="text"]'
_POST_KIND_SELECTOR = 'input[name="kind"]'
_FORM_SUBMIT_SELECTOR = (
    'button[type="submit"], button[name="submit"], input[type="submit"]'
)

# Comment and reply selectors. Reply locators are always resolved inside the
# target comment's ``data-fullname`` container.
_TOP_LEVEL_COMMENT_FORM_SELECTOR = "div.commentarea form.usertext.cloneable"
_REPLY_BUTTON_SELECTOR = (
    "ul.buttons li.reply-button a, "
    'ul.buttons a[data-event-action="comment"], '
    "a.reply-button"
)
_REPLY_FORM_SELECTOR = "form.usertext"
_COMMENT_TEXT_SELECTOR = 'textarea[name="text"]'
_PARENT_FULLNAME_SELECTOR = 'input[name="thing_id"]'
_THING_SELECTOR_TEMPLATE = '.thing[data-fullname="{fullname}"]'
_POST_THING_SELECTOR = '.thing.link[data-fullname^="t3_"]'
_PERMALINK_SELECTOR = (
    'a.bylink[href*="/comments/"], '
    'a[data-event-action="permalink"][href*="/comments/"]'
)

# Page-state selectors. These are detection only; the adapter never attempts
# to solve or bypass a challenge. Message text is read only from Reddit-owned
# status and form-error containers, never from user content or the whole page.
_PARENT_COMMENT_ANCESTOR_SELECTOR = (
    'xpath=ancestor::*[starts-with(@data-fullname, "t1_")][1]'
)
_LOGIN_SELECTOR = (
    "form#login_login-main, "
    'form.login-form input[name="user"], '
    'form[action*="/login"] input[name="password"]'
)
_CHALLENGE_SELECTOR = (
    'iframe[src*="captcha"], iframe[src*="challenge"], '
    ".g-recaptcha, [data-sitekey], "
    'form[action*="/challenge"], form[action*="/verify"]'
)
_PLATFORM_MESSAGE_SELECTOR = (
    ".error-page, "
    "#siteTable > .infobar, #siteTable > .notice, "
    "div.content > .infobar, div.content > .notice, "
    "form#newlink .error, form.usertext .error"
)
_ERROR_PAGE_SELECTOR = ".error-page"

_RATE_LIMIT_SIGNALS = (
    "you are doing that too much",
    "you've been doing that a lot",
    "try again in",
    "too many requests",
    "rate limit",
)
_AUTHENTICATION_SIGNALS = (
    "you must be logged in",
    "please log in to continue",
    "your session has expired",
    "log in to reddit",
)
_CHALLENGE_SIGNALS = (
    "are you a human",
    "prove you're human",
    "prove you are human",
    "security check",
    "verify your identity",
    "additional verification",
    "complete the captcha",
)
_SUBREDDIT_UNAVAILABLE_SIGNALS = (
    "this community is private",
    "only approved members can view",
    "you must be invited to visit this community",
    "this subreddit was banned",
    "this community has been banned",
    "subreddit does not exist",
    "there doesn't seem to be anything here",
)
_NOT_FOUND_SIGNALS = (
    "page not found",
    "the page you requested does not exist",
    "there doesn't seem to be anything here",
)

_RATE_LIMIT_CODES = frozenset({"RATELIMIT", "RATE_LIMIT", "TOO_MANY_REQUESTS"})
_AUTHENTICATION_CODES = frozenset(
    {"USER_REQUIRED", "NO_USER", "MODHASH_REQUIRED", "BAD_CSRF", "WRONG_CREDENTIALS"}
)
_CHALLENGE_CODES = frozenset(
    {"BAD_CAPTCHA", "CAPTCHA_REQUIRED", "NEEDS_CAPTCHA", "VERIFY_REQUIRED"}
)
_SUBREDDIT_UNAVAILABLE_CODES = frozenset(
    {
        "SUBREDDIT_NOEXIST",
        "SUBREDDIT_NOTALLOWED",
        "SUBREDDIT_REQUIRED",
        "DELETED_SUBREDDIT",
        "GOLD_SUBREDDIT",
        "QUARANTINE",
    }
)
_TARGET_REJECTION_CODES = frozenset(
    {
        "NO_THING_ID",
        "DELETED_COMMENT",
        "DELETED_LINK",
        "THREAD_LOCKED",
        "LOCKED",
        "ARCHIVED",
        "TOO_OLD",
        "REPLY_DISABLED",
        "FORBIDDEN",
    }
)

_TargetKind = Literal["t1", "t3"]
_PagePurpose = Literal["submit", "target", "created"]


class SubredditNotAllowedError(PlatformActionRejectedError):
    """The configured account may not act in the requested subreddit."""


class SubredditUnavailableError(PlatformActionRejectedError):
    """The requested subreddit is missing, private, banned, or inaccessible."""


class RedditTargetNotFoundError(PlatformActionRejectedError):
    """A requested Reddit post or comment could not be resolved."""


class RedditChallengeError(PlatformAuthenticationError):
    """Reddit requires a human challenge or account verification."""


class BrowserInteractionError(PlatformUnavailableError):
    """A browser operation failed with safe, best-effort diagnostics."""

    def __init__(
        self,
        message: str,
        *,
        current_url: str | None = None,
        screenshot_path: Path | None = None,
        retryable: bool | None = None,
    ) -> None:
        self.base_message = message
        self.current_url = current_url
        self.screenshot_path = screenshot_path
        super().__init__(self._render_message(), retryable=retryable)

    def attach_capture(self, capture: FailureCapture | None) -> None:
        """Add safe diagnostics without changing the exception's type."""

        if capture is None:
            return
        if self.current_url is None:
            self.current_url = _safe_page_url(capture.page_url)
        if self.screenshot_path is None:
            self.screenshot_path = capture.path
        self.args = (self._render_message(),)

    def _render_message(self) -> str:
        details = [self.base_message]
        if self.current_url is not None:
            details.append(f"current_url={self.current_url}")
        if self.screenshot_path is not None:
            details.append(f"screenshot={self.screenshot_path.name}")
        return "; ".join(details)


@dataclass(frozen=True, slots=True)
class _TargetReference:
    fullname: str
    post_fullname: str
    comment_fullname: str | None
    subreddit: str | None
    navigation_url: str
    supplied_permalink: str | None


@dataclass(frozen=True, slots=True)
class _ResolvedTarget:
    fullname: str
    post_fullname: str
    comment_fullname: str | None
    subreddit: str
    permalink: str


@dataclass(slots=True)
class _SubmissionState:
    dispatched: bool = False
    api_rejection: PlatformError | None = None


@dataclass(frozen=True, slots=True)
class _CreatedResponseEvidence:
    fullname: str
    permalink: str | None
    parent_fullname: str | None


@dataclass(frozen=True, slots=True)
class _PublishedContent:
    fullname: str
    permalink: str
    subreddit: str
    post_fullname: str


class RedditAdapter:
    """Publish Reddit actions for one configured, browser-backed account."""

    def __init__(
        self,
        browser_manager: BrowserManager,
        settings: Settings,
        account_name: str,
    ) -> None:
        normalized_account_name = account_name.strip()
        if not normalized_account_name:
            raise PlatformActionRejectedError("Reddit account name is required")

        account = settings.accounts.reddit.get(normalized_account_name)
        if account is None:
            raise PlatformActionRejectedError(
                f"Reddit account '{normalized_account_name}' is not configured"
            )
        if not account.enabled:
            raise PlatformActionRejectedError(
                f"Reddit account '{normalized_account_name}' is disabled"
            )

        try:
            browser_manager.resolve_profile(Platform.REDDIT, normalized_account_name)
        except BrowserAccountNotConfiguredError as error:
            raise PlatformActionRejectedError(
                f"Reddit account '{normalized_account_name}' has no browser profile"
            ) from error

        allowed_subreddits: set[str] = set()
        for configured_subreddit in account.allowed_subreddits:
            try:
                allowed_subreddits.add(
                    _normalize_subreddit(configured_subreddit).casefold()
                )
            except ValueError as error:
                raise PlatformActionRejectedError(
                    "Reddit account has an invalid subreddit allowlist entry"
                ) from error

        self._browser_manager = browser_manager
        self._account_name = normalized_account_name
        self._allowed_subreddits = frozenset(allowed_subreddits)
        self._log = logger.bind(
            platform=Platform.REDDIT.value,
            account_name=normalized_account_name,
        )

    @property
    def account_name(self) -> str:
        """Return the configured account owned by this adapter."""

        return self._account_name

    async def execute(self, action: SocialAction) -> ActionResult:
        """Dispatch a validated Reddit action to its concrete browser flow."""

        if action.platform is not Platform.REDDIT:
            raise PlatformActionRejectedError(
                "RedditAdapter only accepts actions for platform 'reddit'"
            )
        if action.account_name != self._account_name:
            raise PlatformActionRejectedError(
                "Reddit action account does not match this adapter"
            )

        if action.action_type is ActionType.REDDIT_POST:
            if action.subreddit is None or action.title is None:
                raise PlatformActionRejectedError(
                    "Reddit post requires subreddit and title"
                )
            return await self.create_post(
                action.subreddit,
                action.title,
                body=action.content or None,
                url=action.target_url,
                action_id=action.id,
            )

        if action.action_type is ActionType.REDDIT_COMMENT:
            target = action.target_url or action.parent_post_id
            if target is None:
                raise PlatformActionRejectedError(
                    "Reddit comment requires a target post"
                )
            return await self.comment_on_post(
                target,
                action.content,
                subreddit=action.subreddit,
                parent_post_id=action.parent_post_id,
                action_id=action.id,
            )

        if action.action_type is ActionType.REDDIT_REPLY:
            target = action.target_url or action.parent_comment_id
            if target is None:
                raise PlatformActionRejectedError(
                    "Reddit reply requires a target comment"
                )
            return await self.reply_to_comment(
                target,
                action.content,
                subreddit=action.subreddit,
                parent_comment_id=action.parent_comment_id,
                parent_post_id=action.parent_post_id,
                action_id=action.id,
            )

        raise PlatformActionRejectedError(
            f"Unsupported Reddit action type '{action.action_type.value}'"
        )

    async def create_post(
        self,
        subreddit: str,
        title: str,
        body: str | None = None,
        url: str | None = None,
        *,
        action_id: UUID | None = None,
    ) -> ActionResult:
        """Create one old Reddit text or link post."""

        normalized_subreddit = self._require_allowed_subreddit(subreddit)
        normalized_title = title.strip()
        if not normalized_title:
            raise PlatformActionRejectedError("Reddit post title is required")

        has_body = body is not None and bool(body.strip())
        has_url = url is not None and bool(url.strip())
        if has_body == has_url:
            raise PlatformActionRejectedError(
                "Reddit post requires exactly one of body or url"
            )

        link_url: str | None = None
        if has_url:
            try:
                link_url = _normalize_link_post_url(url or "")
            except ValueError as error:
                raise PlatformActionRejectedError(
                    "Reddit link post URL must be absolute HTTP(S), contain no "
                    "credentials, and use only the scheme's default port"
                ) from error

        resolved_action_id = action_id or uuid4()
        post_kind = "text" if has_body else "link"

        async def operation(
            page: Page,
            submission_state: _SubmissionState,
        ) -> ActionResult:
            submit_url = (
                f"{_OLD_REDDIT_ORIGIN}/r/{quote(normalized_subreddit, safe='_')}/"
                f"submit{'?selftext=true' if has_body else ''}"
            )
            await self._navigate(
                page,
                submit_url,
                purpose="submit",
                subreddit=normalized_subreddit,
            )

            form = page.locator(_SUBMIT_FORM_SELECTOR).first
            await form.wait_for(state="visible", timeout=_INTERACTION_TIMEOUT_MS)
            await self._prepare_post_form(form, is_text=has_body)

            title_input = form.locator(_POST_TITLE_SELECTOR).first
            await title_input.wait_for(
                state="visible", timeout=_INTERACTION_TIMEOUT_MS
            )
            await title_input.fill(normalized_title)

            if has_body:
                text_input = form.locator(_POST_TEXT_SELECTOR).first
                await text_input.wait_for(
                    state="visible", timeout=_INTERACTION_TIMEOUT_MS
                )
                await text_input.fill(body or "")
            else:
                url_input = form.locator(_POST_URL_SELECTOR).first
                await url_input.wait_for(
                    state="visible", timeout=_INTERACTION_TIMEOUT_MS
                )
                await url_input.fill(link_url or "")

            response, published = await self._submit_form(
                page,
                form,
                endpoint="/api/submit",
                subreddit=normalized_subreddit,
                expected_kind="t3",
                excluded_fullnames=frozenset(),
                expected_parent_fullname=None,
                submission_state=submission_state,
            )
            return ActionResult(
                action_id=resolved_action_id,
                success=True,
                external_content_id=published.fullname,
                external_content_url=published.permalink,
                message=f"Reddit {post_kind} post created",
                metadata={
                    "subreddit": published.subreddit,
                    "post_kind": post_kind,
                    "response_status": response.status,
                },
            )

        return await self._run_operation(
            resolved_action_id,
            ActionType.REDDIT_POST,
            operation,
        )

    async def comment_on_post(
        self,
        post_url: str,
        body: str,
        *,
        subreddit: str | None = None,
        parent_post_id: str | None = None,
        action_id: UUID | None = None,
    ) -> ActionResult:
        """Comment on a Reddit post URL or ``t3_`` fullname."""

        if not body.strip():
            raise PlatformActionRejectedError("Reddit comment body is required")
        subreddit_hint = (
            self._require_allowed_subreddit(subreddit)
            if subreddit is not None
            else None
        )
        expected_parent = _optional_fullname(
            parent_post_id,
            expected_kind="t3",
            field_name="parent_post_id",
        )
        resolved_action_id = action_id or uuid4()

        async def operation(
            page: Page,
            submission_state: _SubmissionState,
        ) -> ActionResult:
            target = await self._resolve_target(
                page,
                post_url,
                expected_kind="t3",
                subreddit_hint=subreddit_hint,
                expected_fullname=expected_parent,
                expected_post_fullname=expected_parent,
            )
            form = page.locator(_TOP_LEVEL_COMMENT_FORM_SELECTOR).first
            await form.wait_for(state="visible", timeout=_INTERACTION_TIMEOUT_MS)
            await self._verify_parent_field(form, target.fullname)

            textarea = form.locator(_COMMENT_TEXT_SELECTOR).first
            await textarea.wait_for(
                state="visible", timeout=_INTERACTION_TIMEOUT_MS
            )
            await textarea.fill(body)

            response, published = await self._submit_form(
                page,
                form,
                endpoint="/api/comment",
                subreddit=target.subreddit,
                expected_kind="t1",
                excluded_fullnames=frozenset({target.fullname}),
                expected_parent_fullname=target.fullname,
                submission_state=submission_state,
            )
            return ActionResult(
                action_id=resolved_action_id,
                success=True,
                external_content_id=published.fullname,
                external_content_url=published.permalink,
                message="Reddit comment created",
                metadata={
                    "subreddit": published.subreddit,
                    "parent_post_id": target.fullname,
                    "response_status": response.status,
                },
            )

        return await self._run_operation(
            resolved_action_id,
            ActionType.REDDIT_COMMENT,
            operation,
        )

    async def reply_to_comment(
        self,
        comment_url: str,
        body: str,
        *,
        subreddit: str | None = None,
        parent_comment_id: str | None = None,
        parent_post_id: str | None = None,
        action_id: UUID | None = None,
    ) -> ActionResult:
        """Reply through the UI scoped to a Reddit comment or ``t1_`` fullname."""

        if not body.strip():
            raise PlatformActionRejectedError("Reddit reply body is required")
        subreddit_hint = (
            self._require_allowed_subreddit(subreddit)
            if subreddit is not None
            else None
        )
        expected_comment = _optional_fullname(
            parent_comment_id,
            expected_kind="t1",
            field_name="parent_comment_id",
        )
        expected_post = _optional_fullname(
            parent_post_id,
            expected_kind="t3",
            field_name="parent_post_id",
        )
        resolved_action_id = action_id or uuid4()

        async def operation(
            page: Page,
            submission_state: _SubmissionState,
        ) -> ActionResult:
            target = await self._resolve_target(
                page,
                comment_url,
                expected_kind="t1",
                subreddit_hint=subreddit_hint,
                expected_fullname=expected_comment,
                expected_post_fullname=expected_post,
            )
            target_thing = self._thing_locator(page, target.fullname)
            await target_thing.wait_for(
                state="visible", timeout=_INTERACTION_TIMEOUT_MS
            )

            reply_button = target_thing.locator(_REPLY_BUTTON_SELECTOR).first
            await reply_button.wait_for(
                state="visible", timeout=_INTERACTION_TIMEOUT_MS
            )
            await reply_button.click()

            reply_form = target_thing.locator(_REPLY_FORM_SELECTOR).first
            await reply_form.wait_for(
                state="visible", timeout=_INTERACTION_TIMEOUT_MS
            )
            await self._verify_parent_field(reply_form, target.fullname)

            textarea = reply_form.locator(_COMMENT_TEXT_SELECTOR).first
            await textarea.wait_for(
                state="visible", timeout=_INTERACTION_TIMEOUT_MS
            )
            await textarea.fill(body)

            response, published = await self._submit_form(
                page,
                reply_form,
                endpoint="/api/comment",
                subreddit=target.subreddit,
                expected_kind="t1",
                excluded_fullnames=frozenset({target.fullname}),
                expected_parent_fullname=target.fullname,
                submission_state=submission_state,
            )
            return ActionResult(
                action_id=resolved_action_id,
                success=True,
                external_content_id=published.fullname,
                external_content_url=published.permalink,
                message="Reddit reply created",
                metadata={
                    "subreddit": published.subreddit,
                    "parent_post_id": target.post_fullname,
                    "parent_comment_id": target.fullname,
                    "response_status": response.status,
                },
            )

        return await self._run_operation(
            resolved_action_id,
            ActionType.REDDIT_REPLY,
            operation,
        )

    async def _run_operation(
        self,
        action_id: UUID,
        action_type: ActionType,
        operation: Callable[[Page, _SubmissionState], Awaitable[ActionResult]],
    ) -> ActionResult:
        self._log.info(
            "reddit_action_started",
            action_id=str(action_id),
            action_type=action_type.value,
        )
        submission_state = _SubmissionState()
        try:
            async with self._browser_manager.get_page(
                Platform.REDDIT,
                self._account_name,
            ) as page:
                try:
                    result = await operation(page, submission_state)
                except BrowserInteractionError as error:
                    if submission_state.dispatched:
                        error.retryable = False
                    capture = await self._capture_failure(page, action_id)
                    error.attach_capture(capture)
                    raise
                except PlatformError as error:
                    capture = await self._capture_failure(page, action_id)
                    if (
                        submission_state.dispatched
                        and submission_state.api_rejection is None
                    ):
                        raise BrowserInteractionError(
                            "Reddit submission was dispatched but its outcome could not be confirmed",
                            current_url=_safe_page_url(
                                capture.page_url if capture is not None else page.url
                            ),
                            screenshot_path=(
                                capture.path if capture is not None else None
                            ),
                            retryable=False,
                        ) from error
                    raise
                except PlaywrightError as error:
                    capture = await self._capture_failure(page, action_id)
                    raise BrowserInteractionError(
                        "Reddit browser interaction failed",
                        current_url=_safe_page_url(
                            capture.page_url if capture is not None else page.url
                        ),
                        screenshot_path=(
                            capture.path if capture is not None else None
                        ),
                        retryable=not submission_state.dispatched,
                    ) from error
                except Exception as error:
                    capture = await self._capture_failure(page, action_id)
                    raise BrowserInteractionError(
                        "Reddit browser operation failed",
                        current_url=_safe_page_url(
                            capture.page_url if capture is not None else page.url
                        ),
                        screenshot_path=(
                            capture.path if capture is not None else None
                        ),
                        retryable=not submission_state.dispatched,
                    ) from error
        except PlatformError as error:
            if (
                submission_state.api_rejection is not None
                and error is not submission_state.api_rejection
            ):
                raise submission_state.api_rejection from error
            if (
                submission_state.dispatched
                and submission_state.api_rejection is None
                and not isinstance(error, BrowserInteractionError)
            ):
                raise BrowserInteractionError(
                    "Reddit submission was dispatched but its outcome could not be confirmed",
                    retryable=False,
                ) from error
            raise
        except BrowserManagerError as error:
            if submission_state.api_rejection is not None:
                raise submission_state.api_rejection from error
            raise BrowserInteractionError(
                "Reddit browser session failed",
                retryable=not submission_state.dispatched,
            ) from error
        except Exception as error:
            if submission_state.api_rejection is not None:
                raise submission_state.api_rejection from error
            raise BrowserInteractionError(
                "Reddit browser session failed",
                retryable=not submission_state.dispatched,
            ) from error

        self._log.info(
            "reddit_action_succeeded",
            action_id=str(action_id),
            action_type=action_type.value,
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
                "reddit_failure_capture_failed",
                action_id=str(action_id),
                error_type=type(error).__name__,
            )
            return None

    def _require_allowed_subreddit(self, subreddit: str) -> str:
        try:
            normalized = _normalize_subreddit(subreddit)
        except ValueError as error:
            raise PlatformActionRejectedError(
                "Subreddit must be a valid Reddit community name"
            ) from error
        if normalized.casefold() not in self._allowed_subreddits:
            raise SubredditNotAllowedError(
                f"Subreddit 'r/{normalized}' is not allowed for this account"
            )
        return normalized

    async def _navigate(
        self,
        page: Page,
        url: str,
        *,
        purpose: _PagePurpose,
        subreddit: str | None,
    ) -> Response | None:
        response = await page.goto(
            url,
            wait_until="domcontentloaded",
            timeout=_NAVIGATION_TIMEOUT_MS,
        )
        await self._raise_for_page(
            page,
            response,
            purpose=purpose,
            subreddit=subreddit,
        )
        return response

    async def _raise_for_page(
        self,
        page: Page,
        response: Response | None,
        *,
        purpose: _PagePurpose,
        subreddit: str | None,
    ) -> None:
        parsed_url = _split_safe_url(page.url)
        if (
            parsed_url is None
            or parsed_url.hostname not in _REDDIT_TARGET_HOSTS
            or not _uses_scheme_default_port(parsed_url)
        ):
            raise PlatformAuthenticationError(
                "Reddit redirected the browser to an unexpected origin"
            )

        platform_text = (
            await _visible_text(page, _PLATFORM_MESSAGE_SELECTOR)
        ).casefold()

        if await _has_visible(page, _CHALLENGE_SELECTOR) or _contains_signal(
            platform_text, _CHALLENGE_SIGNALS
        ):
            raise RedditChallengeError(
                "Reddit requires a human challenge or account verification"
            )
        if await _has_visible(page, _LOGIN_SELECTOR) or _contains_signal(
            platform_text, _AUTHENTICATION_SIGNALS
        ):
            raise PlatformAuthenticationError(
                "Reddit account session is not authenticated"
            )
        if _contains_signal(platform_text, _RATE_LIMIT_SIGNALS) or (
            response is not None and response.status == 429
        ):
            raise PlatformRateLimitError(
                "Reddit rate limit reached",
                retry_after_seconds=_retry_after_seconds(
                    platform_text,
                    response.headers if response is not None else {},
                ),
            )

        subreddit_unavailable = _contains_signal(
            platform_text, _SUBREDDIT_UNAVAILABLE_SIGNALS
        )
        if subreddit_unavailable and subreddit is not None:
            raise SubredditUnavailableError(
                f"Subreddit 'r/{subreddit}' is unavailable"
            )

        not_found = await _has_visible(page, _ERROR_PAGE_SELECTOR) or _contains_signal(
            platform_text, _NOT_FOUND_SIGNALS
        )
        if not_found:
            if purpose == "submit" and subreddit is not None:
                raise SubredditUnavailableError(
                    f"Subreddit 'r/{subreddit}' is unavailable"
                )
            raise RedditTargetNotFoundError("Reddit target was not found")

        if response is None:
            return
        if response.status in {401, 407}:
            raise PlatformAuthenticationError(
                "Reddit account session is not authenticated"
            )
        if response.status == 403:
            if purpose == "submit" and subreddit is not None:
                raise SubredditUnavailableError(
                    f"Subreddit 'r/{subreddit}' is unavailable"
                )
            raise PlatformActionRejectedError(
                "Reddit rejected access to the requested target"
            )
        if response.status == 404:
            if purpose == "submit" and subreddit is not None:
                raise SubredditUnavailableError(
                    f"Subreddit 'r/{subreddit}' is unavailable"
                )
            raise RedditTargetNotFoundError("Reddit target was not found")
        if response.status >= 500:
            raise PlatformUnavailableError(
                "Reddit is temporarily unavailable"
            )

    async def _prepare_post_form(self, form: Locator, *, is_text: bool) -> None:
        content_selector = _POST_TEXT_SELECTOR if is_text else _POST_URL_SELECTOR
        content_input = form.locator(content_selector).first
        await content_input.wait_for(
            state="visible", timeout=_INTERACTION_TIMEOUT_MS
        )

        kind_input = form.locator(_POST_KIND_SELECTOR).first
        if await kind_input.count():
            kind_value = (await kind_input.get_attribute("value") or "").casefold()
            expected_values = {"self", "selftext"} if is_text else {"link"}
            if kind_value and kind_value not in expected_values:
                raise BrowserInteractionError(
                    "Old Reddit displayed the wrong post submission form"
                )

    async def _verify_parent_field(
        self,
        form: Locator,
        expected_fullname: str,
    ) -> None:
        parent_input = form.locator(_PARENT_FULLNAME_SELECTOR).first
        if not await parent_input.count():
            return
        parent_value = (await parent_input.input_value()).strip().casefold()
        if parent_value and parent_value != expected_fullname:
            raise BrowserInteractionError(
                "Reddit reply form does not target the requested parent"
            )

    async def _submit_form(
        self,
        page: Page,
        form: Locator,
        *,
        endpoint: str,
        subreddit: str,
        expected_kind: _TargetKind,
        excluded_fullnames: frozenset[str],
        expected_parent_fullname: str | None,
        submission_state: _SubmissionState,
    ) -> tuple[Response, _PublishedContent]:
        submit = form.locator(_FORM_SUBMIT_SELECTOR).first
        await submit.wait_for(state="visible", timeout=_INTERACTION_TIMEOUT_MS)
        if not await submit.is_enabled():
            raise PlatformActionRejectedError(
                "Reddit submission form is disabled"
            )

        try:
            async with page.expect_response(
                lambda candidate: _is_expected_form_response(candidate, endpoint),
                timeout=_NAVIGATION_TIMEOUT_MS,
            ) as response_info:
                # This is the point of no safe retry: the click may have reached
                # Reddit even if every subsequent browser observation fails.
                submission_state.dispatched = True
                await submit.click()
            response = await response_info.value
            payload = await _response_payload(response)
        except Exception as error:
            if not submission_state.dispatched:
                raise
            raise BrowserInteractionError(
                "Reddit submission was dispatched but its outcome could not be confirmed",
                current_url=_safe_page_url(page.url),
                retryable=False,
            ) from error

        try:
            self._raise_for_api_response(response, payload, subreddit=subreddit)
        except PlatformError as error:
            # A matched API response can positively reject the action; preserve
            # the existing typed rejection and its retry semantics.
            if not isinstance(error, BrowserInteractionError):
                submission_state.api_rejection = error
            raise
        except Exception as error:
            raise BrowserInteractionError(
                "Reddit submission was dispatched but its outcome could not be confirmed",
                current_url=_safe_page_url(page.url),
                retryable=False,
            ) from error

        try:
            await page.wait_for_load_state(
                "domcontentloaded", timeout=_NAVIGATION_TIMEOUT_MS
            )
            await self._raise_for_page(
                page,
                None,
                purpose="created",
                subreddit=subreddit,
            )
            published = await self._resolve_published_content(
                page,
                payload,
                expected_kind=expected_kind,
                excluded_fullnames=excluded_fullnames,
                subreddit_hint=subreddit,
                expected_parent_fullname=expected_parent_fullname,
            )
        except Exception as error:
            raise BrowserInteractionError(
                "Reddit submission was dispatched but its outcome could not be confirmed",
                current_url=_safe_page_url(page.url),
                retryable=False,
            ) from error
        return response, published

    def _raise_for_api_response(
        self,
        response: Response,
        payload: Mapping[str, Any],
        *,
        subreddit: str,
    ) -> None:
        location = response.headers.get("location", "")
        location_path = urlsplit(location).path.casefold() if location else ""
        if "/login" in location_path:
            raise PlatformAuthenticationError(
                "Reddit account session is not authenticated"
            )
        if "/challenge" in location_path or "/verify" in location_path:
            raise RedditChallengeError(
                "Reddit requires a human challenge or account verification"
            )

        errors = _api_errors(payload)
        if errors:
            codes = {code for code, _ in errors}
            messages = " ".join(message for _, message in errors)
            if codes & _RATE_LIMIT_CODES:
                raise PlatformRateLimitError(
                    "Reddit rate limit reached",
                    retry_after_seconds=_retry_after_seconds(
                        messages,
                        response.headers,
                    ),
                )
            if codes & _CHALLENGE_CODES:
                raise RedditChallengeError(
                    "Reddit requires a human challenge or account verification"
                )
            if codes & _AUTHENTICATION_CODES:
                raise PlatformAuthenticationError(
                    "Reddit account session is not authenticated"
                )
            if codes & _SUBREDDIT_UNAVAILABLE_CODES:
                raise SubredditUnavailableError(
                    f"Subreddit 'r/{subreddit}' is unavailable"
                )
            if codes & _TARGET_REJECTION_CODES:
                raise PlatformActionRejectedError(
                    "Reddit target does not accept this action"
                )
            safe_codes = ", ".join(sorted(codes)) or "UNKNOWN"
            raise PlatformActionRejectedError(
                f"Reddit rejected the action ({safe_codes})"
            )

        if response.status == 429:
            raise PlatformRateLimitError(
                "Reddit rate limit reached",
                retry_after_seconds=_retry_after_seconds("", response.headers),
            )
        if response.status in {401, 407}:
            raise PlatformAuthenticationError(
                "Reddit account session is not authenticated"
            )
        if response.status == 403:
            raise PlatformActionRejectedError("Reddit rejected the action")
        if response.status == 404:
            raise RedditTargetNotFoundError("Reddit target was not found")
        if response.status >= 500:
            raise BrowserInteractionError(
                "Reddit submission received a server error after dispatch",
                retryable=False,
            )
        if response.status >= 400:
            raise PlatformActionRejectedError("Reddit rejected the action")

    async def _resolve_target(
        self,
        page: Page,
        target: str,
        *,
        expected_kind: _TargetKind,
        subreddit_hint: str | None,
        expected_fullname: str | None,
        expected_post_fullname: str | None,
    ) -> _ResolvedTarget:
        reference = _parse_target_reference(target, expected_kind=expected_kind)
        if expected_fullname is not None and reference.fullname != expected_fullname:
            raise PlatformActionRejectedError(
                "Reddit target does not match the supplied parent fullname"
            )
        if (
            expected_post_fullname is not None
            and reference.post_fullname != expected_post_fullname
            and not (
                expected_kind == "t1"
                and reference.navigation_url.endswith(f"/{reference.fullname}/")
            )
        ):
            raise PlatformActionRejectedError(
                "Reddit target does not match the supplied parent post"
            )

        known_subreddit = reference.subreddit
        if known_subreddit is not None:
            known_subreddit = self._require_allowed_subreddit(known_subreddit)
            _require_matching_subreddit(known_subreddit, subreddit_hint)

        await self._navigate(
            page,
            reference.navigation_url,
            purpose="target",
            subreddit=known_subreddit or subreddit_hint,
        )
        thing = self._thing_locator(page, reference.fullname)
        try:
            await thing.wait_for(
                state="attached", timeout=_INTERACTION_TIMEOUT_MS
            )
        except PlaywrightTimeoutError as error:
            raise RedditTargetNotFoundError(
                "Reddit target was not found"
            ) from error

        permalink = await self._permalink_for_thing(
            page,
            thing,
            reference.fullname,
        )
        if permalink is None:
            permalink = reference.supplied_permalink
        if permalink is None:
            raise RedditTargetNotFoundError(
                "Reddit target has no resolvable permalink"
            )

        if not _same_reddit_path(page.url, permalink):
            await self._navigate(
                page,
                _as_old_reddit_url(permalink),
                purpose="target",
                subreddit=known_subreddit or subreddit_hint,
            )
            thing = self._thing_locator(page, reference.fullname)
            try:
                await thing.wait_for(
                    state="attached", timeout=_INTERACTION_TIMEOUT_MS
                )
            except PlaywrightTimeoutError as error:
                raise RedditTargetNotFoundError(
                    "Reddit target was not found at its permalink"
                ) from error

        permalink = (
            await self._permalink_for_thing(page, thing, reference.fullname)
            or permalink
        )
        resolved_subreddit = await self._subreddit_for_thing(
            page,
            thing,
            fallback=known_subreddit,
        )
        if resolved_subreddit is None:
            raise PlatformActionRejectedError(
                "Reddit target subreddit could not be verified against the allowlist"
            )
        resolved_subreddit = self._require_allowed_subreddit(resolved_subreddit)
        _require_matching_subreddit(resolved_subreddit, subreddit_hint)

        post_fullname = await self._post_fullname_for_thing(
            page,
            thing,
            fallback=reference.post_fullname,
        )
        if expected_post_fullname is not None and post_fullname != expected_post_fullname:
            raise PlatformActionRejectedError(
                "Reddit target does not match the supplied parent post"
            )

        return _ResolvedTarget(
            fullname=reference.fullname,
            post_fullname=post_fullname,
            comment_fullname=reference.comment_fullname,
            subreddit=resolved_subreddit,
            permalink=permalink,
        )

    async def _resolve_published_content(
        self,
        page: Page,
        payload: Mapping[str, Any],
        *,
        expected_kind: _TargetKind,
        excluded_fullnames: frozenset[str],
        subreddit_hint: str,
        expected_parent_fullname: str | None,
    ) -> _PublishedContent:
        evidence = _extract_created_response(
            payload,
            expected_kind=expected_kind,
            excluded=excluded_fullnames,
        )
        if evidence is None:
            raise BrowserInteractionError(
                "Reddit accepted the action but did not return a unique content fullname",
                retryable=False,
            )

        fullname = evidence.fullname
        hinted_permalink = evidence.permalink
        thing = self._thing_locator(page, fullname)
        permalink: str | None = None
        try:
            await thing.wait_for(
                state="attached", timeout=_CREATED_CONTENT_TIMEOUT_MS
            )
        except PlaywrightTimeoutError:
            pass
        else:
            permalink = await self._permalink_for_thing(page, thing, fullname)

        if permalink is None and hinted_permalink is not None:
            await self._navigate(
                page,
                _as_old_reddit_url(hinted_permalink),
                purpose="created",
                subreddit=subreddit_hint,
            )
            thing = self._thing_locator(page, fullname)
            try:
                await thing.wait_for(
                    state="attached", timeout=_INTERACTION_TIMEOUT_MS
                )
            except PlaywrightTimeoutError:
                pass
            else:
                permalink = (
                    await self._permalink_for_thing(page, thing, fullname)
                    or hinted_permalink
                )

        if permalink is None:
            await self._navigate(
                page,
                f"{_OLD_REDDIT_ORIGIN}/by_id/{quote(fullname, safe='_')}/",
                purpose="created",
                subreddit=subreddit_hint,
            )
            thing = self._thing_locator(page, fullname)
            try:
                await thing.wait_for(
                    state="attached", timeout=_INTERACTION_TIMEOUT_MS
                )
            except PlaywrightTimeoutError as error:
                raise BrowserInteractionError(
                    "Reddit accepted the action but the created content could not be resolved",
                    retryable=False,
                ) from error
            permalink = await self._permalink_for_thing(page, thing, fullname)

        if permalink is None:
            raise BrowserInteractionError(
                "Reddit accepted the action but did not expose a canonical permalink",
                retryable=False,
            )

        if expected_parent_fullname is not None:
            observed_parents = set(
                await self._parent_fullnames_for_thing(
                    thing,
                    expected_kind=(
                        "t1"
                        if expected_parent_fullname.startswith("t1_")
                        else "t3"
                    ),
                )
            )
            if evidence.parent_fullname is not None:
                observed_parents.add(evidence.parent_fullname)
            if not observed_parents or any(
                parent != expected_parent_fullname for parent in observed_parents
            ):
                raise BrowserInteractionError(
                    "Reddit accepted the action but its parent could not be verified",
                    retryable=False,
                )

        resolved_subreddit = await self._subreddit_for_thing(
            page,
            thing,
            fallback=subreddit_hint,
        )
        if resolved_subreddit is None:
            raise BrowserInteractionError(
                "Reddit accepted the action but its subreddit could not be verified",
                retryable=False,
            )
        resolved_subreddit = self._require_allowed_subreddit(resolved_subreddit)
        _require_matching_subreddit(resolved_subreddit, subreddit_hint)

        post_fullname = await self._post_fullname_for_thing(
            page,
            thing,
            fallback=(
                fullname
                if expected_kind == "t3"
                else _fullname_from_permalink(permalink, expected_kind="t3")
            ),
        )
        return _PublishedContent(
            fullname=fullname,
            permalink=permalink,
            subreddit=resolved_subreddit,
            post_fullname=post_fullname,
        )

    def _thing_locator(self, page: Page, fullname: str) -> Locator:
        return page.locator(
            _THING_SELECTOR_TEMPLATE.format(fullname=fullname)
        ).first

    async def _permalink_for_thing(
        self,
        page: Page,
        thing: Locator,
        fullname: str,
    ) -> str | None:
        data_permalink = await thing.get_attribute("data-permalink")
        permalink = _canonical_permalink(
            data_permalink,
            expected_fullname=fullname,
        )
        if permalink is not None:
            return permalink

        links = thing.locator(_PERMALINK_SELECTOR)
        for index in range(await links.count()):
            href = await links.nth(index).get_attribute("href")
            permalink = _canonical_permalink(
                href,
                expected_fullname=fullname,
            )
            if permalink is not None:
                return permalink

        return _canonical_permalink(page.url, expected_fullname=fullname)

    async def _subreddit_for_thing(
        self,
        page: Page,
        thing: Locator,
        *,
        fallback: str | None,
    ) -> str | None:
        candidates = (
            await thing.get_attribute("data-subreddit"),
            await thing.get_attribute("data-subreddit-prefixed"),
            _subreddit_from_url(page.url),
            fallback,
        )
        for candidate in candidates:
            if candidate is None:
                continue
            try:
                return _normalize_subreddit(candidate)
            except ValueError:
                continue

        post = page.locator(_POST_THING_SELECTOR).first
        if await post.count():
            candidate = await post.get_attribute("data-subreddit")
            if candidate is not None:
                try:
                    return _normalize_subreddit(candidate)
                except ValueError:
                    return None
        return None

    async def _post_fullname_for_thing(
        self,
        page: Page,
        thing: Locator,
        *,
        fallback: str | None,
    ) -> str:
        candidates = (
            await thing.get_attribute("data-link-id"),
            fallback,
            _fullname_from_permalink(page.url, expected_kind="t3"),
        )
        for candidate in candidates:
            normalized = _try_fullname(candidate, expected_kind="t3")
            if normalized is not None:
                return normalized

        post = page.locator(_POST_THING_SELECTOR).first
        if await post.count():
            normalized = _try_fullname(
                await post.get_attribute("data-fullname"),
                expected_kind="t3",
            )
            if normalized is not None:
                return normalized
        raise BrowserInteractionError(
            "Reddit page did not expose the parent post fullname"
        )

    async def _parent_fullnames_for_thing(
        self,
        thing: Locator,
        *,
        expected_kind: _TargetKind,
    ) -> frozenset[str]:
        raw_candidates = [
            await thing.get_attribute("data-parent-fullname"),
            await thing.get_attribute("data-parent-id"),
        ]
        if expected_kind == "t3":
            raw_candidates.append(await thing.get_attribute("data-link-id"))
        else:
            parent = thing.locator(_PARENT_COMMENT_ANCESTOR_SELECTOR).first
            if await parent.count():
                raw_candidates.append(await parent.get_attribute("data-fullname"))

        return frozenset(
            candidate
            for raw_candidate in raw_candidates
            if (candidate := _try_any_fullname(raw_candidate)) is not None
        )


def _normalize_subreddit(value: str) -> str:
    normalized = value.strip()
    if normalized.casefold().startswith("r/"):
        normalized = normalized[2:]
    if not _SUBREDDIT_RE.fullmatch(normalized):
        raise ValueError("invalid subreddit")
    return normalized


def _optional_fullname(
    value: str | None,
    *,
    expected_kind: _TargetKind,
    field_name: str,
) -> str | None:
    if value is None:
        return None
    normalized = _try_fullname(value, expected_kind=expected_kind)
    if normalized is None:
        raise PlatformActionRejectedError(
            f"{field_name} must be a Reddit {expected_kind}_ fullname"
        )
    return normalized


def _try_fullname(
    value: str | None,
    *,
    expected_kind: _TargetKind,
) -> str | None:
    if value is None:
        return None
    match = _FULLNAME_RE.fullmatch(value.strip())
    if match is None or match.group("kind").casefold() != expected_kind:
        return None
    return f"{expected_kind}_{match.group('id').casefold()}"


def _parse_target_reference(
    value: str,
    *,
    expected_kind: _TargetKind,
) -> _TargetReference:
    target = value.strip()
    fullname = _try_fullname(target, expected_kind=expected_kind)
    if fullname is not None:
        if expected_kind == "t3":
            return _TargetReference(
                fullname=fullname,
                post_fullname=fullname,
                comment_fullname=None,
                subreddit=None,
                navigation_url=(
                    f"{_OLD_REDDIT_ORIGIN}/by_id/{quote(fullname, safe='_')}/"
                ),
                supplied_permalink=None,
            )
        return _TargetReference(
            fullname=fullname,
            post_fullname="",
            comment_fullname=fullname,
            subreddit=None,
            navigation_url=(
                f"{_OLD_REDDIT_ORIGIN}/by_id/{quote(fullname, safe='_')}/"
            ),
            supplied_permalink=None,
        )

    parsed = _split_safe_url(target)
    if parsed is None or parsed.hostname not in _REDDIT_TARGET_HOSTS:
        raise PlatformActionRejectedError(
            "Reddit target must be a reddit.com URL or Reddit fullname"
        )
    if parsed.username is not None or parsed.password is not None:
        raise PlatformActionRejectedError(
            "Reddit target URL must not contain credentials"
        )
    if not _uses_scheme_default_port(parsed):
        raise PlatformActionRejectedError(
            "Reddit target URL must use port 80 for HTTP or port 443 for HTTPS"
        )

    path_match = _COMMENTS_PATH_RE.fullmatch(unquote(parsed.path))
    if path_match is None:
        raise PlatformActionRejectedError(
            "Reddit target URL must be a post or comment permalink"
        )

    subreddit = path_match.group("subreddit")
    post_id = path_match.group("post_id").casefold()
    comment_id = path_match.group("comment_id")
    post_fullname = f"t3_{post_id}"
    comment_fullname = (
        f"t1_{comment_id.casefold()}" if comment_id is not None else None
    )
    if expected_kind == "t1" and comment_fullname is None:
        raise PlatformActionRejectedError(
            "Reddit reply target URL must identify a comment"
        )

    slug = path_match.group("slug") or "-"
    comments_path = _build_comments_path(
        subreddit=subreddit,
        post_id=post_id,
        slug=slug,
        comment_id=(comment_id if expected_kind == "t1" else None),
    )
    target_fullname = (
        comment_fullname if expected_kind == "t1" else post_fullname
    )
    if target_fullname is None:
        raise PlatformActionRejectedError("Reddit target could not be identified")

    canonical_permalink = urlunsplit(
        ("https", "www.reddit.com", comments_path, "", "")
    )
    return _TargetReference(
        fullname=target_fullname,
        post_fullname=post_fullname,
        comment_fullname=comment_fullname,
        subreddit=subreddit,
        navigation_url=urlunsplit(
            ("https", "old.reddit.com", comments_path, "", "")
        ),
        supplied_permalink=canonical_permalink,
    )


def _build_comments_path(
    *,
    subreddit: str | None,
    post_id: str,
    slug: str,
    comment_id: str | None,
) -> str:
    prefix = (
        f"/r/{quote(subreddit, safe='_')}"
        if subreddit is not None
        else ""
    )
    path = (
        f"{prefix}/comments/{quote(post_id, safe='')}/"
        f"{quote(slug, safe='-_~')}/"
    )
    if comment_id is not None:
        path += f"{quote(comment_id.casefold(), safe='')}/"
    return path


def _canonical_permalink(
    value: str | None,
    *,
    expected_fullname: str | None,
) -> str | None:
    if not value:
        return None
    decoded = html.unescape(value.strip())
    absolute = urljoin(f"{_CANONICAL_REDDIT_ORIGIN}/", decoded)
    parsed = _split_safe_url(absolute)
    if (
        parsed is None
        or parsed.hostname not in _REDDIT_TARGET_HOSTS
        or not _uses_scheme_default_port(parsed)
    ):
        return None
    if parsed.username is not None or parsed.password is not None:
        return None
    path_match = _COMMENTS_PATH_RE.fullmatch(unquote(parsed.path))
    if path_match is None:
        return None

    if expected_fullname is not None:
        match = _FULLNAME_RE.fullmatch(expected_fullname)
        if match is None:
            return None
        expected_id = match.group("id").casefold()
        if match.group("kind").casefold() == "t3":
            actual_id = path_match.group("post_id").casefold()
        else:
            comment_id = path_match.group("comment_id")
            actual_id = comment_id.casefold() if comment_id is not None else ""
        if actual_id != expected_id:
            return None

    path = parsed.path.rstrip("/") + "/"
    return urlunsplit(("https", "www.reddit.com", path, "", ""))


def _fullname_from_permalink(
    value: str,
    *,
    expected_kind: _TargetKind,
) -> str | None:
    parsed = _split_safe_url(value)
    if parsed is None or parsed.hostname not in _REDDIT_TARGET_HOSTS:
        return None
    path_match = _COMMENTS_PATH_RE.fullmatch(unquote(parsed.path))
    if path_match is None:
        return None
    if expected_kind == "t3":
        return f"t3_{path_match.group('post_id').casefold()}"
    comment_id = path_match.group("comment_id")
    return f"t1_{comment_id.casefold()}" if comment_id is not None else None


def _subreddit_from_url(value: str) -> str | None:
    parsed = _split_safe_url(value)
    if parsed is None or parsed.hostname not in _REDDIT_TARGET_HOSTS:
        return None
    path_match = _COMMENTS_PATH_RE.fullmatch(unquote(parsed.path))
    return path_match.group("subreddit") if path_match is not None else None


def _as_old_reddit_url(permalink: str) -> str:
    parsed = urlsplit(permalink)
    return urlunsplit(("https", "old.reddit.com", parsed.path, "", ""))


def _same_reddit_path(left: str, right: str) -> bool:
    left_parsed = _split_safe_url(left)
    right_parsed = _split_safe_url(right)
    if left_parsed is None or right_parsed is None:
        return False
    if (
        left_parsed.hostname not in _REDDIT_TARGET_HOSTS
        or right_parsed.hostname not in _REDDIT_TARGET_HOSTS
    ):
        return False
    return unquote(left_parsed.path).rstrip("/").casefold() == unquote(
        right_parsed.path
    ).rstrip("/").casefold()


def _require_matching_subreddit(actual: str, expected: str | None) -> None:
    if expected is not None and actual.casefold() != expected.casefold():
        raise PlatformActionRejectedError(
            "Reddit target subreddit does not match the action"
        )


def _split_safe_url(value: str):
    try:
        parsed = urlsplit(value.strip())
        _ = parsed.port
    except (AttributeError, ValueError):
        return None
    if parsed.scheme.casefold() not in {"http", "https"} or not parsed.hostname:
        return None
    return parsed

def _uses_scheme_default_port(parsed) -> bool:
    port = parsed.port
    if port is None:
        return True
    scheme = parsed.scheme.casefold()
    return (scheme == "http" and port == 80) or (
        scheme == "https" and port == 443
    )


def _normalize_link_post_url(value: str) -> str:
    parsed = _split_safe_url(value)
    if parsed is None:
        raise ValueError("link URL must be an absolute HTTP(S) URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("link URL must not contain credentials")
    if not _uses_scheme_default_port(parsed):
        raise ValueError("link URL must use its scheme's default port")
    return normalize_target_url(value)


def _safe_page_url(value: str | None) -> str | None:
    if not value:
        return None
    parsed = _split_safe_url(value)
    if parsed is None:
        return None
    hostname = parsed.hostname.casefold() if parsed.hostname else ""
    path = parsed.path if hostname in _REDDIT_TARGET_HOSTS else "/"
    return urlunsplit((parsed.scheme.casefold(), hostname, path, "", ""))


def _contains_signal(text: str, signals: Sequence[str]) -> bool:
    return any(signal in text for signal in signals)


async def _has_visible(page: Page, selector: str) -> bool:
    locator = page.locator(selector)
    for index in range(await locator.count()):
        if await locator.nth(index).is_visible():
            return True
    return False

async def _visible_text(page: Page, selector: str) -> str:
    visible_text: list[str] = []
    locator = page.locator(selector)
    for index in range(await locator.count()):
        candidate = locator.nth(index)
        if not await candidate.is_visible():
            continue
        text = (await candidate.inner_text(timeout=_INTERACTION_TIMEOUT_MS)).strip()
        if text:
            visible_text.append(text)
    return " ".join(visible_text)


def _is_expected_form_response(response: Response, endpoint: str) -> bool:
    parsed = _split_safe_url(response.url)
    return bool(
        response.request.method.casefold() == "post"
        and parsed is not None
        and parsed.hostname in _REDDIT_TARGET_HOSTS
        and _uses_scheme_default_port(parsed)
        and parsed.path.rstrip("/").casefold() == endpoint.casefold()
    )


async def _response_payload(response: Response) -> dict[str, Any]:
    try:
        payload = await response.json()
    except (PlaywrightError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _api_errors(payload: Mapping[str, Any]) -> list[tuple[str, str]]:
    json_payload = payload.get("json")
    if not isinstance(json_payload, Mapping):
        return []
    raw_errors = json_payload.get("errors")
    if not isinstance(raw_errors, Sequence) or isinstance(raw_errors, (str, bytes)):
        return []

    errors: list[tuple[str, str]] = []
    for raw_error in raw_errors:
        if not isinstance(raw_error, Sequence) or isinstance(raw_error, (str, bytes)):
            continue
        code = _safe_error_code(raw_error[0] if len(raw_error) > 0 else "UNKNOWN")
        message = str(raw_error[1]) if len(raw_error) > 1 else ""
        errors.append((code, message))
    return errors


def _safe_error_code(value: object) -> str:
    normalized = re.sub(r"[^A-Z0-9_]", "_", str(value).upper())
    return normalized[:64] or "UNKNOWN"


def _retry_after_seconds(
    text: str,
    headers: Mapping[str, str],
) -> float | None:
    retry_after = headers.get("retry-after")
    if retry_after is not None:
        try:
            value = float(retry_after)
        except ValueError:
            pass
        else:
            return max(0.0, value)

    match = _RETRY_AFTER_RE.search(text)
    if match is None:
        return None
    amount = float(match.group("amount"))
    multiplier = {
        "second": 1.0,
        "minute": 60.0,
        "hour": 3600.0,
    }[match.group("unit").casefold()]
    return amount * multiplier


def _extract_created_response(
    payload: Mapping[str, Any],
    *,
    expected_kind: _TargetKind,
    excluded: frozenset[str],
) -> _CreatedResponseEvidence | None:
    json_payload = payload.get("json")
    data = json_payload.get("data") if isinstance(json_payload, Mapping) else None
    if not isinstance(data, Mapping):
        return None

    records: list[_CreatedResponseEvidence] = []
    try:
        direct_record = _created_response_record(
            data,
            expected_kind=expected_kind,
            declared_kind=None,
            excluded=excluded,
        )
        if direct_record is not None:
            records.append(direct_record)

        things = data.get("things")
        if isinstance(things, Sequence) and not isinstance(things, (str, bytes)):
            for thing in things:
                if not isinstance(thing, Mapping):
                    continue
                declared_kind = _string_value(thing.get("kind"))
                thing_data = thing.get("data")
                if declared_kind is None or not isinstance(thing_data, Mapping):
                    continue
                record = _created_response_record(
                    thing_data,
                    expected_kind=expected_kind,
                    declared_kind=declared_kind.casefold(),
                    excluded=excluded,
                )
                if record is not None:
                    records.append(record)
    except ValueError:
        return None

    fullnames = {record.fullname for record in records}
    if len(fullnames) != 1:
        return None

    parents = {
        record.parent_fullname
        for record in records
        if record.parent_fullname is not None
    }
    if len(parents) > 1:
        return None

    permalinks = sorted(
        {
            record.permalink
            for record in records
            if record.permalink is not None
        }
    )
    return _CreatedResponseEvidence(
        fullname=next(iter(fullnames)),
        permalink=permalinks[0] if permalinks else None,
        parent_fullname=next(iter(parents), None),
    )


def _created_response_record(
    data: Mapping[str, Any],
    *,
    expected_kind: _TargetKind,
    declared_kind: str | None,
    excluded: frozenset[str],
) -> _CreatedResponseEvidence | None:
    if declared_kind is not None and declared_kind != expected_kind:
        return None

    fullnames: set[str] = set()
    for key in ("name", "fullname"):
        raw_fullname = _string_value(data.get(key))
        if raw_fullname is None:
            continue
        fullname = _try_fullname(raw_fullname, expected_kind=expected_kind)
        if fullname is None:
            raise ValueError("invalid created fullname evidence")
        fullnames.add(fullname)

    raw_id = _string_value(data.get("id"))
    if raw_id is not None and (declared_kind == expected_kind or fullnames):
        if re.fullmatch(r"[a-z0-9]+", raw_id, re.IGNORECASE) is None:
            raise ValueError("invalid created id evidence")
        fullnames.add(f"{expected_kind}_{raw_id.casefold()}")

    raw_permalinks = [
        raw_value
        for key in ("permalink", "url")
        if (raw_value := _string_value(data.get(key))) is not None
    ]

    if len(fullnames) != 1:
        return None
    fullname = next(iter(fullnames))
    if fullname in excluded:
        return None

    permalinks = sorted(
        {
            permalink
            for raw_permalink in raw_permalinks
            if (
                permalink := _canonical_permalink(
                    raw_permalink,
                    expected_fullname=fullname,
                )
            )
            is not None
        }
    )

    raw_parent = data.get("parent_id")
    parent_fullname: str | None = None
    if raw_parent is not None:
        parent_fullname = _try_any_fullname(_string_value(raw_parent))
        if parent_fullname is None:
            raise ValueError("invalid parent fullname evidence")

    return _CreatedResponseEvidence(
        fullname=fullname,
        permalink=permalinks[0] if permalinks else None,
        parent_fullname=parent_fullname,
    )


def _try_any_fullname(value: str | None) -> str | None:
    for expected_kind in ("t1", "t3"):
        fullname = _try_fullname(value, expected_kind=expected_kind)
        if fullname is not None:
            return fullname
    return None


def _string_value(value: object) -> str | None:
    return value if isinstance(value, str) else None
