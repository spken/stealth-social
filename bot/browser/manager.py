"""Serialized ownership of Cloak Browser persistent account profiles."""

from __future__ import annotations

import asyncio
import re
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path, PureWindowsPath
from typing import AsyncContextManager, AsyncIterator, cast
from urllib.parse import urlsplit, urlunsplit
from uuid import UUID

from filelock import FileLock, Timeout as FileLockTimeout
import structlog
from cloakbrowser import ensure_binary, launch_persistent_context_async
from playwright.async_api import BrowserContext, Error as PlaywrightError, Page

from bot.browser.sessions import (
    BrowserSession,
    FailureCapture,
    classify_auth_state,
)
from bot.config import Settings
from bot.models import Platform


logger = structlog.get_logger(__name__)

_SAFE_ACTION_ID = re.compile(r"[^A-Za-z0-9._-]+")
_RELEASE_CHANNEL = "stable"
_PROFILE_LOCK_DIRECTORY = ".profile-locks"
_LOGIN_HOSTS = {
    Platform.X: ("x.com", "twitter.com"),
    Platform.REDDIT: ("reddit.com",),
}


class BrowserManagerError(RuntimeError):
    """Base error for browser session ownership failures."""


class BrowserManagerClosedError(BrowserManagerError):
    """Raised when a session is requested after shutdown begins."""


class BrowserAccountNotConfiguredError(BrowserManagerError, LookupError):
    """Raised when no configured account matches a session request."""


class InvalidSessionProfileError(BrowserManagerError, ValueError):
    """Raised when a configured profile cannot be contained safely."""


def _validated_login_url(platform: Platform, value: str) -> str:
    try:
        candidate = value.strip()
        parsed = urlsplit(candidate)
        host = (parsed.hostname or "").casefold()
        port = parsed.port
    except (AttributeError, ValueError) as error:
        raise BrowserManagerError(
            "login_url must be an absolute HTTPS platform login URL"
        ) from error

    expected_hosts = _LOGIN_HOSTS[platform]
    host_is_allowed = any(
        host == expected or host.endswith(f".{expected}")
        for expected in expected_hosts
    )
    if (
        not candidate
        or parsed.scheme.casefold() != "https"
        or not host_is_allowed
        or parsed.username is not None
        or parsed.password is not None
        or port not in {None, 443}
    ):
        raise BrowserManagerError(
            "login_url must use an HTTPS login origin owned by the platform"
        )
    return candidate


def _sanitized_page_url(value: str | None) -> str | None:
    if not value:
        return None
    try:
        parsed = urlsplit(value.strip())
        host = parsed.hostname
        port = parsed.port
    except (AttributeError, ValueError):
        return None
    scheme = parsed.scheme.casefold()
    if scheme not in {"http", "https"} or not host:
        return None

    safe_host = host.casefold()
    netloc = f"[{safe_host}]" if ":" in safe_host else safe_host
    if port is not None:
        netloc = f"{netloc}:{port}"
    return urlunsplit((scheme, netloc, parsed.path or "/", "", ""))


@dataclass(slots=True)
class _ProfileState:
    platform: Platform
    account_name: str
    profile_name: str
    profile_directory: Path
    profile_lock_path: Path
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    context: BrowserContext | None = None
    page: Page | None = None
    headless: bool | None = None
    closed: bool = True
    close_failed: bool = False
    profile_lock: FileLock | None = None
    owner_task: asyncio.Task[object] | None = None


class BrowserManager:
    """Own persistent contexts and serialize access to each account profile.

    A profile lock is held for the complete lifetime of every ``session()``
    lease. Contexts remain live between leases for reuse, but only the current
    lease holder may operate its page. Shutdown rejects new leases, waits for
    existing holders to release their locks, and closes every owned context.
    """

    def __init__(self, settings: Settings) -> None:
        browser = settings.browser
        self._default_headless = browser.headless
        self._sessions_directory = (
            browser.sessions_directory.expanduser().resolve(strict=False)
        )
        self._screenshots_directory = (
            browser.screenshots_directory.expanduser().resolve(strict=False)
        )
        self._profiles = self._build_profile_states(settings)
        self._shutdown_lock = asyncio.Lock()
        self._preflight_lock = asyncio.Lock()
        self._binary_path: Path | None = None
        self._closed = False

    def _build_profile_states(
        self,
        settings: Settings,
    ) -> dict[tuple[Platform, str], _ProfileState]:
        states: dict[tuple[Platform, str], _ProfileState] = {}
        profile_owners: dict[str, tuple[Platform, str]] = {}
        account_maps = (
            (Platform.X, settings.accounts.x),
            (Platform.REDDIT, settings.accounts.reddit),
        )

        for platform, accounts in account_maps:
            for account_name, account in accounts.items():
                profile_name = account.session_profile
                profile_directory = self._resolve_profile_directory(
                    profile_name
                )
                profile_key = str(profile_directory).casefold()
                previous_owner = profile_owners.get(profile_key)
                if previous_owner is not None:
                    previous_platform, previous_account = previous_owner
                    raise InvalidSessionProfileError(
                        f"{platform.value}/{account_name} and "
                        f"{previous_platform.value}/{previous_account} resolve "
                        "to the same browser session profile"
                    )

                profile_owners[profile_key] = (platform, account_name)
                states[(platform, account_name)] = _ProfileState(
                    platform=platform,
                    account_name=account_name,
                    profile_name=profile_name,
                    profile_directory=profile_directory,
                    profile_lock_path=(
                        self._sessions_directory
                        / _PROFILE_LOCK_DIRECTORY
                        / f"{profile_name}.lock"
                    ),
                )

        return states

    def _resolve_profile_directory(self, profile_name: str) -> Path:
        if (
            not profile_name
            or profile_name in {".", ".."}
            or profile_name.casefold() == _PROFILE_LOCK_DIRECTORY.casefold()
            or "\x00" in profile_name
            or "/" in profile_name
            or "\\" in profile_name
        ):
            raise InvalidSessionProfileError(
                "session_profile must be a single safe directory name"
            )

        windows_path = PureWindowsPath(profile_name)
        if windows_path.is_absolute() or windows_path.drive:
            raise InvalidSessionProfileError(
                "session_profile must not be an absolute or drive-relative path"
            )

        profile_directory = (
            self._sessions_directory / profile_name
        ).resolve(strict=False)
        try:
            profile_directory.relative_to(self._sessions_directory)
        except ValueError as error:
            raise InvalidSessionProfileError(
                "session_profile must remain under browser.sessions_directory"
            ) from error
        return profile_directory

    def _resolve_state(
        self,
        platform: Platform | str,
        account_name: str,
    ) -> _ProfileState:
        try:
            resolved_platform = Platform(platform)
        except (TypeError, ValueError) as error:
            raise BrowserAccountNotConfiguredError(
                f"Unsupported browser platform: {platform!r}"
            ) from error

        try:
            return self._profiles[(resolved_platform, account_name)]
        except KeyError as error:
            raise BrowserAccountNotConfiguredError(
                f"No browser account configured for "
                f"{resolved_platform.value}/{account_name}"
            ) from error

    def resolve_profile(
        self,
        platform: Platform | str,
        account_name: str,
    ) -> tuple[str, Path]:
        """Return the validated profile name and contained directory."""

        state = self._resolve_state(platform, account_name)
        return state.profile_name, state.profile_directory

    async def preflight(self) -> Path:
        """Provision the stable Cloak Browser binary off the event loop."""

        if self._binary_path is not None:
            return self._binary_path

        async with self._preflight_lock:
            if self._binary_path is None:
                try:
                    binary_path = await asyncio.to_thread(
                        ensure_binary,
                        release_channel=_RELEASE_CHANNEL,
                    )
                    self._binary_path = Path(binary_path)
                except asyncio.CancelledError:
                    raise
                except BrowserManagerError:
                    raise
                except Exception as error:
                    raise BrowserManagerError(
                        "Failed to provision the browser binary"
                    ) from error
                logger.info("browser_binary_preflight_complete")
            return self._binary_path

    def session(
        self,
        platform: Platform | str,
        account_name: str,
        *,
        headless: bool | None = None,
    ) -> AsyncContextManager[BrowserSession]:
        """Acquire exclusive use of one configured account profile."""

        return self._session(platform, account_name, headless=headless)

    @asynccontextmanager
    async def _session(
        self,
        platform: Platform | str,
        account_name: str,
        *,
        headless: bool | None,
    ) -> AsyncIterator[BrowserSession]:
        state = self._resolve_state(platform, account_name)
        current_task = asyncio.current_task()
        if current_task is not None and state.owner_task is current_task:
            raise BrowserManagerError(
                "Browser profile leases are serialized and not reentrant"
            )

        async with state.lock:
            if self._closed:
                raise BrowserManagerClosedError(
                    "Browser manager shutdown has begun"
                )

            state.owner_task = cast(asyncio.Task[object] | None, current_task)
            try:
                use_headless = (
                    self._default_headless
                    if headless is None
                    else headless
                )
                context, page = await self._context_and_page(
                    state,
                    headless=use_headless,
                )
                try:
                    status = await classify_auth_state(page, state.platform)
                except asyncio.CancelledError:
                    raise
                except BrowserManagerError:
                    raise
                except Exception as error:
                    raise BrowserManagerError(
                        "Failed to inspect browser authentication state"
                    ) from error
                yield BrowserSession(
                    platform=state.platform,
                    account_name=state.account_name,
                    profile_name=state.profile_name,
                    profile_directory=state.profile_directory,
                    context=context,
                    page=page,
                    status=status,
                )
            finally:
                state.owner_task = None

    def get_page(
        self,
        platform: Platform | str,
        account_name: str,
        *,
        headless: bool | None = None,
    ) -> AsyncContextManager[Page]:
        """Acquire a page while retaining the profile's exclusive lease."""

        return self._get_page(platform, account_name, headless=headless)

    @asynccontextmanager
    async def _get_page(
        self,
        platform: Platform | str,
        account_name: str,
        *,
        headless: bool | None,
    ) -> AsyncIterator[Page]:
        async with self.session(
            platform,
            account_name,
            headless=headless,
        ) as browser_session:
            yield browser_session.page

    def interactive_login(
        self,
        platform: Platform | str,
        account_name: str,
        login_url: str,
    ) -> AsyncContextManager[BrowserSession]:
        """Open a headed login page for caller-orchestrated confirmation."""

        return self._interactive_login(platform, account_name, login_url)

    @asynccontextmanager
    async def _interactive_login(
        self,
        platform: Platform | str,
        account_name: str,
        login_url: str,
    ) -> AsyncIterator[BrowserSession]:
        state = self._resolve_state(platform, account_name)
        validated_login_url = _validated_login_url(
            state.platform,
            login_url,
        )
        async with self.session(
            state.platform,
            account_name,
            headless=False,
        ) as browser_session:
            try:
                await browser_session.page.goto(validated_login_url)
                status = await classify_auth_state(
                    browser_session.page,
                    browser_session.platform,
                )
            except asyncio.CancelledError:
                raise
            except BrowserManagerError:
                raise
            except Exception as error:
                raise BrowserManagerError(
                    "Failed to open the interactive login page"
                ) from error
            yield BrowserSession(
                platform=browser_session.platform,
                account_name=browser_session.account_name,
                profile_name=browser_session.profile_name,
                profile_directory=browser_session.profile_directory,
                context=browser_session.context,
                page=browser_session.page,
                status=status,
            )

    async def _context_and_page(
        self,
        state: _ProfileState,
        *,
        headless: bool,
    ) -> tuple[BrowserContext, Page]:
        try:
            context = state.context
            if context is not None and not self._context_is_healthy(state):
                await self._close_context(state)
                context = state.context

            if context is not None and state.headless is not headless:
                await self._close_context(state)
                context = state.context

            if context is None:
                context = await self._launch_context(
                    state,
                    headless=headless,
                )

            pages = context.pages
            page = state.page
            if (
                page is None
                or page.is_closed()
                or page not in pages
            ):
                page = next(
                    (
                        candidate
                        for candidate in pages
                        if not candidate.is_closed()
                    ),
                    None,
                )
                if page is None:
                    page = await context.new_page()
                state.page = page
            return context, page
        except asyncio.CancelledError:
            raise
        except BrowserManagerError:
            raise
        except Exception as error:
            raise BrowserManagerError(
                "Failed to acquire a browser context and page"
            ) from error

    @staticmethod
    def _context_is_healthy(state: _ProfileState) -> bool:
        context = state.context
        if context is None or state.closed or state.close_failed:
            return False
        try:
            browser = context.browser
            return browser is None or browser.is_connected()
        except PlaywrightError:
            return False

    async def _launch_context(
        self,
        state: _ProfileState,
        *,
        headless: bool,
    ) -> BrowserContext:
        await self.preflight()
        if state.context is not None or state.profile_lock is not None:
            raise BrowserManagerError(
                "Browser profile ownership was not released"
            )

        await self._acquire_profile_lock(state)
        try:
            await self._prepare_profile(state)
            launch_task = asyncio.create_task(
                launch_persistent_context_async(
                    user_data_dir=state.profile_directory,
                    headless=headless,
                    release_channel=_RELEASE_CHANNEL,
                    humanize=False,
                )
            )
            try:
                launched_context = await asyncio.shield(launch_task)
            except asyncio.CancelledError as cancellation:
                while not launch_task.done():
                    try:
                        await asyncio.shield(launch_task)
                    except asyncio.CancelledError:
                        continue
                    except Exception:
                        break

                try:
                    launched_context = launch_task.result()
                except BaseException:
                    raise cancellation

                context = cast(BrowserContext, launched_context)
                self._adopt_context(state, context, headless=headless)
                state.close_failed = True
                close_task = asyncio.create_task(self._close_context(state))
                while not close_task.done():
                    try:
                        await asyncio.shield(close_task)
                    except asyncio.CancelledError:
                        continue
                    except Exception:
                        break
                try:
                    close_task.result()
                except BaseException as cleanup_error:
                    logger.error(
                        "cancelled_browser_launch_cleanup_failed",
                        platform=state.platform.value,
                        account_name=state.account_name,
                        error_type=type(cleanup_error).__name__,
                    )
                raise cancellation
            except Exception as error:
                raise BrowserManagerError(
                    "Failed to launch the browser context"
                ) from error

            context = cast(BrowserContext, launched_context)
            self._adopt_context(state, context, headless=headless)

            def mark_closed(_: BrowserContext | None = None) -> None:
                if state.context is context:
                    state.closed = True

            try:
                context.on("close", mark_closed)
            except Exception as error:
                state.close_failed = True
                try:
                    await self._close_context(state)
                except asyncio.CancelledError:
                    raise
                except BrowserManagerError as close_error:
                    raise BrowserManagerError(
                        "Browser context initialization failed and ownership "
                        "could not be released"
                    ) from ExceptionGroup(
                        "Browser context initialization and close failures",
                        [error, close_error],
                    )
                raise BrowserManagerError(
                    "Failed to initialize browser context ownership"
                ) from error

            logger.info(
                "browser_context_launched",
                platform=state.platform.value,
                account_name=state.account_name,
                headless=headless,
            )
            return context
        except asyncio.CancelledError:
            if state.context is None and state.profile_lock is not None:
                try:
                    await self._release_profile_lock(state)
                except BaseException as cleanup_error:
                    logger.error(
                        "cancelled_browser_startup_lock_release_failed",
                        platform=state.platform.value,
                        account_name=state.account_name,
                        error_type=type(cleanup_error).__name__,
                    )
            raise
        except BrowserManagerError as error:
            if state.context is None:
                await self._release_after_startup_error(state, error)
            raise
        except Exception as error:
            wrapped_error = BrowserManagerError(
                "Failed to initialize browser context ownership"
            )
            if state.context is None:
                await self._release_after_startup_error(
                    state,
                    wrapped_error,
                )
            raise wrapped_error from error

    @staticmethod
    def _adopt_context(
        state: _ProfileState,
        context: BrowserContext,
        *,
        headless: bool,
    ) -> None:
        state.context = context
        state.page = None
        state.headless = headless
        state.closed = False
        state.close_failed = False

    async def _acquire_profile_lock(self, state: _ProfileState) -> None:
        try:
            await asyncio.to_thread(self._prepare_profile_lock_directory)
        except asyncio.CancelledError:
            raise
        except BrowserManagerError:
            raise
        except Exception as error:
            raise BrowserManagerError(
                "Failed to prepare browser profile locking"
            ) from error

        lock_task = asyncio.create_task(
            asyncio.to_thread(
                self._acquire_file_lock,
                state.profile_lock_path,
            )
        )
        try:
            profile_lock = await asyncio.shield(lock_task)
        except asyncio.CancelledError as cancellation:
            try:
                profile_lock = await lock_task
            except BaseException:
                raise cancellation
            state.profile_lock = profile_lock
            try:
                await self._release_profile_lock(state)
            except BaseException as cleanup_error:
                logger.error(
                    "cancelled_profile_lock_release_failed",
                    platform=state.platform.value,
                    account_name=state.account_name,
                    error_type=type(cleanup_error).__name__,
                )
            raise cancellation
        except FileLockTimeout as error:
            raise BrowserManagerError(
                "Browser session profile is already in use by another process"
            ) from error
        except Exception as error:
            raise BrowserManagerError(
                "Failed to acquire browser profile ownership"
            ) from error
        state.profile_lock = profile_lock

    @staticmethod
    def _acquire_file_lock(lock_path: Path) -> FileLock:
        profile_lock = FileLock(lock_path, thread_local=False)
        profile_lock.acquire(timeout=0)
        return profile_lock

    async def _release_profile_lock(self, state: _ProfileState) -> None:
        profile_lock = state.profile_lock
        if profile_lock is None:
            return

        release_task = asyncio.create_task(
            asyncio.to_thread(profile_lock.release)
        )
        try:
            await asyncio.shield(release_task)
        except asyncio.CancelledError as cancellation:
            while not release_task.done():
                try:
                    await asyncio.shield(release_task)
                except asyncio.CancelledError:
                    continue
                except Exception:
                    break
            try:
                release_task.result()
            except BaseException:
                raise cancellation
            state.profile_lock = None
            raise cancellation
        except Exception as error:
            raise BrowserManagerError(
                "Failed to release browser profile ownership"
            ) from error
        state.profile_lock = None

    async def _release_after_startup_error(
        self,
        state: _ProfileState,
        startup_error: Exception,
    ) -> None:
        try:
            await self._release_profile_lock(state)
        except asyncio.CancelledError:
            raise
        except BrowserManagerError as release_error:
            raise BrowserManagerError(
                "Browser startup failed and profile ownership "
                "could not be released"
            ) from ExceptionGroup(
                "Browser startup and profile lock release failures",
                [startup_error, release_error],
            )

    def _prepare_sessions_directory(self) -> Path:
        self._sessions_directory.mkdir(parents=True, exist_ok=True)
        resolved_root = self._sessions_directory.resolve(strict=True)
        if resolved_root != self._sessions_directory:
            raise InvalidSessionProfileError(
                "browser.sessions_directory changed after configuration"
            )
        return resolved_root

    def _prepare_profile_lock_directory(self) -> None:
        resolved_root = self._prepare_sessions_directory()
        lock_directory = resolved_root / _PROFILE_LOCK_DIRECTORY
        lock_directory.mkdir(parents=False, exist_ok=True)
        if lock_directory.resolve(strict=True) != lock_directory:
            raise InvalidSessionProfileError(
                "browser profile lock directory changed after configuration"
            )

    async def _prepare_profile(self, state: _ProfileState) -> None:
        profile_task = asyncio.create_task(
            asyncio.to_thread(self._prepare_profile_directory, state)
        )
        try:
            await asyncio.shield(profile_task)
        except asyncio.CancelledError as cancellation:
            try:
                await profile_task
            except BaseException:
                pass
            raise cancellation
        except BrowserManagerError:
            raise
        except Exception as error:
            raise BrowserManagerError(
                "Failed to prepare the browser profile"
            ) from error

    def _prepare_profile_directory(self, state: _ProfileState) -> None:
        resolved_root = self._prepare_sessions_directory()
        candidate = resolved_root / state.profile_name
        candidate.mkdir(parents=False, exist_ok=True)
        resolved_candidate = candidate.resolve(strict=True)
        try:
            resolved_candidate.relative_to(resolved_root)
        except ValueError as error:
            raise InvalidSessionProfileError(
                "session_profile escaped browser.sessions_directory"
            ) from error
        if resolved_candidate != state.profile_directory:
            raise InvalidSessionProfileError(
                "session_profile path changed after configuration"
            )

    async def _close_context(self, state: _ProfileState) -> None:
        context = state.context
        if context is not None:
            try:
                await context.close()
            except asyncio.CancelledError:
                state.close_failed = True
                raise
            except Exception as error:
                state.close_failed = True
                raise BrowserManagerError(
                    "Failed to close the browser context"
                ) from error
            except BaseException:
                state.close_failed = True
                raise

        try:
            await self._release_profile_lock(state)
        except BaseException:
            state.close_failed = True
            raise
        self._reset_state(state)

    @staticmethod
    def _reset_state(state: _ProfileState) -> None:
        state.context = None
        state.page = None
        state.headless = None
        state.closed = True
        state.close_failed = False
        state.profile_lock = None

    async def capture_failure(
        self,
        page: Page,
        action_id: UUID | str,
    ) -> FailureCapture:
        """Capture safe diagnostics without masking the action's failure."""

        page_url = _sanitized_page_url(page.url or None)
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
        safe_action_id = _SAFE_ACTION_ID.sub("_", str(action_id)).strip(
            "._-"
        )
        safe_action_id = (safe_action_id or "action")[:96]
        screenshot_path = self._screenshots_directory / (
            f"{timestamp}-{safe_action_id}.png"
        )

        try:
            await asyncio.to_thread(
                self._screenshots_directory.mkdir,
                parents=True,
                exist_ok=True,
            )
            await page.screenshot(path=screenshot_path, full_page=True)
        except Exception as error:
            logger.warning(
                "browser_failure_screenshot_failed",
                action_id=safe_action_id,
                error_type=type(error).__name__,
            )
            return FailureCapture(path=None, page_url=page_url)

        logger.info(
            "browser_failure_screenshot_captured",
            action_id=safe_action_id,
            screenshot_name=screenshot_path.name,
        )
        return FailureCapture(path=screenshot_path, page_url=page_url)

    async def shutdown(self) -> None:
        """Stop accepting leases and close every manager-owned context."""

        async with self._shutdown_lock:
            current_task = asyncio.current_task()
            if any(
                current_task is not None and state.owner_task is current_task
                for state in self._profiles.values()
            ):
                raise BrowserManagerError(
                    "shutdown cannot run inside an active browser session lease"
                )

            self._closed = True
            close_errors: list[Exception] = []
            for state in self._profiles.values():
                async with state.lock:
                    try:
                        await self._close_context(state)
                    except Exception as error:
                        close_errors.append(error)
                        logger.error(
                            "browser_context_close_failed",
                            platform=state.platform.value,
                            account_name=state.account_name,
                            error_type=type(error).__name__,
                        )

            if close_errors:
                error_group = ExceptionGroup(
                    "Browser context close failures",
                    close_errors,
                )
                raise BrowserManagerError(
                    "One or more browser contexts failed to close"
                ) from error_group
            logger.info("browser_manager_shutdown_complete")
