"""Installable, safety-first command line interface for social-bot."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import signal
import sys
import threading
import unicodedata
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, is_dataclass
from datetime import UTC, datetime
from enum import Enum, StrEnum
from functools import wraps
from pathlib import Path
from typing import Annotated, Any, ParamSpec, TypeVar, cast
from urllib.parse import quote, unquote, urlsplit, urlunsplit
from uuid import UUID

import click
import structlog
import typer
from pydantic import BaseModel, ValidationError

from bot.browser.manager import BrowserManager, BrowserManagerError
from bot.browser.sessions import SessionStatus, classify_auth_state
from bot.config import ConfigurationError, Settings, load_settings
from bot.models import (
    ActionStatus,
    ActionType,
    Platform,
    SocialAction,
    normalize_target_url,
)
from bot.scheduler import ImportReport, SchedulePreview, SchedulerService
from bot.storage.database import Database
from bot.storage.repositories import (
    AccountStateRepository,
    ActionRepository,
    RepositoryError,
)
from bot.worker import (
    ActionExecutionReport,
    ExecutionDisposition,
    RunOnceReport,
    Worker,
    WorkerReport,
)

P = ParamSpec("P")
R = TypeVar("R")
T = TypeVar("T")

_REDACTED = "[REDACTED]"
_SENSITIVE_KEY_PARTS = frozenset(
    {
        "authorization",
        "cookie",
        "credential",
        "exception",
        "excinfo",
        "header",
        "headers",
        "localstorage",
        "password",
        "passwd",
        "privatekey",
        "secret",
        "secretkey",
        "sessionstorage",
        "setcookie",
        "storagestate",
        "rawstorage",
        "token",
        "apikey",
        "accesstoken",
        "refreshtoken",
        "traceback",
    }
)
_LOG_CONTENT_KEYS = frozenset({"body", "content", "prompt", "text", "title"})
_URL_KEYS = frozenset({"externalcontenturl", "targeturl", "url"})
_SECRET_ASSIGNMENT = re.compile(
    r"(?i)\b(?:authorization|body|content|cookie|set-cookie|headers?|password|"
    r"passwd|prompt|text|title|token|secret|api[_-]?key|storage[_-]?state|"
    r"(?:local|session)[_-]?storage)\b\s*[:=]\s*"
    r"(?:\"[^\"]*\"|'[^']*'|[^\s,;]+)"
)
_REDDIT_FULLNAME = re.compile(r"^(?P<kind>t[13])_(?P<id>[a-z0-9]+)$", re.IGNORECASE)
_REDDIT_ID = re.compile(r"^[a-z0-9]+$", re.IGNORECASE)
_SUBREDDIT = re.compile(r"^[a-z0-9_]{2,21}$", re.IGNORECASE)
_REDDIT_COMMENTS_PATH = re.compile(
    r"^/(?:r/(?P<subreddit>[a-z0-9_]{2,21})/)?comments/"
    r"(?P<post_id>[a-z0-9]+)"
    r"(?:/(?P<slug>[^/]+))?"
    r"(?:/(?P<comment_id>[a-z0-9]+))?/?$",
    re.IGNORECASE,
)
_REDDIT_TARGET_HOSTS = frozenset({"reddit.com", "www.reddit.com", "old.reddit.com"})
_LOGIN_URLS = {
    Platform.X: "https://x.com/i/flow/login",
    Platform.REDDIT: "https://www.reddit.com/login/",
}

logger = structlog.get_logger(__name__)
_logging_configured = False


class LogLevel(StrEnum):
    """Supported application log thresholds."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class CliInputError(ValueError):
    """An actionable command-line input error."""


class CliRuntimeError(RuntimeError):
    """An actionable runtime error safe to display."""


@dataclass(frozen=True, slots=True)
class CliState:
    """Options shared by every command."""

    config_path: Path | None
    log_level: LogLevel
    json_logs: bool


@dataclass(frozen=True, slots=True)
class RedditTarget:
    """A safe, normalized Reddit target and any derivable fullnames."""

    target_url: str | None
    parent_post_id: str | None
    parent_comment_id: str | None
    subreddit: str | None



@dataclass(frozen=True, slots=True)
class WorkerRuntime:
    """Resources composed for Worker-owned cleanup."""

    worker: Worker
    database: Database
    actions: ActionRepository


class _ApplicationLogFilter(logging.Filter):
    """Keep third-party logs from bypassing the application's redaction."""

    def filter(self, record: logging.LogRecord) -> bool:
        return record.name == "bot" or record.name.startswith("bot.")


class _SafeLogFormatter(logging.Formatter):
    """Apply a final redaction pass to every emitted application log."""

    def format(self, record: logging.LogRecord) -> str:
        return _redact_assignments(super().format(record))


def _normalized_key(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).casefold())


def _sensitive_key(value: object) -> bool:
    normalized = _normalized_key(value)
    return any(part in normalized for part in _SENSITIVE_KEY_PARTS)


def _redact_assignments(value: str) -> str:
    return _SECRET_ASSIGNMENT.sub(_REDACTED, value)


def _sanitize_log_value(value: Any, *, key: object | None = None) -> Any:
    if key is not None:
        normalized = _normalized_key(key)
        if _sensitive_key(key) or normalized in _LOG_CONTENT_KEYS or normalized == "metadata":
            return _REDACTED
    if isinstance(value, Mapping):
        return {
            str(item_key): _sanitize_log_value(item_value, key=item_key)
            for item_key, item_value in value.items()
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_sanitize_log_value(item) for item in value]
    if isinstance(value, str):
        return _redact_assignments(value)[:1000]
    if isinstance(value, (UUID, Path, Enum)):
        return str(value)
    if isinstance(value, datetime):
        return _format_datetime(value)
    return value


def _redact_log_event(
    _logger: Any,
    _method_name: str,
    event_dict: dict[str, Any],
) -> dict[str, Any]:
    return cast(dict[str, Any], _sanitize_log_value(event_dict))


def _configure_logging(level: LogLevel, *, json_logs: bool) -> None:
    global _logging_configured

    numeric_level = getattr(logging, level.value.upper())
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(_SafeLogFormatter("%(message)s"))
    handler.addFilter(_ApplicationLogFilter())

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(numeric_level)

    renderer: Callable[..., str]
    if json_logs:
        renderer = structlog.processors.JSONRenderer(sort_keys=True)
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=False)

    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            _redact_log_event,
            renderer,
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        cache_logger_on_first_use=False,
    )
    _logging_configured = True


def _safe_line(value: object, *, limit: int = 1000) -> str:
    text = _redact_assignments(str(value))
    cleaned = "".join(
        " " if unicodedata.category(character) in {"Cc", "Cf"} else character
        for character in text
    )
    return " ".join(cleaned.split())[:limit]


def _validation_error_message(error: ValidationError) -> str:
    details: list[str] = []
    for issue in error.errors(include_url=False):
        location = ".".join(str(part) for part in issue.get("loc", ())) or "action"
        message = _safe_line(issue.get("msg", "invalid value"), limit=300)
        details.append(f"{location}: {message}")
    return "; ".join(details) or "The action is invalid"


def _safe_exception_message(error: Exception) -> tuple[str, int]:
    if isinstance(error, ValidationError):
        return _validation_error_message(error), 2
    if isinstance(error, json.JSONDecodeError):
        return f"Invalid JSON at line {error.lineno}, column {error.colno}", 2
    if isinstance(error, OSError):
        filename = Path(str(error.filename)).name if error.filename else "requested file"
        return f"Could not access '{_safe_line(filename, limit=120)}' ({type(error).__name__})", 2
    if isinstance(error, (CliInputError, ConfigurationError)):
        return _safe_line(error), 2
    if isinstance(error, (RepositoryError, BrowserManagerError, CliRuntimeError)):
        return _safe_line(error), 1
    if isinstance(error, (TypeError, ValueError)):
        return _safe_line(error), 2
    if isinstance(error, RuntimeError):
        return _safe_line(error), 1
    return (
        f"Operation failed safely ({type(error).__name__}); review the configuration "
        "and structured event log",
        1,
    )


def _safe_command(function: Callable[P, R]) -> Callable[P, R]:
    @wraps(function)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return function(*args, **kwargs)
        except (click.ClickException, click.exceptions.Exit, click.Abort):
            raise
        except (KeyboardInterrupt, asyncio.CancelledError):
            typer.echo("Interrupted; runtime resources were closed.", err=True)
            raise typer.Exit(code=130) from None
        except Exception as error:
            if _logging_configured:
                logger.error(
                    "cli_command_failed",
                    command=function.__name__,
                    error_type=type(error).__name__,
                )
            message, code = _safe_exception_message(error)
            typer.echo(f"Error: {message}", err=True)
            raise typer.Exit(code=code) from None

    return wrapped


def _format_datetime(value: datetime) -> str:
    if value.tzinfo is not None and value.utcoffset() is not None:
        value = value.astimezone(UTC)
    rendered = value.isoformat()
    return rendered[:-6] + "Z" if rendered.endswith("+00:00") else rendered


def _sanitize_url(value: str) -> str:
    try:
        parsed = urlsplit(value.strip())
        scheme = parsed.scheme.casefold()
        hostname = (parsed.hostname or "").casefold()
        port = parsed.port
    except (AttributeError, ValueError):
        return _REDACTED
    if scheme not in {"http", "https"} or not hostname:
        return _REDACTED
    authority = f"[{hostname}]" if ":" in hostname else hostname
    is_default_port = (scheme == "http" and port == 80) or (
        scheme == "https" and port == 443
    )
    if port is not None and not is_default_port:
        authority += f":{port}"
    return urlunsplit((scheme, authority, parsed.path or "/", "", ""))


def _safe_public_data(
    value: Any,
    *,
    key: object | None = None,
    allow_user_content: bool = False,
) -> Any:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    elif is_dataclass(value) and not isinstance(value, type):
        value = asdict(value)

    if key is not None:
        normalized = _normalized_key(key)
        if _sensitive_key(key):
            return _REDACTED
        if normalized in _LOG_CONTENT_KEYS and not allow_user_content:
            return _REDACTED
        if normalized in _URL_KEYS and isinstance(value, str):
            return _sanitize_url(value)

    if isinstance(value, Mapping):
        return {
            str(item_key): _safe_public_data(
                item_value,
                key=item_key,
                allow_user_content=allow_user_content,
            )
            for item_key, item_value in value.items()
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        return [
            _safe_public_data(item, allow_user_content=allow_user_content)
            for item in value
        ]
    if isinstance(value, datetime):
        return _format_datetime(value)
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, str) and not (
        allow_user_content and key is not None and _normalized_key(key) in _LOG_CONTENT_KEYS
    ):
        return _redact_assignments(value)
    return value


def _emit_json(value: Any, *, allow_user_content: bool = False) -> None:
    safe_value = _safe_public_data(value, allow_user_content=allow_user_content)
    typer.echo(json.dumps(safe_value, ensure_ascii=True, indent=2, sort_keys=True))


def _action_summary(action: SocialAction) -> dict[str, Any]:
    return {
        "id": str(action.id),
        "action_type": action.action_type.value,
        "platform": action.platform.value,
        "account_name": action.account_name,
        "status": action.status.value,
        "created_at": _format_datetime(action.created_at),
        "scheduled_at": (
            _format_datetime(action.scheduled_at) if action.scheduled_at is not None else None
        ),
        "attempts": action.attempts,
        "max_attempts": action.max_attempts,
    }


def _state() -> CliState:
    context = click.get_current_context(silent=True)
    if context is None or not isinstance(context.find_root().obj, CliState):
        raise CliRuntimeError("CLI context was not initialized")
    return cast(CliState, context.find_root().obj)


def _settings() -> Settings:
    return load_settings(_state().config_path)


def _run_async(awaitable: Awaitable[T]) -> T:
    return asyncio.run(awaitable)


def _interactive_terminal() -> bool:
    return sys.stdin.isatty() and sys.stdout.isatty()


def _prompt_required(label: str, *, option_name: str) -> str:
    if not _interactive_terminal():
        raise CliInputError(f"{option_name} is required when input is noninteractive")
    value = typer.prompt(label)
    if not value.strip():
        raise CliInputError(f"{option_name} cannot be empty")
    return value


def _required_value(value: str | None, label: str, *, option_name: str) -> str:
    if value is not None and value.strip():
        return value
    return _prompt_required(label, option_name=option_name)


def _resolve_text(
    direct: str | None,
    text_file: Path | None,
    *,
    label: str,
    direct_option: str,
    file_option: str,
    required: bool,
) -> str | None:
    if direct is not None and text_file is not None:
        raise CliInputError(f"{direct_option} and {file_option} are mutually exclusive")

    value: str | None
    if text_file is not None:
        value = text_file.read_text(encoding="utf-8")
    else:
        value = direct

    if value is not None and value.strip():
        return value
    if required:
        return _prompt_required(label, option_name=direct_option)
    return None


def _parse_aware_datetime(value: str, *, option_name: str) -> datetime:
    candidate = value.strip()
    if candidate.endswith(("Z", "z")):
        candidate = candidate[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(candidate)
    except (TypeError, ValueError) as error:
        raise CliInputError(
            f"{option_name} must be a valid timezone-aware ISO 8601 timestamp"
        ) from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise CliInputError(f"{option_name} must include a UTC offset or Z")
    return parsed.astimezone(UTC)


def _optional_datetime(value: str | None, *, option_name: str) -> datetime | None:
    return _parse_aware_datetime(value, option_name=option_name) if value else None


def _validated_account(settings: Settings, platform: Platform, account: str) -> str:
    account_name = account.strip()
    if not account_name:
        raise CliInputError("--account cannot be empty")
    accounts = settings.accounts.x if platform is Platform.X else settings.accounts.reddit
    configured = accounts.get(account_name)
    if configured is None:
        raise CliInputError(
            f"No configured {platform.value} account named '{_safe_line(account_name, limit=120)}'"
        )
    if not configured.enabled:
        raise CliInputError(
            f"The configured {platform.value} account '{_safe_line(account_name, limit=120)}' is disabled"
        )
    return account_name


def _normalize_subreddit(value: str) -> str:
    normalized = value.strip()
    if normalized.casefold().startswith("r/"):
        normalized = normalized[2:]
    if not _SUBREDDIT.fullmatch(normalized):
        raise CliInputError("--subreddit must be a valid Reddit community name")
    return normalized.casefold()


def _normalize_link_url(value: str) -> str:
    try:
        parsed = urlsplit(value.strip())
        port = parsed.port
    except (AttributeError, ValueError) as error:
        raise CliInputError("--url must be a valid absolute HTTP(S) URL") from error
    scheme = parsed.scheme.casefold()
    if (
        scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or (port is not None and port != (80 if scheme == "http" else 443))
    ):
        raise CliInputError(
            "--url must be absolute HTTP(S), contain no credentials, and use the default port"
        )
    return normalize_target_url(value)


def _canonical_reddit_path(
    *,
    subreddit: str | None,
    post_id: str,
    slug: str,
    comment_id: str | None,
) -> str:
    prefix = f"/r/{quote(subreddit, safe='_')}/" if subreddit else "/"
    path = f"{prefix}comments/{quote(post_id, safe='')}/{quote(slug, safe='-._~')}"
    if comment_id is not None:
        path += f"/{quote(comment_id, safe='')}"
    return f"{path}/"


def _parse_reddit_target(value: str, *, expected_kind: str) -> RedditTarget:
    target = value.strip()
    fullname_match = _REDDIT_FULLNAME.fullmatch(target)
    if fullname_match is not None:
        kind = fullname_match.group("kind").casefold()
        if kind != expected_kind:
            expected = "post" if expected_kind == "t3" else "comment"
            raise CliInputError(f"--target must identify a Reddit {expected}")
        fullname = f"{kind}_{fullname_match.group('id').casefold()}"
        return RedditTarget(
            target_url=None,
            parent_post_id=fullname if kind == "t3" else None,
            parent_comment_id=fullname if kind == "t1" else None,
            subreddit=None,
        )

    if _REDDIT_ID.fullmatch(target):
        fullname = f"{expected_kind}_{target.casefold()}"
        return RedditTarget(
            target_url=None,
            parent_post_id=fullname if expected_kind == "t3" else None,
            parent_comment_id=fullname if expected_kind == "t1" else None,
            subreddit=None,
        )

    try:
        parsed = urlsplit(target)
        scheme = parsed.scheme.casefold()
        hostname = (parsed.hostname or "").casefold()
        port = parsed.port
    except (AttributeError, ValueError) as error:
        raise CliInputError(
            "--target must be a Reddit permalink, fullname, or base-36 ID"
        ) from error
    if (
        scheme not in {"http", "https"}
        or hostname not in _REDDIT_TARGET_HOSTS
        or parsed.username is not None
        or parsed.password is not None
        or (port is not None and port != (80 if scheme == "http" else 443))
    ):
        raise CliInputError(
            "--target must be a credential-free reddit.com permalink using the default port"
        )

    path_match = _REDDIT_COMMENTS_PATH.fullmatch(unquote(parsed.path))
    if path_match is None:
        raise CliInputError("--target must be a Reddit post or comment permalink")

    post_id = path_match.group("post_id").casefold()
    comment_id = path_match.group("comment_id")
    if expected_kind == "t1" and comment_id is None:
        raise CliInputError("--target must identify a Reddit comment")
    normalized_comment_id = comment_id.casefold() if comment_id is not None else None
    subreddit = path_match.group("subreddit")
    normalized_subreddit = subreddit.casefold() if subreddit is not None else None
    slug = path_match.group("slug") or "-"
    canonical_path = _canonical_reddit_path(
        subreddit=normalized_subreddit,
        post_id=post_id,
        slug=slug,
        comment_id=normalized_comment_id if expected_kind == "t1" else None,
    )
    return RedditTarget(
        target_url=urlunsplit(("https", "www.reddit.com", canonical_path, "", "")),
        parent_post_id=f"t3_{post_id}",
        parent_comment_id=(
            f"t1_{normalized_comment_id}"
            if expected_kind == "t1" and normalized_comment_id is not None
            else None
        ),
        subreddit=normalized_subreddit,
    )


def _resolve_target_subreddit(
    supplied: str | None,
    derived: str | None,
) -> str | None:
    normalized = _normalize_subreddit(supplied) if supplied is not None else None
    if normalized is not None and derived is not None and normalized != derived.casefold():
        raise CliInputError("--subreddit does not match the subreddit in --target")
    return normalized or derived


@asynccontextmanager
async def _scheduler_context(settings: Settings):
    async with Database(settings.database_url) as database:
        actions = ActionRepository(database.session_factory)
        yield SchedulerService(settings, actions)


async def _create_action(
    settings: Settings,
    action: SocialAction,
    *,
    scheduled_at: datetime | None,
) -> SocialAction:
    async with _scheduler_context(settings) as scheduler:
        return await scheduler.create_action(action, scheduled_at=scheduled_at)


async def _get_preview(
    settings: Settings,
    action_id: UUID,
) -> tuple[SocialAction, SchedulePreview]:
    async with _scheduler_context(settings) as scheduler:
        action = await scheduler.get(action_id)
        preview = await scheduler.preview(action_id)
        return action, preview


async def _approve_action(
    settings: Settings,
    action_id: UUID,
    scheduled_at: datetime | None,
) -> SocialAction:
    async with _scheduler_context(settings) as scheduler:
        return await scheduler.approve(action_id, scheduled_at=scheduled_at)


async def _schedule_action(
    settings: Settings,
    action_id: UUID,
    scheduled_at: datetime,
) -> SocialAction:
    async with _scheduler_context(settings) as scheduler:
        return await scheduler.schedule(action_id, scheduled_at=scheduled_at)


async def _cancel_action(
    settings: Settings,
    action_id: UUID,
    reason: str | None,
) -> SocialAction:
    async with _scheduler_context(settings) as scheduler:
        return await scheduler.cancel(action_id, reason=reason)


async def _list_actions(
    settings: Settings,
    *,
    statuses: list[ActionStatus] | None,
    platform: Platform | None,
    account_name: str | None,
    limit: int,
    offset: int,
) -> list[SocialAction]:
    async with _scheduler_context(settings) as scheduler:
        return await scheduler.list(
            statuses=statuses,
            platform=platform,
            account_name=account_name.strip() if account_name else None,
            limit=limit,
            offset=offset,
        )


async def _import_actions(
    settings: Settings,
    actions: Sequence[Any],
) -> ImportReport:
    async with _scheduler_context(settings) as scheduler:
        return await scheduler.import_actions(cast(Sequence[Mapping[str, Any]], actions))


def _build_worker(settings: Settings) -> WorkerRuntime:
    browser_manager = BrowserManager(settings)
    database = Database(settings.database_url)
    actions = ActionRepository(database.session_factory)
    account_states = AccountStateRepository(database.session_factory)
    worker = Worker(
        settings,
        action_repository=actions,
        account_state_repository=account_states,
        browser_manager=browser_manager,
        database=database,
    )
    return WorkerRuntime(worker=worker, database=database, actions=actions)


async def _execute_action(
    settings: Settings,
    action_id: UUID,
) -> ActionExecutionReport:
    runtime = _build_worker(settings)
    try:
        await runtime.database.initialize()
        scheduler = SchedulerService(settings, runtime.actions, worker=runtime.worker)
        return await scheduler.execute_now(action_id)
    finally:
        await runtime.worker.close()


async def _run_worker(
    settings: Settings,
    *,
    once: bool,
) -> RunOnceReport | WorkerReport:
    runtime = _build_worker(settings)
    worker = runtime.worker
    loop = asyncio.get_running_loop()
    installed_signals: list[signal.Signals] = []
    try:
        for interrupt_signal in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(interrupt_signal, worker.stop)
            except (NotImplementedError, RuntimeError, ValueError):
                continue
            installed_signals.append(interrupt_signal)

        if once:
            return await worker.run_once()
        return await worker.run()
    finally:
        try:
            for interrupt_signal in installed_signals:
                loop.remove_signal_handler(interrupt_signal)
        finally:
            await worker.close()


async def _wait_for_login_confirmation() -> None:
    """Wait for Enter without making loop shutdown join a blocked input thread."""

    loop = asyncio.get_running_loop()
    confirmation: asyncio.Future[BaseException | None] = loop.create_future()

    def deliver(outcome: BaseException | None) -> None:
        if not confirmation.done():
            confirmation.set_result(outcome)

    def read_confirmation() -> None:
        outcome: BaseException | None
        try:
            input()
        except BaseException as error:
            outcome = error
        else:
            outcome = None
        try:
            loop.call_soon_threadsafe(deliver, outcome)
        except RuntimeError:
            return

    login_task = asyncio.current_task()
    sigterm_installed = False
    if login_task is not None:
        try:
            loop.add_signal_handler(signal.SIGTERM, login_task.cancel)
        except (NotImplementedError, RuntimeError, ValueError):
            pass
        else:
            sigterm_installed = True

    try:
        threading.Thread(
            target=read_confirmation,
            name="social-bot-login-input",
            daemon=True,
        ).start()
        outcome = await confirmation
    finally:
        if sigterm_installed:
            loop.remove_signal_handler(signal.SIGTERM)

    if outcome is None:
        return
    if isinstance(outcome, EOFError):
        raise CliInputError(
            "Login confirmation requires an interactive terminal"
        ) from outcome
    raise outcome


async def _interactive_login(
    settings: Settings,
    platform: Platform,
    account_name: str,
) -> SessionStatus:
    browser_manager = BrowserManager(settings)
    database = Database(settings.database_url)
    try:
        await database.initialize()
        ActionRepository(database.session_factory)
        async with browser_manager.interactive_login(
            platform,
            account_name,
            _LOGIN_URLS[platform],
        ) as browser_session:
            typer.echo(
                "A headed browser is open. Complete login or any required human "
                "challenge, then press Enter here.",
                err=True,
            )
            await _wait_for_login_confirmation()
            return await classify_auth_state(browser_session.page, platform)
    finally:
        try:
            await browser_manager.shutdown()
        finally:
            await database.close()


def _persist_and_report(
    settings: Settings,
    payload: Mapping[str, Any],
    scheduled_at: str | None,
) -> None:
    parsed_schedule = _optional_datetime(scheduled_at, option_name="--scheduled-at")
    action = SocialAction.model_validate(payload)
    created = _run_async(_create_action(settings, action, scheduled_at=parsed_schedule))
    _emit_json({"operation": "create", "action": _action_summary(created)})


def _print_action_table(actions: Sequence[SocialAction]) -> None:
    if not actions:
        typer.echo("No actions found.")
        return
    headers = ("ID", "TYPE", "PLATFORM", "ACCOUNT", "STATUS", "SCHEDULED", "ATTEMPTS")
    rows = [
        (
            str(action.id),
            action.action_type.value,
            action.platform.value,
            _safe_line(action.account_name, limit=32),
            action.status.value,
            _format_datetime(action.scheduled_at) if action.scheduled_at else "-",
            f"{action.attempts}/{action.max_attempts}",
        )
        for action in actions
    ]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]
    typer.echo("  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    typer.echo("  ".join("-" * width for width in widths))
    for row in rows:
        typer.echo("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))


app = typer.Typer(
    name="social-bot",
    help="Create, review, schedule, and safely execute social actions.",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
    rich_markup_mode=None,
    context_settings={"help_option_names": ["-h", "--help"]},
)
create_app = typer.Typer(
    help="Create a validated social action.",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
    rich_markup_mode=None,
)
app.add_typer(create_app, name="create")


@app.callback()
def root_callback(
    ctx: typer.Context,
    config: Annotated[
        Path | None,
        typer.Option(
            "--config",
            envvar="STEALTH_BOT_CONFIG",
            help="JSON or YAML configuration file.",
            dir_okay=False,
            readable=True,
            resolve_path=True,
        ),
    ] = None,
    log_level: Annotated[
        LogLevel,
        typer.Option("--log-level", help="Structured application log threshold."),
    ] = LogLevel.INFO,
    json_logs: Annotated[
        bool,
        typer.Option("--json-logs/--console-logs", help="Emit structured logs as JSON."),
    ] = False,
) -> None:
    """Initialize command context without opening runtime resources."""

    _configure_logging(log_level, json_logs=json_logs)
    ctx.obj = CliState(config_path=config, log_level=log_level, json_logs=json_logs)


@app.command("login")
@_safe_command
def login_command(
    platform: Annotated[Platform, typer.Argument(help="Platform: x or reddit.")],
    account: Annotated[str, typer.Option("--account", help="Configured account name.")],
) -> None:
    """Open a headed browser for manual, credential-free login."""

    if not _interactive_terminal():
        raise CliInputError("login requires an interactive terminal")
    settings = _settings()
    account_name = _validated_account(settings, platform, account)
    status = _run_async(_interactive_login(settings, platform, account_name))
    next_steps = {
        SessionStatus.AUTHENTICATED: "Session is authenticated.",
        SessionStatus.AUTH_REQUIRED: "Authentication was not confirmed; rerun login to continue.",
        SessionStatus.CHALLENGE_REQUIRED: "A human challenge remains; resolve it and rerun login.",
        SessionStatus.CLOSED: "The browser closed before authentication could be confirmed.",
        SessionStatus.UNKNOWN: "Authentication state could not be confirmed; rerun login.",
    }
    _emit_json(
        {
            "platform": platform.value,
            "account_name": account_name,
            "status": status.value,
            "authenticated": status is SessionStatus.AUTHENTICATED,
            "next_step": next_steps[status],
        }
    )
    if status is not SessionStatus.AUTHENTICATED:
        raise typer.Exit(code=1)


@create_app.command("x-post")
@_safe_command
def create_x_post(
    account: Annotated[str, typer.Option("--account", help="Configured X account name.")],
    content: Annotated[str | None, typer.Option("--content", help="Post content.")] = None,
    content_file: Annotated[
        Path | None,
        typer.Option(
            "--content-file",
            help="UTF-8 file containing post content.",
            exists=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
        ),
    ] = None,
    scheduled_at: Annotated[
        str | None,
        typer.Option("--scheduled-at", help="Aware ISO 8601 requested schedule."),
    ] = None,
) -> None:
    """Create an X post action without publishing it directly."""

    settings = _settings()
    account_name = _validated_account(settings, Platform.X, account)
    resolved_content = _resolve_text(
        content,
        content_file,
        label="Post content",
        direct_option="--content",
        file_option="--content-file",
        required=True,
    )
    _persist_and_report(
        settings,
        {
            "action_type": ActionType.X_POST,
            "platform": Platform.X,
            "account_name": account_name,
            "content": resolved_content,
            "metadata": {"source": "cli", "command": "create x-post"},
        },
        scheduled_at,
    )


@create_app.command("reddit-post")
@_safe_command
def create_reddit_post(
    account: Annotated[
        str, typer.Option("--account", help="Configured Reddit account name.")
    ],
    title: Annotated[str | None, typer.Option("--title", help="Post title.")] = None,
    subreddit: Annotated[
        str | None, typer.Option("--subreddit", help="Destination subreddit.")
    ] = None,
    body: Annotated[str | None, typer.Option("--body", help="Text-post body.")] = None,
    body_file: Annotated[
        Path | None,
        typer.Option(
            "--body-file",
            help="UTF-8 file containing the text-post body.",
            exists=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
        ),
    ] = None,
    url: Annotated[str | None, typer.Option("--url", help="Link-post URL.")] = None,
    scheduled_at: Annotated[
        str | None,
        typer.Option("--scheduled-at", help="Aware ISO 8601 requested schedule."),
    ] = None,
) -> None:
    """Create a Reddit text or link post action."""

    settings = _settings()
    account_name = _validated_account(settings, Platform.REDDIT, account)
    resolved_title = _required_value(title, "Post title", option_name="--title")
    resolved_subreddit = _normalize_subreddit(
        _required_value(subreddit, "Subreddit", option_name="--subreddit")
    )
    resolved_body = _resolve_text(
        body,
        body_file,
        label="Post body",
        direct_option="--body",
        file_option="--body-file",
        required=False,
    )
    resolved_url = url if url is not None and url.strip() else None
    if resolved_body is not None and resolved_url is not None:
        raise CliInputError("Reddit posts require exactly one of body/body-file or --url")
    if resolved_body is None and resolved_url is None:
        if not _interactive_terminal():
            raise CliInputError(
                "Reddit posts require exactly one of --body/--body-file or --url"
            )
        post_kind = typer.prompt(
            "Post type",
            type=click.Choice(["text", "link"], case_sensitive=False),
            default="text",
        )
        if post_kind == "text":
            resolved_body = _prompt_required("Post body", option_name="--body")
        else:
            resolved_url = _prompt_required("Link URL", option_name="--url")

    normalized_url = _normalize_link_url(resolved_url) if resolved_url is not None else None
    _persist_and_report(
        settings,
        {
            "action_type": ActionType.REDDIT_POST,
            "platform": Platform.REDDIT,
            "account_name": account_name,
            "content": resolved_body or "",
            "title": resolved_title,
            "subreddit": resolved_subreddit,
            "target_url": normalized_url,
            "metadata": {"source": "cli", "command": "create reddit-post"},
        },
        scheduled_at,
    )


@create_app.command("reddit-comment")
@_safe_command
def create_reddit_comment(
    account: Annotated[
        str, typer.Option("--account", help="Configured Reddit account name.")
    ],
    target: Annotated[
        str | None,
        typer.Option("--target", help="Reddit post permalink, fullname, or ID."),
    ] = None,
    body: Annotated[str | None, typer.Option("--body", help="Comment body.")] = None,
    body_file: Annotated[
        Path | None,
        typer.Option(
            "--body-file",
            help="UTF-8 file containing the comment body.",
            exists=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
        ),
    ] = None,
    subreddit: Annotated[
        str | None, typer.Option("--subreddit", help="Optional subreddit hint.")
    ] = None,
    scheduled_at: Annotated[
        str | None,
        typer.Option("--scheduled-at", help="Aware ISO 8601 requested schedule."),
    ] = None,
) -> None:
    """Create an action that comments on a Reddit post."""

    settings = _settings()
    account_name = _validated_account(settings, Platform.REDDIT, account)
    resolved_target = _parse_reddit_target(
        _required_value(target, "Target post", option_name="--target"),
        expected_kind="t3",
    )
    resolved_body = _resolve_text(
        body,
        body_file,
        label="Comment body",
        direct_option="--body",
        file_option="--body-file",
        required=True,
    )
    resolved_subreddit = _resolve_target_subreddit(subreddit, resolved_target.subreddit)
    _persist_and_report(
        settings,
        {
            "action_type": ActionType.REDDIT_COMMENT,
            "platform": Platform.REDDIT,
            "account_name": account_name,
            "content": resolved_body,
            "subreddit": resolved_subreddit,
            "target_url": resolved_target.target_url,
            "parent_post_id": resolved_target.parent_post_id,
            "metadata": {"source": "cli", "command": "create reddit-comment"},
        },
        scheduled_at,
    )


@create_app.command("reddit-reply")
@_safe_command
def create_reddit_reply(
    account: Annotated[
        str, typer.Option("--account", help="Configured Reddit account name.")
    ],
    target: Annotated[
        str | None,
        typer.Option("--target", help="Reddit comment permalink, fullname, or ID."),
    ] = None,
    body: Annotated[str | None, typer.Option("--body", help="Reply body.")] = None,
    body_file: Annotated[
        Path | None,
        typer.Option(
            "--body-file",
            help="UTF-8 file containing the reply body.",
            exists=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
        ),
    ] = None,
    subreddit: Annotated[
        str | None, typer.Option("--subreddit", help="Optional subreddit hint.")
    ] = None,
    scheduled_at: Annotated[
        str | None,
        typer.Option("--scheduled-at", help="Aware ISO 8601 requested schedule."),
    ] = None,
) -> None:
    """Create an action that replies to a Reddit comment."""

    settings = _settings()
    account_name = _validated_account(settings, Platform.REDDIT, account)
    resolved_target = _parse_reddit_target(
        _required_value(target, "Target comment", option_name="--target"),
        expected_kind="t1",
    )
    resolved_body = _resolve_text(
        body,
        body_file,
        label="Reply body",
        direct_option="--body",
        file_option="--body-file",
        required=True,
    )
    resolved_subreddit = _resolve_target_subreddit(subreddit, resolved_target.subreddit)
    _persist_and_report(
        settings,
        {
            "action_type": ActionType.REDDIT_REPLY,
            "platform": Platform.REDDIT,
            "account_name": account_name,
            "content": resolved_body,
            "subreddit": resolved_subreddit,
            "target_url": resolved_target.target_url,
            "parent_post_id": resolved_target.parent_post_id,
            "parent_comment_id": resolved_target.parent_comment_id,
            "metadata": {"source": "cli", "command": "create reddit-reply"},
        },
        scheduled_at,
    )


@app.command("import-json")
@_safe_command
def import_json_command(
    source: Annotated[
        Path,
        typer.Argument(
            help="JSON file containing a list or an object with an actions list.",
            exists=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
        ),
    ],
) -> None:
    """Import independently validated actions and report every failure."""

    payload = json.loads(source.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        actions = payload
    elif isinstance(payload, dict) and isinstance(payload.get("actions"), list):
        actions = payload["actions"]
    else:
        raise CliInputError("JSON must be a list or an object containing an actions list")

    settings = _settings()
    try:
        report = _run_async(_import_actions(settings, actions))
    except Exception as error:
        if _logging_configured:
            logger.error("json_import_aborted", error_type=type(error).__name__)
        raise CliRuntimeError(
            "Import stopped before completion; earlier entries may have been created"
        ) from error

    _emit_json(
        {
            "total_count": report.total_count,
            "created_count": report.created_count,
            "failure_count": report.failure_count,
            "created": [_action_summary(action) for action in report.created],
            "failures": [asdict(failure) for failure in report.failures],
        }
    )
    if report.failure_count:
        raise typer.Exit(code=1)


@app.command("preview")
@_safe_command
def preview_command(
    action_id: Annotated[UUID, typer.Argument(help="Action UUID.")],
) -> None:
    """Intentionally display the complete action content and schedule preview."""

    action, preview = _run_async(_get_preview(_settings(), action_id))
    _emit_json(
        {"action": action.model_dump(mode="json"), "schedule": asdict(preview)},
        allow_user_content=True,
    )


@app.command("approve")
@_safe_command
def approve_command(
    action_id: Annotated[UUID, typer.Argument(help="Action UUID.")],
    at: Annotated[
        str | None,
        typer.Option("--at", help="Optional aware ISO 8601 schedule override."),
    ] = None,
) -> None:
    """Approve a pending action through the configured approval gate."""

    scheduled_at = _optional_datetime(at, option_name="--at")
    action = _run_async(_approve_action(_settings(), action_id, scheduled_at))
    _emit_json({"operation": "approve", "action": _action_summary(action)})


@app.command("schedule")
@_safe_command
def schedule_command(
    action_id: Annotated[UUID, typer.Argument(help="Action UUID.")],
    at: Annotated[str, typer.Option("--at", help="Aware ISO 8601 schedule time.")],
) -> None:
    """Schedule an action at an explicit timezone-aware instant."""

    scheduled_at = _parse_aware_datetime(at, option_name="--at")
    action = _run_async(_schedule_action(_settings(), action_id, scheduled_at))
    _emit_json({"operation": "schedule", "action": _action_summary(action)})


@app.command("cancel")
@_safe_command
def cancel_command(
    action_id: Annotated[UUID, typer.Argument(help="Action UUID.")],
    reason: Annotated[
        str | None,
        typer.Option("--reason", help="Optional private cancellation reason."),
    ] = None,
) -> None:
    """Cancel an action without displaying its content or reason."""

    action = _run_async(_cancel_action(_settings(), action_id, reason))
    _emit_json({"operation": "cancel", "action": _action_summary(action)})


@app.command("execute")
@_safe_command
def execute_command(
    action_id: Annotated[UUID, typer.Argument(help="Action UUID.")],
) -> None:
    """Execute through Worker safety, approval, pause, and dry-run gates."""

    report = _run_async(_execute_action(_settings(), action_id))
    _emit_json({"operation": "execute", "report": asdict(report)})
    if report.disposition in {
        ExecutionDisposition.FAILED,
        ExecutionDisposition.RETRY_SCHEDULED,
    }:
        raise typer.Exit(code=1)


@app.command("list")
@_safe_command
def list_command(
    statuses: Annotated[
        list[ActionStatus] | None,
        typer.Option("--status", help="Status filter; repeat for multiple values."),
    ] = None,
    platform: Annotated[
        Platform | None,
        typer.Option("--platform", help="Platform filter."),
    ] = None,
    account: Annotated[
        str | None,
        typer.Option("--account", help="Exact account-name filter."),
    ] = None,
    limit: Annotated[
        int,
        typer.Option("--limit", min=1, max=1000, help="Maximum rows."),
    ] = 100,
    offset: Annotated[
        int,
        typer.Option("--offset", min=0, help="Rows to skip."),
    ] = 0,
    as_json: Annotated[
        bool,
        typer.Option("--json", help="Emit safe JSON summaries instead of a table."),
    ] = False,
) -> None:
    """List safe action summaries without content, targets, or metadata."""

    actions = _run_async(
        _list_actions(
            _settings(),
            statuses=statuses,
            platform=platform,
            account_name=account,
            limit=limit,
            offset=offset,
        )
    )
    if as_json:
        _emit_json([_action_summary(action) for action in actions])
    else:
        _print_action_table(actions)


@app.command("worker")
@_safe_command
def worker_command(
    once: Annotated[
        bool,
        typer.Option("--once", help="Process one due batch and exit."),
    ] = False,
) -> None:
    """Run the safe action worker once or until Ctrl-C requests a graceful stop."""

    if not once:
        typer.echo("Worker running; press Ctrl-C to stop gracefully.", err=True)
    report = _run_async(_run_worker(_settings(), once=once))
    _emit_json({"mode": "once" if once else "persistent", "report": asdict(report)})


def main() -> None:
    """Run the installed command-line application."""

    app(prog_name="social-bot")


__all__ = ["app", "create_app", "main"]
