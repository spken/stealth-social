"""Shared, redacted CLI infrastructure and safe input helpers."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import sys
import unicodedata
from collections.abc import Awaitable, Callable, Coroutine, Mapping
from dataclasses import asdict, dataclass, is_dataclass
from datetime import UTC, datetime
from enum import Enum, StrEnum
from functools import wraps
from pathlib import Path
from typing import Any, ParamSpec, TypeVar, cast
from urllib.parse import quote, unquote, urlsplit, urlunsplit
from uuid import UUID

import click
import structlog
from structlog.typing import EventDict, Processor
import typer
from pydantic import BaseModel, ValidationError

from bot.browser.manager import BrowserManagerError
from bot.config import ConfigurationError, Settings, load_settings
from bot.models import ActionStatus, Platform, SocialAction, normalize_target_url
from bot.ollama.errors import OllamaError
from bot.scheduler import ImportReport, SchedulePreview, SchedulerService
from bot.storage.content_repository import ContentRepository
from bot.storage.database import Database
from bot.storage.repositories import ActionRepository, RepositoryError

P = ParamSpec("P")
R = TypeVar("R")
T = TypeVar("T")

_REDACTED = "[REDACTED]"
_SENSITIVE_KEY_PARTS = frozenset(
    {
        "authorization", "cookie", "credential", "exception", "excinfo", "header",
        "headers", "localstorage", "password", "passwd", "privatekey", "secret",
        "secretkey", "sessionstorage", "setcookie", "storagestate", "rawstorage",
        "token", "apikey", "accesstoken", "refreshtoken", "traceback",
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
    r"(?P<post_id>[a-z0-9]+)(?:/(?P<slug>[^/]+))?"
    r"(?:/(?P<comment_id>[a-z0-9]+))?/?$",
    re.IGNORECASE,
)
_REDDIT_TARGET_HOSTS = frozenset({"reddit.com", "www.reddit.com", "old.reddit.com"})

logger = structlog.get_logger(__name__)
_logging_configured = False


class LogLevel(StrEnum):
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
    config_path: Path | None
    log_level: LogLevel
    json_logs: bool


@dataclass(frozen=True, slots=True)
class RedditTarget:
    target_url: str | None
    parent_post_id: str | None
    parent_comment_id: str | None
    subreddit: str | None


class _ApplicationLogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.name == "bot" or record.name.startswith("bot.")


class _SafeLogFormatter(logging.Formatter):
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
        return {str(k): _sanitize_log_value(v, key=k) for k, v in value.items()}
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
    _logger: Any, _method_name: str, event_dict: EventDict
) -> EventDict:
    return cast(EventDict, _sanitize_log_value(event_dict))


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
    renderer: Processor
    renderer = (
        structlog.processors.JSONRenderer(sort_keys=True)
        if json_logs
        else structlog.dev.ConsoleRenderer(colors=False)
    )
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
    details = []
    for issue in error.errors(include_url=False):
        location = ".".join(str(part) for part in issue.get("loc", ())) or "action"
        details.append(f"{location}: {_safe_line(issue.get('msg', 'invalid value'), limit=300)}")
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
    if isinstance(error, (RepositoryError, BrowserManagerError, CliRuntimeError, OllamaError)):
        return _safe_line(error), 1
    if isinstance(error, (TypeError, ValueError)):
        return _safe_line(error), 2
    return (
        f"Operation failed safely ({type(error).__name__}); review the configuration and structured event log",
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
                logger.error("cli_command_failed", command=function.__name__, error_type=type(error).__name__)
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
    if port is not None and port not in {80 if scheme == "http" else 443}:
        authority += f":{port}"
    return urlunsplit((scheme, authority, parsed.path or "/", "", ""))


def _safe_public_data(value: Any, *, key: object | None = None, allow_user_content: bool = False) -> Any:
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
        return {str(k): _safe_public_data(v, key=k, allow_user_content=allow_user_content) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_safe_public_data(item, allow_user_content=allow_user_content) for item in value]
    if isinstance(value, datetime):
        return _format_datetime(value)
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, str) and not (allow_user_content and key is not None and _normalized_key(key) in _LOG_CONTENT_KEYS):
        return _redact_assignments(value)
    return value


def _emit_json(value: Any, *, allow_user_content: bool = False) -> None:
    typer.echo(json.dumps(_safe_public_data(value, allow_user_content=allow_user_content), ensure_ascii=True, indent=2, sort_keys=True))


def _action_summary(action: SocialAction) -> dict[str, Any]:
    return {
        "id": str(action.id),
        "action_type": action.action_type.value,
        "platform": action.platform.value,
        "account_name": action.account_name,
        "status": action.status.value,
        "created_at": _format_datetime(action.created_at),
        "scheduled_at": _format_datetime(action.scheduled_at) if action.scheduled_at else None,
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


def _run_async(awaitable: Coroutine[Any, Any, T]) -> T:
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
    return value if value is not None and value.strip() else _prompt_required(label, option_name=option_name)


def _resolve_text(direct: str | None, text_file: Path | None, *, label: str, direct_option: str, file_option: str, required: bool) -> str | None:
    if direct is not None and text_file is not None:
        raise CliInputError(f"{direct_option} and {file_option} are mutually exclusive")
    value = text_file.read_text(encoding="utf-8") if text_file is not None else direct
    if value is not None and value.strip():
        return value
    return _prompt_required(label, option_name=direct_option) if required else None


def _parse_aware_datetime(value: str, *, option_name: str) -> datetime:
    candidate = value.strip()
    if candidate.endswith(("Z", "z")):
        candidate = candidate[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(candidate)
    except (TypeError, ValueError) as error:
        raise CliInputError(f"{option_name} must be a valid timezone-aware ISO 8601 timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise CliInputError(f"{option_name} must include a UTC offset or Z")
    return parsed.astimezone(UTC)


def _optional_datetime(value: str | None, *, option_name: str) -> datetime | None:
    return _parse_aware_datetime(value, option_name=option_name) if value else None


def _validated_account(settings: Settings, platform: Platform, account: str) -> str:
    name = account.strip()
    accounts = settings.accounts.x if platform is Platform.X else settings.accounts.reddit
    configured = accounts.get(name)
    if not name or configured is None:
        raise CliInputError(f"No configured {platform.value} account named '{_safe_line(name, limit=120)}'")
    if not configured.enabled:
        raise CliInputError(f"The configured {platform.value} account '{_safe_line(name, limit=120)}' is disabled")
    return name


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
    if scheme not in {"http", "https"} or not parsed.hostname or parsed.username is not None or parsed.password is not None or (port is not None and port != (80 if scheme == "http" else 443)):
        raise CliInputError("--url must be absolute HTTP(S), contain no credentials, and use the default port")
    return normalize_target_url(value)


def _canonical_reddit_path(*, subreddit: str | None, post_id: str, slug: str, comment_id: str | None) -> str:
    prefix = f"/r/{quote(subreddit, safe='_')}/" if subreddit else "/"
    path = f"{prefix}comments/{quote(post_id, safe='')}/{quote(slug, safe='-._~')}"
    if comment_id is not None:
        path += f"/{quote(comment_id, safe='')}"
    return f"{path}/"


def _parse_reddit_target(value: str, *, expected_kind: str) -> RedditTarget:
    target = value.strip()
    fullname = _REDDIT_FULLNAME.fullmatch(target)
    if fullname:
        kind = fullname.group("kind").casefold()
        if kind != expected_kind:
            raise CliInputError("--target has the wrong Reddit target kind")
        identifier = f"{kind}_{fullname.group('id').casefold()}"
        return RedditTarget(None, identifier if kind == "t3" else None, identifier if kind == "t1" else None, None)
    if _REDDIT_ID.fullmatch(target):
        identifier = f"{expected_kind}_{target.casefold()}"
        return RedditTarget(None, identifier if expected_kind == "t3" else None, identifier if expected_kind == "t1" else None, None)
    try:
        parsed = urlsplit(target)
        scheme = parsed.scheme.casefold()
        hostname = (parsed.hostname or "").casefold()
        port = parsed.port
    except (AttributeError, ValueError) as error:
        raise CliInputError("--target must be a Reddit permalink, fullname, or base-36 ID") from error
    if scheme not in {"http", "https"} or hostname not in _REDDIT_TARGET_HOSTS or parsed.username is not None or parsed.password is not None or (port is not None and port != (80 if scheme == "http" else 443)):
        raise CliInputError("--target must be a credential-free reddit.com permalink using the default port")
    match = _REDDIT_COMMENTS_PATH.fullmatch(unquote(parsed.path))
    if match is None:
        raise CliInputError("--target must be a Reddit post or comment permalink")
    post_id = match.group("post_id").casefold()
    comment_id = match.group("comment_id")
    if expected_kind == "t1" and comment_id is None:
        raise CliInputError("--target must identify a Reddit comment")
    subreddit = match.group("subreddit")
    normalized_subreddit = subreddit.casefold() if subreddit else None
    canonical = _canonical_reddit_path(
        subreddit=normalized_subreddit,
        post_id=post_id,
        slug=match.group("slug") or "-",
        comment_id=comment_id.casefold() if expected_kind == "t1" and comment_id else None,
    )
    return RedditTarget(
        urlunsplit(("https", "www.reddit.com", canonical, "", "")),
        f"t3_{post_id}",
        f"t1_{comment_id.casefold()}" if expected_kind == "t1" and comment_id else None,
        normalized_subreddit,
    )


def _resolve_target_subreddit(supplied: str | None, derived: str | None) -> str | None:
    normalized = _normalize_subreddit(supplied) if supplied is not None else None
    if normalized is not None and derived is not None and normalized != derived.casefold():
        raise CliInputError("--subreddit does not match the subreddit in --target")
    return normalized or derived


def _scheduler_context(settings: Settings):
    """Return a context manager for the action lifecycle service."""

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def context():
        async with Database(settings.database_url) as database:
            actions = ActionRepository(database.session_factory)
            content = ContentRepository(database.session_factory)
            yield SchedulerService(
                settings,
                actions,
                approved_generated_action_lookup=content,
            )

    return context()


def _persist_and_report(settings: Settings, payload: Mapping[str, Any], scheduled_at: str | None) -> None:
    parsed_schedule = _optional_datetime(scheduled_at, option_name="--scheduled-at")
    action = SocialAction.model_validate(payload)
    async def operation() -> SocialAction:
        async with _scheduler_context(settings) as scheduler:
            return await scheduler.create_action(action, scheduled_at=parsed_schedule)
    created = _run_async(operation())
    _emit_json({"operation": "create", "action": _action_summary(created)})


def _print_action_table(actions: list[SocialAction]) -> None:
    if not actions:
        typer.echo("No actions found.")
        return
    headers = ("ID", "TYPE", "PLATFORM", "ACCOUNT", "STATUS", "SCHEDULED", "ATTEMPTS")
    rows = [
        (str(a.id), a.action_type.value, a.platform.value, _safe_line(a.account_name, limit=32), a.status.value, _format_datetime(a.scheduled_at) if a.scheduled_at else "-", f"{a.attempts}/{a.max_attempts}")
        for a in actions
    ]
    widths = [max(len(headers[i]), *(len(row[i]) for row in rows)) for i in range(len(headers))]
    typer.echo("  ".join(header.ljust(widths[i]) for i, header in enumerate(headers)))
    typer.echo("  ".join("-" * width for width in widths))
    for row in rows:
        typer.echo("  ".join(value.ljust(widths[i]) for i, value in enumerate(row)))


__all__ = [
    "ActionStatus", "CliInputError", "CliRuntimeError", "CliState", "LogLevel",
    "Platform", "RedditTarget", "SchedulePreview", "Settings", "SocialAction",
    "_action_summary", "_configure_logging", "_emit_json", "_format_datetime",
    "_interactive_terminal", "_normalize_link_url", "_normalize_subreddit",
    "_optional_datetime", "_parse_aware_datetime", "_parse_reddit_target",
    "_persist_and_report", "_print_action_table", "_prompt_required", "_required_value",
    "_redact_assignments", "_resolve_text", "_resolve_target_subreddit", "_run_async",
    "_safe_command", "_safe_exception_message", "_safe_line", "_scheduler_context",
    "_settings", "_state", "_validated_account", "load_settings",
]
