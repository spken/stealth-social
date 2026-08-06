"""Browser-only public-example collection and management commands."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Annotated
from urllib.parse import urlsplit
from uuid import UUID

import typer

from bot.commands.common import (
    CliInputError,
    _emit_json,
    _format_datetime,
    _normalize_subreddit,
    _run_async,
    _safe_command,
    _safe_line,
    _settings,
    _validated_account,
)
from bot.config import Settings
from bot.content.runtime import content_runtime
from bot.examples.models import (
    CollectionRunStatus,
    ContentExample,
    ExampleChallengeError,
    ExampleCollectionError,
    ExampleCollectionRequest,
    ExampleListFilters,
    ExampleRateLimitedError,
    ExampleType,
)
from bot.models import Platform

examples_app = typer.Typer(
    help="Collect and manage browser-visible public content examples.",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
    rich_markup_mode=None,
)
collect_app = typer.Typer(
    help="Collect public examples from an authenticated browser profile.",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
    rich_markup_mode=None,
)
examples_app.add_typer(collect_app, name="collect")


def _override(default: Iterable[str], supplied: list[str] | None) -> tuple[str, ...]:
    return tuple(supplied) if supplied else tuple(default)


def _collection_account(
    settings: Settings,
    platform: Platform,
    supplied: str | None,
) -> str:
    if supplied and supplied.strip():
        return _validated_account(settings, platform, supplied)
    source = (
        settings.example_collection.x
        if platform is Platform.X
        else settings.example_collection.reddit
    )
    configured = source.account
    if configured:
        return _validated_account(settings, platform, configured)
    accounts = settings.accounts.x if platform is Platform.X else settings.accounts.reddit
    enabled = [name for name, account in accounts.items() if account.enabled]
    if len(enabled) == 1:
        return enabled[0]
    raise CliInputError(
        f"--account is required when zero or multiple enabled {platform.value} accounts exist"
    )


def _collection_request(
    settings: Settings,
    platform: Platform,
    account: str | None,
    *,
    queries: list[str] | None = None,
    post_urls: list[str] | None = None,
    subreddits: list[str] | None = None,
) -> ExampleCollectionRequest:
    if not settings.example_collection.enabled:
        raise CliInputError("example collection is disabled by configuration")
    account_name = _collection_account(settings, platform, account)
    common = settings.example_collection
    if platform is Platform.X:
        source = common.x
        return ExampleCollectionRequest(
            platform=platform,
            account_name=account_name,
            accounts=source.accounts,
            queries=_override(source.queries, queries),
            post_urls=_override(source.post_urls, post_urls),
            maximum_items_per_source=common.maximum_items_per_source,
            maximum_comments_per_post=common.maximum_comments_per_post,
            minimum_score=common.minimum_score,
            include_own_content=common.include_own_content,
        )
    source = common.reddit
    configured_subreddits = tuple(source.subreddits)
    override_subreddits = (
        tuple(_normalize_subreddit(value) for value in subreddits)
        if subreddits
        else configured_subreddits
    )
    return ExampleCollectionRequest(
        platform=platform,
        account_name=account_name,
        subreddits=override_subreddits,
        queries=_override(source.queries, queries),
        post_urls=_override(source.post_urls, post_urls),
        sort=source.sort,
        time_filter=source.time_filter,
        maximum_items_per_source=common.maximum_items_per_source,
        maximum_comments_per_post=common.maximum_comments_per_post,
        minimum_score=common.minimum_score,
        include_own_content=common.include_own_content,
    )


def _run_summary(run) -> dict[str, object]:
    return {
        "run_id": str(run.id),
        "platform": run.platform.value,
        "status": run.status.value,
        "collected_count": run.collected_count,
        "rejected_count": run.rejected_count,
        "duplicate_count": run.duplicate_count,
        "finished_at": _format_datetime(run.finished_at) if run.finished_at else None,
    }


def _collection_error_payload(error: BaseException) -> dict[str, object]:
    retry_after = getattr(error, "retry_after_seconds", None)
    if isinstance(error, ExampleChallengeError):
        next_action = "Resolve the human login/challenge in the browser, then rerun the collection."
    elif isinstance(error, ExampleRateLimitedError):
        next_action = "Wait for the permitted retry interval, then rerun the collection."
    else:
        next_action = "Check the configured public source and authenticated browser profile, then rerun."
    if retry_after is not None:
        next_action += f" retry_after_seconds={retry_after:g}."
    return {
        "run_id": str(getattr(error, "run_id", "unknown")),
        "saved_count": int(getattr(error, "saved_count", 0)),
        "status": getattr(error, "run_status", CollectionRunStatus.PARTIAL.value),
        "error_type": type(error).__name__,
        "next_action": next_action,
    }


async def _collect(settings: Settings, request: ExampleCollectionRequest):
    async with content_runtime(settings) as runtime:
        return await runtime.example_service.collect(request)


def _collect_command(
    settings: Settings,
    request: ExampleCollectionRequest,
    *,
    as_json: bool,
) -> None:
    try:
        run = _run_async(_collect(settings, request))
    except ExampleCollectionError as error:
        payload = _collection_error_payload(error)
        if as_json:
            _emit_json(payload)
        else:
            typer.echo(
                f"run_id={payload['run_id']} saved_count={payload['saved_count']} "
                f"status={payload['status']}"
            )
            typer.echo(f"next_action={payload['next_action']}", err=True)
        raise typer.Exit(code=1) from error
    payload = _run_summary(run)
    if as_json:
        _emit_json(payload)
    else:
        for key, value in payload.items():
            typer.echo(f"{key}={value}")


@collect_app.command("x")
@_safe_command
def collect_x_command(
    account: Annotated[str | None, typer.Option("--account", help="Configured X account.")] = None,
    query: Annotated[list[str] | None, typer.Option("--query", help="Public search query; repeatable.")] = None,
    post_url: Annotated[list[str] | None, typer.Option("--post-url", help="Public X post URL; repeatable.")] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Emit a safe JSON result.")] = False,
) -> None:
    """Collect configured or explicitly supplied public X sources."""

    settings = _settings()
    _collect_command(
        settings,
        _collection_request(settings, Platform.X, account, queries=query, post_urls=post_url),
        as_json=as_json,
    )


@collect_app.command("reddit")
@_safe_command
def collect_reddit_command(
    account: Annotated[str | None, typer.Option("--account", help="Configured Reddit account.")] = None,
    subreddit: Annotated[list[str] | None, typer.Option("--subreddit", help="Allowlisted subreddit; repeatable.")] = None,
    query: Annotated[list[str] | None, typer.Option("--query", help="Public subreddit query; repeatable.")] = None,
    post_url: Annotated[list[str] | None, typer.Option("--post-url", help="Public Reddit post URL; repeatable.")] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Emit a safe JSON result.")] = False,
) -> None:
    """Collect configured or explicitly supplied public Reddit sources."""

    settings = _settings()
    _collect_command(
        settings,
        _collection_request(
            settings,
            Platform.REDDIT,
            account,
            queries=query,
            post_urls=post_url,
            subreddits=subreddit,
        ),
        as_json=as_json,
    )


def _example_summary(example: ContentExample) -> dict[str, object]:
    parsed = urlsplit(example.source_url)
    return {
        "id": str(example.id),
        "platform": example.platform.value,
        "type": example.content_type.value,
        "subreddit": example.subreddit,
        "collected_at": _format_datetime(example.collected_at),
        "source": f"{parsed.hostname or 'unknown'}{parsed.path or '/'}",
        "own_content": example.is_own_content,
        "active": example.is_active,
        "quarantined": example.is_quarantined,
        "title_preview": _safe_line(example.title or "", limit=120),
        "body_preview": _safe_line(example.body, limit=240),
    }


async def _list_examples(settings: Settings, filters: ExampleListFilters):
    async with content_runtime(settings) as runtime:
        return await runtime.example_service.list(filters)


@examples_app.command("list")
@_safe_command
def list_examples_command(
    platform: Annotated[Platform | None, typer.Option("--platform")] = None,
    example_type: Annotated[ExampleType | None, typer.Option("--type")] = None,
    subreddit: Annotated[str | None, typer.Option("--subreddit")] = None,
    active_only: Annotated[bool, typer.Option("--active-only/--all")] = True,
    limit: Annotated[int, typer.Option("--limit", min=1, max=1000)] = 100,
    as_json: Annotated[bool, typer.Option("--json", help="Emit safe JSON rows.")] = False,
) -> None:
    """List bounded example metadata and previews without usernames or session data."""

    examples = _run_async(
        _list_examples(
            _settings(),
            ExampleListFilters(
                platform=platform,
                content_type=example_type,
                subreddit=_normalize_subreddit(subreddit) if subreddit else None,
                active_only=active_only,
                limit=limit,
            ),
        )
    )
    rows = [_example_summary(example) for example in examples]
    if as_json:
        _emit_json(rows)
        return
    if not rows:
        typer.echo("No examples found.")
        return
    for row in rows:
        typer.echo(
            f"{row['id']} {row['platform']}/{row['type']} "
            f"active={str(row['active']).lower()} source={row['source']} "
            f"preview={row['body_preview']}"
        )


async def _show_example(settings: Settings, example_id: UUID):
    async with content_runtime(settings) as runtime:
        return await runtime.example_service.show(example_id)


@examples_app.command("show")
@_safe_command
def show_example_command(
    example_id: Annotated[UUID, typer.Argument(help="Example UUID.")],
) -> None:
    """Show stored public text and safety findings without raw usernames."""

    example = _run_async(_show_example(_settings(), example_id))
    _emit_json(
        {
            "id": str(example.id),
            "platform": example.platform.value,
            "type": example.content_type.value,
            "source_url": example.source_url,
            "title": example.title,
            "body": example.body,
            "parent_text": example.parent_text,
            "subreddit": example.subreddit,
            "collected_at": _format_datetime(example.collected_at),
            "published_at": (
                _format_datetime(example.published_at)
                if example.published_at
                else None
            ),
            "active": example.is_active,
            "quarantined": example.is_quarantined,
            "injection_findings": example.injection_findings,
        },
        allow_user_content=True,
    )


async def _disable_example(settings: Settings, example_id: UUID):
    async with content_runtime(settings) as runtime:
        return await runtime.example_service.disable(example_id)


@examples_app.command("disable")
@_safe_command
def disable_example_command(
    example_id: Annotated[UUID, typer.Argument(help="Example UUID.")],
) -> None:
    """Disable an example idempotently."""

    example = _run_async(_disable_example(_settings(), example_id))
    _emit_json({"id": str(example.id), "active": example.is_active})


async def _refresh(settings: Settings, platforms: list[Platform], account: str | None):
    async with content_runtime(settings) as runtime:
        results = []
        for platform in platforms:
            account_name = _collection_account(settings, platform, account)
            results.append(
                await runtime.example_service.refresh(
                    platform,
                    account_name=account_name,
                )
            )
        return results


@examples_app.command("refresh")
@_safe_command
def refresh_examples_command(
    platform: Annotated[Platform | None, typer.Option("--platform")] = None,
    account: Annotated[str | None, typer.Option("--account")] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Emit safe JSON results.")] = False,
) -> None:
    """Refresh configured public sources for one or both platforms."""

    if account and platform is None:
        raise CliInputError("--platform is required when --account is supplied")
    platforms = [platform] if platform is not None else [Platform.X, Platform.REDDIT]
    results = _run_async(_refresh(_settings(), platforms, account))
    payload = [_run_summary(result) for result in results]
    if as_json:
        _emit_json(payload)
    else:
        for result in payload:
            typer.echo(" ".join(f"{key}={value}" for key, value in result.items()))


def register_example_commands(app: typer.Typer) -> None:
    app.add_typer(examples_app, name="examples")


__all__ = ["examples_app", "register_example_commands"]
