"""Existing action lifecycle commands and their mechanical CLI helpers."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Any, Mapping, Sequence, cast
from uuid import UUID

import click
import typer

from bot.models import ActionType, Platform, SocialAction
from bot.scheduler import ImportReport

from bot.commands.common import (
    ActionStatus,
    CliInputError,
    _action_summary,
    _emit_json,
    _interactive_terminal,
    _normalize_link_url,
    _normalize_subreddit,
    _optional_datetime,
    _parse_aware_datetime,
    _parse_reddit_target,
    _persist_and_report,
    _print_action_table,
    _prompt_required,
    _required_value,
    _resolve_text,
    _resolve_target_subreddit,
    _run_async,
    _safe_command,
    _scheduler_context,
    _settings,
    _validated_account,
)
from bot.commands.worker import execute_action

create_app = typer.Typer(
    help="Create a validated social action.",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
    rich_markup_mode=None,
)


async def _get_preview(settings: Any, action_id: UUID):
    async with _scheduler_context(settings) as scheduler:
        action = await scheduler.get(action_id)
        return action, await scheduler.preview(action_id)


async def _approve_action(settings: Any, action_id: UUID, scheduled_at):
    async with _scheduler_context(settings) as scheduler:
        return await scheduler.approve(action_id, scheduled_at=scheduled_at)


async def _schedule_action(settings: Any, action_id: UUID, scheduled_at):
    async with _scheduler_context(settings) as scheduler:
        return await scheduler.schedule(action_id, scheduled_at=scheduled_at)


async def _cancel_action(settings: Any, action_id: UUID, reason: str | None):
    async with _scheduler_context(settings) as scheduler:
        return await scheduler.cancel(action_id, reason=reason)


async def _list_action_rows(
    settings: Any,
    *,
    statuses: list[ActionStatus] | None,
    platform: Platform | None,
    account_name: str | None,
    limit: int,
    offset: int,
):
    async with _scheduler_context(settings) as scheduler:
        return await scheduler.list(
            statuses=statuses,
            platform=platform,
            account_name=account_name.strip() if account_name else None,
            limit=limit,
            offset=offset,
        )


async def _import_actions(settings: Any, actions: Sequence[Any]) -> ImportReport:
    async with _scheduler_context(settings) as scheduler:
        return await scheduler.import_actions(cast(Sequence[Mapping[str, Any]], actions))


@create_app.command("x-post")
@_safe_command
def create_x_post(
    account: Annotated[str, typer.Option("--account", help="Configured X account name.")],
    content: Annotated[str | None, typer.Option("--content", help="Post content.")] = None,
    content_file: Annotated[Path | None, typer.Option("--content-file", exists=True, dir_okay=False, readable=True, resolve_path=True)] = None,
    scheduled_at: Annotated[str | None, typer.Option("--scheduled-at", help="Aware ISO 8601 requested schedule.")] = None,
) -> None:
    settings = _settings()
    account_name = _validated_account(settings, Platform.X, account)
    resolved = _resolve_text(content, content_file, label="Post content", direct_option="--content", file_option="--content-file", required=True)
    _persist_and_report(settings, {"action_type": ActionType.X_POST, "platform": Platform.X, "account_name": account_name, "content": resolved, "metadata": {"source": "cli", "command": "create x-post"}}, scheduled_at)


@create_app.command("reddit-post")
@_safe_command
def create_reddit_post(
    account: Annotated[str, typer.Option("--account", help="Configured Reddit account name.")],
    title: Annotated[str | None, typer.Option("--title", help="Post title.")] = None,
    subreddit: Annotated[str | None, typer.Option("--subreddit", help="Destination subreddit.")] = None,
    body: Annotated[str | None, typer.Option("--body", help="Text-post body.")] = None,
    body_file: Annotated[Path | None, typer.Option("--body-file", exists=True, dir_okay=False, readable=True, resolve_path=True)] = None,
    url: Annotated[str | None, typer.Option("--url", help="Link-post URL.")] = None,
    scheduled_at: Annotated[str | None, typer.Option("--scheduled-at", help="Aware ISO 8601 requested schedule.")] = None,
) -> None:
    settings = _settings()
    account_name = _validated_account(settings, Platform.REDDIT, account)
    resolved_title = _required_value(title, "Post title", option_name="--title")
    resolved_subreddit = _normalize_subreddit(_required_value(subreddit, "Subreddit", option_name="--subreddit"))
    resolved_body = _resolve_text(body, body_file, label="Post body", direct_option="--body", file_option="--body-file", required=False)
    resolved_url = url if url and url.strip() else None
    if resolved_body is not None and resolved_url is not None:
        raise CliInputError("Reddit posts require exactly one of body/body-file or --url")
    if resolved_body is None and resolved_url is None:
        if not _interactive_terminal_for_actions():
            raise CliInputError("Reddit posts require exactly one of --body/--body-file or --url")
        post_kind = typer.prompt("Post type", type=click.Choice(["text", "link"], case_sensitive=False), default="text")
        if post_kind == "text":
            resolved_body = _prompt_required("Post body", option_name="--body")
        else:
            resolved_url = _prompt_required("Link URL", option_name="--url")
    normalized_url = _normalize_link_url(resolved_url) if resolved_url is not None else None
    _persist_and_report(settings, {"action_type": ActionType.REDDIT_POST, "platform": Platform.REDDIT, "account_name": account_name, "content": resolved_body or "", "title": resolved_title, "subreddit": resolved_subreddit, "target_url": normalized_url, "metadata": {"source": "cli", "command": "create reddit-post"}}, scheduled_at)


@create_app.command("reddit-comment")
@_safe_command
def create_reddit_comment(
    account: Annotated[str, typer.Option("--account", help="Configured Reddit account name.")],
    target: Annotated[str | None, typer.Option("--target", help="Reddit post permalink, fullname, or ID.")] = None,
    body: Annotated[str | None, typer.Option("--body", help="Comment body.")] = None,
    body_file: Annotated[Path | None, typer.Option("--body-file", exists=True, dir_okay=False, readable=True, resolve_path=True)] = None,
    subreddit: Annotated[str | None, typer.Option("--subreddit", help="Optional subreddit hint.")] = None,
    scheduled_at: Annotated[str | None, typer.Option("--scheduled-at", help="Aware ISO 8601 requested schedule.")] = None,
) -> None:
    settings = _settings()
    account_name = _validated_account(settings, Platform.REDDIT, account)
    resolved_target = _parse_reddit_target(_required_value(target, "Target post", option_name="--target"), expected_kind="t3")
    resolved_body = _resolve_text(body, body_file, label="Comment body", direct_option="--body", file_option="--body-file", required=True)
    _persist_and_report(settings, {"action_type": ActionType.REDDIT_COMMENT, "platform": Platform.REDDIT, "account_name": account_name, "content": resolved_body, "subreddit": _resolve_target_subreddit(subreddit, resolved_target.subreddit), "target_url": resolved_target.target_url, "parent_post_id": resolved_target.parent_post_id, "metadata": {"source": "cli", "command": "create reddit-comment"}}, scheduled_at)


@create_app.command("reddit-reply")
@_safe_command
def create_reddit_reply(
    account: Annotated[str, typer.Option("--account", help="Configured Reddit account name.")],
    target: Annotated[str | None, typer.Option("--target", help="Reddit comment permalink, fullname, or ID.")] = None,
    body: Annotated[str | None, typer.Option("--body", help="Reply body.")] = None,
    body_file: Annotated[Path | None, typer.Option("--body-file", exists=True, dir_okay=False, readable=True, resolve_path=True)] = None,
    subreddit: Annotated[str | None, typer.Option("--subreddit", help="Optional subreddit hint.")] = None,
    scheduled_at: Annotated[str | None, typer.Option("--scheduled-at", help="Aware ISO 8601 requested schedule.")] = None,
) -> None:
    settings = _settings()
    account_name = _validated_account(settings, Platform.REDDIT, account)
    resolved_target = _parse_reddit_target(_required_value(target, "Target comment", option_name="--target"), expected_kind="t1")
    resolved_body = _resolve_text(body, body_file, label="Reply body", direct_option="--body", file_option="--body-file", required=True)
    _persist_and_report(settings, {"action_type": ActionType.REDDIT_REPLY, "platform": Platform.REDDIT, "account_name": account_name, "content": resolved_body, "subreddit": _resolve_target_subreddit(subreddit, resolved_target.subreddit), "target_url": resolved_target.target_url, "parent_post_id": resolved_target.parent_post_id, "parent_comment_id": resolved_target.parent_comment_id, "metadata": {"source": "cli", "command": "create reddit-reply"}}, scheduled_at)


def _interactive_terminal_for_actions() -> bool:
    return _interactive_terminal()


@_safe_command
def import_json_command(
    source: Annotated[Path, typer.Argument(help="JSON file containing a list or an object with an actions list.", exists=True, dir_okay=False, readable=True, resolve_path=True)],
) -> None:
    payload = json.loads(source.read_text(encoding="utf-8"))
    actions = payload if isinstance(payload, list) else payload.get("actions") if isinstance(payload, dict) else None
    if not isinstance(actions, list):
        raise CliInputError("JSON must be a list or an object containing an actions list")
    try:
        report = _run_async(_import_actions(_settings(), actions))
    except Exception as error:
        raise RuntimeError("Import stopped before completion; earlier entries may have been created") from error
    _emit_json({"total_count": report.total_count, "created_count": report.created_count, "failure_count": report.failure_count, "created": [_action_summary(action) for action in report.created], "failures": [asdict(failure) for failure in report.failures]})
    if report.failure_count:
        raise typer.Exit(code=1)


@_safe_command
def preview_command(action_id: Annotated[UUID, typer.Argument(help="Action UUID.")]) -> None:
    action, preview = _run_async(_get_preview(_settings(), action_id))
    _emit_json({"action": action.model_dump(mode="json"), "schedule": asdict(preview)}, allow_user_content=True)


@_safe_command
def approve_command(action_id: Annotated[UUID, typer.Argument(help="Action UUID.")], at: Annotated[str | None, typer.Option("--at", help="Optional aware ISO 8601 schedule override.")] = None) -> None:
    action = _run_async(_approve_action(_settings(), action_id, _optional_datetime(at, option_name="--at")))
    _emit_json({"operation": "approve", "action": _action_summary(action)})


@_safe_command
def schedule_command(action_id: Annotated[UUID, typer.Argument(help="Action UUID.")], at: Annotated[str, typer.Option("--at", help="Aware ISO 8601 schedule time.")]) -> None:
    action = _run_async(_schedule_action(_settings(), action_id, _parse_aware_datetime(at, option_name="--at")))
    _emit_json({"operation": "schedule", "action": _action_summary(action)})


@_safe_command
def cancel_command(action_id: Annotated[UUID, typer.Argument(help="Action UUID.")], reason: Annotated[str | None, typer.Option("--reason", help="Optional private cancellation reason.")] = None) -> None:
    action = _run_async(_cancel_action(_settings(), action_id, reason))
    _emit_json({"operation": "cancel", "action": _action_summary(action)})


@_safe_command
def execute_command(action_id: Annotated[UUID, typer.Argument(help="Action UUID.")]) -> None:
    report = _run_async(execute_action(_settings(), action_id))
    _emit_json({"operation": "execute", "report": asdict(report)})
    if getattr(report, "disposition", None) is not None and report.disposition.value in {"failed", "retry_scheduled"}:
        raise typer.Exit(code=1)


@_safe_command
def list_command(
    statuses: Annotated[list[ActionStatus] | None, typer.Option("--status", help="Status filter; repeat for multiple values.")] = None,
    platform: Annotated[Platform | None, typer.Option("--platform", help="Platform filter.")] = None,
    account: Annotated[str | None, typer.Option("--account", help="Exact account-name filter.")] = None,
    limit: Annotated[int, typer.Option("--limit", min=1, max=1000, help="Maximum rows.")] = 100,
    offset: Annotated[int, typer.Option("--offset", min=0, help="Rows to skip.")] = 0,
    as_json: Annotated[bool, typer.Option("--json", help="Emit safe JSON summaries instead of a table.")] = False,
) -> None:
    actions = _run_async(_list_action_rows(_settings(), statuses=statuses, platform=platform, account_name=account, limit=limit, offset=offset))
    if as_json:
        _emit_json([_action_summary(action) for action in actions])
    else:
        _print_action_table(actions)


def register_action_commands(app: typer.Typer) -> None:
    app.add_typer(create_app, name="create")
    app.command("import-json")(import_json_command)
    app.command("preview")(preview_command)
    app.command("approve")(approve_command)
    app.command("schedule")(schedule_command)
    app.command("cancel")(cancel_command)
    app.command("execute")(execute_command)
    app.command("list")(list_command)


__all__ = ["create_app", "register_action_commands"]
