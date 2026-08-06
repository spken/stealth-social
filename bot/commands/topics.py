"""Transparent topic discovery and topic-to-generation commands."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Annotated, Any
from uuid import UUID

import typer

from bot.commands.common import (
    _emit_json,
    _format_datetime,
    _interactive_terminal,
    _normalize_subreddit,
    _run_async,
    _safe_command,
    _settings,
)
from bot.commands.generate import (
    CliGenerationType,
    GenerationOptionValues,
    parse_purpose,
    resolve_generation_options,
)
from bot.content.models import GenerationType
from bot.content.runtime import content_runtime
from bot.examples.collectors.browser_common import validate_public_url
from bot.models import Platform

topics_app = typer.Typer(
    help="Discover transparent themes from stored examples and generate from them.",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
    rich_markup_mode=None,
)


def _topic_summary(topic) -> dict[str, object]:
    return {
        "id": str(topic.id),
        "platform": topic.platform.value,
        "label": topic.label,
        "keywords": list(topic.keywords),
        "support_count": topic.support_count,
        "distinct_source_count": topic.distinct_source_count,
        "supporting_example_ids": [str(item) for item in topic.supporting_example_ids[:12]],
        "median_recency": (
            _format_datetime(topic.median_recency) if topic.median_recency else None
        ),
        "expires_at": _format_datetime(topic.expires_at) if topic.expires_at else None,
    }


async def _discover(settings, platform: Platform, since: datetime):
    async with content_runtime(settings) as runtime:
        return await runtime.topic_service.discover(platform, since=since)


@topics_app.command("discover")
@_safe_command
def discover_topics_command(
    platform: Annotated[Platform, typer.Option("--platform")],
    since_hours: Annotated[int, typer.Option("--since-hours", min=1, max=8760)] = 720,
    as_json: Annotated[bool, typer.Option("--json", help="Emit safe JSON rows.")] = False,
) -> None:
    """Discover repeated themes from recent stored examples."""

    since = datetime.now(UTC) - timedelta(hours=since_hours)
    topics = _run_async(_discover(_settings(), platform, since))
    payload = [_topic_summary(topic) for topic in topics]
    if as_json:
        _emit_json(payload)
    else:
        for topic in payload:
            supporting_ids = topic["supporting_example_ids"]
            examples = (
                ",".join(str(item) for item in supporting_ids)
                if isinstance(supporting_ids, list)
                else ""
            )
            typer.echo(
                f"{topic['id']} {topic['label']} support={topic['support_count']} "
                f"sources={topic['distinct_source_count']} "
                f"examples={examples}"
            )


async def _list_topics(settings, platform: Platform | None):
    async with content_runtime(settings) as runtime:
        return await runtime.topic_service.list(platform)


@topics_app.command("list")
@_safe_command
def list_topics_command(
    platform: Annotated[Platform | None, typer.Option("--platform")] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Emit safe JSON rows.")] = False,
) -> None:
    """List active persisted topics only."""

    topics = _run_async(_list_topics(_settings(), platform))
    payload = [_topic_summary(topic) for topic in topics]
    if as_json:
        _emit_json(payload)
    else:
        for topic in payload:
            typer.echo(
                f"{topic['id']} {topic['label']} support={topic['support_count']} "
                f"sources={topic['distinct_source_count']}"
            )


def _topic_overrides(
    options: GenerationOptionValues,
    *,
    target: str | None,
    subreddit: str | None,
) -> dict[str, Any]:
    return {
        "content_purpose": parse_purpose(options.purpose),
        "goal": options.goal,
        "product_context": options.product_context,
        "project_context": options.project_context,
        "target_audience": options.target_audience,
        "tone": options.tone,
        "desired_length": options.desired_length,
        "call_to_action": options.call_to_action,
        "required_facts": tuple(
            {"statement": item} for item in options.required_facts
        ),
        "forbidden_claims": options.forbidden_claims,
        "forbidden_phrases": options.forbidden_phrases,
        "keywords": options.keywords,
        "additional_instructions": options.additional_instructions,
        "candidate_count": options.candidate_count,
        "profile_name": options.profile_name,
        "campaign_id": options.campaign_id,
        "desired_generation_time": options.desired_generation_time,
        "unattended_approval_requested": options.bypass_approval,
        "target_url": target,
        "subreddit": subreddit,
        "resolved_parameters": {
            "content_purpose_explicit": options.purpose is not None,
        },
    }


async def _generate_from_topic(
    settings,
    topic_id: UUID,
    generation_type: GenerationType,
    account: str,
    overrides: dict[str, Any],
    *,
    review: bool,
) -> None:
    from bot.commands.candidates import interactive_review

    async with content_runtime(settings) as runtime:
        result = await runtime.topic_service.create_generation_request(
            topic_id,
            generation_type,
            account,
            overrides,
        )
        typer.echo(f"request_id={result.request.id}")
        typer.echo(f"status={result.request.status.value}")
        for candidate in result.candidates:
            typer.echo(f"candidate_id={candidate.id} status={candidate.approval_status.value}")
        if review and result.candidates:
            await interactive_review(runtime, result.request.id)


def _run_topic_command(
    settings,
    topic_id: UUID,
    action_type: CliGenerationType,
    account: str,
    options: GenerationOptionValues,
    *,
    target: str | None,
    subreddit: str | None,
) -> None:
    should_review = (
        not options.no_review
        and not options.bypass_approval
        and settings.manual_approval
        and _interactive_terminal()
    )
    overrides = _topic_overrides(options, target=target, subreddit=subreddit)
    overrides["candidate_count"] = (
        options.candidate_count or settings.content_generation.candidate_count
    )
    _run_async(
        _generate_from_topic(
            settings,
            topic_id,
            action_type.domain_type,
            account,
            overrides,
            review=should_review,
        )
    )


@topics_app.command("generate")
@_safe_command
def generate_from_topic_command(
    topic_id: Annotated[UUID, typer.Argument(help="Active topic UUID.")],
    action_type: Annotated[CliGenerationType, typer.Option("--action-type")],
    account: Annotated[str, typer.Option("--account")],
    subreddit: Annotated[str | None, typer.Option("--subreddit")] = None,
    target: Annotated[str | None, typer.Option("--target")] = None,
    purpose: Annotated[str | None, typer.Option("--purpose")] = None,
    goal: Annotated[str | None, typer.Option("--goal")] = None,
    product_context: Annotated[str | None, typer.Option("--product-context")] = None,
    project_context: Annotated[str | None, typer.Option("--project-context")] = None,
    target_audience: Annotated[str | None, typer.Option("--target-audience")] = None,
    tone: Annotated[str | None, typer.Option("--tone")] = None,
    desired_length: Annotated[str | None, typer.Option("--desired-length")] = None,
    call_to_action: Annotated[str | None, typer.Option("--call-to-action")] = None,
    fact: Annotated[list[str] | None, typer.Option("--fact")] = None,
    forbidden_claim: Annotated[list[str] | None, typer.Option("--forbidden-claim")] = None,
    forbidden_phrase: Annotated[list[str] | None, typer.Option("--forbidden-phrase")] = None,
    keyword: Annotated[list[str] | None, typer.Option("--keyword")] = None,
    additional_instructions: Annotated[str | None, typer.Option("--additional-instructions")] = None,
    candidate_count: Annotated[int | None, typer.Option("--candidate-count", min=1, max=10)] = None,
    profile: Annotated[str | None, typer.Option("--profile")] = None,
    campaign_id: Annotated[str | None, typer.Option("--campaign-id")] = None,
    generate_at: Annotated[str | None, typer.Option("--generate-at")] = None,
    no_review: Annotated[bool, typer.Option("--no-review")] = False,
    bypass_approval: Annotated[bool, typer.Option("--bypass-approval")] = False,
) -> None:
    """Create a normal generation request using a persisted topic's examples."""

    settings = _settings()
    options = resolve_generation_options(
        purpose=purpose,
        goal=goal,
        product_context=product_context,
        project_context=project_context,
        target_audience=target_audience,
        tone=tone,
        desired_length=desired_length,
        call_to_action=call_to_action,
        required_facts=fact,
        forbidden_claims=forbidden_claim,
        forbidden_phrases=forbidden_phrase,
        keywords=keyword,
        additional_instructions=additional_instructions,
        candidate_count=candidate_count,
        profile_name=profile,
        campaign_id=campaign_id,
        generate_at=generate_at,
        no_review=no_review,
        bypass_approval=bypass_approval,
    )
    if action_type.domain_type is GenerationType.REDDIT_POST and not subreddit:
        raise ValueError("--subreddit is required for Reddit post generation")
    if subreddit:
        subreddit = _normalize_subreddit(subreddit)
    if action_type.domain_type in {
        GenerationType.X_REPLY,
        GenerationType.REDDIT_COMMENT,
        GenerationType.REDDIT_REPLY,
    } and not target:
        raise ValueError("--target is required for comment/reply generation")
    if target:
        platform = (
            Platform.X
            if action_type.domain_type in {GenerationType.X_POST, GenerationType.X_REPLY}
            else Platform.REDDIT
        )
        target = validate_public_url(
            target,
            platform,
            target_kind=(
                "comment"
                if action_type.domain_type is GenerationType.REDDIT_REPLY
                else "post" if platform is Platform.REDDIT else None
            ),
        )
    _run_topic_command(
        settings,
        topic_id,
        action_type,
        account,
        options,
        target=target,
        subreddit=subreddit,
    )


def register_topic_commands(app: typer.Typer) -> None:
    app.add_typer(topics_app, name="topics")


__all__ = ["register_topic_commands", "topics_app"]
