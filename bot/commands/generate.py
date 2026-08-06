"""Generation commands and the single common request-option resolver."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Any

import typer
from pydantic import BaseModel, ConfigDict, Field

from bot.commands.common import (
    CliInputError,
    _interactive_terminal,
    _normalize_subreddit,
    _run_async,
    _safe_command,
    _settings,
    _validated_account,
)
from bot.config import Settings
from bot.content.models import (
    ContentPurpose,
    ContentRequest,
    FactRequirement,
    GenerationType,
)
from bot.content.requests import configured_account_context
from bot.content.runtime import content_runtime
from bot.examples.collectors.browser_common import validate_public_url
from bot.models import Platform


class CliGenerationType(StrEnum):
    X_POST = "x-post"
    X_REPLY = "x-reply"
    REDDIT_POST = "reddit-post"
    REDDIT_COMMENT = "reddit-comment"
    REDDIT_REPLY = "reddit-reply"

    @property
    def domain_type(self) -> GenerationType:
        return {
            CliGenerationType.X_POST: GenerationType.X_POST,
            CliGenerationType.X_REPLY: GenerationType.X_REPLY,
            CliGenerationType.REDDIT_POST: GenerationType.REDDIT_POST,
            CliGenerationType.REDDIT_COMMENT: GenerationType.REDDIT_COMMENT,
            CliGenerationType.REDDIT_REPLY: GenerationType.REDDIT_REPLY,
        }[self]


class GenerationOptionValues(BaseModel):
    """Validated values shared by every generation command."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    purpose: str | None = None
    topic: str | None = None
    goal: str | None = None
    product_context: str | None = None
    project_context: str | None = None
    target_audience: str | None = None
    tone: str | None = None
    desired_length: str | None = None
    call_to_action: str | None = None
    required_facts: tuple[str, ...] = ()
    forbidden_claims: tuple[str, ...] = ()
    forbidden_phrases: tuple[str, ...] = ()
    keywords: tuple[str, ...] = ()
    additional_instructions: str | None = None
    candidate_count: int | None = Field(default=None, ge=1, le=10)
    profile_name: str | None = None
    campaign_id: str | None = None
    desired_generation_time: datetime | None = None
    no_review: bool = False
    bypass_approval: bool = False


_PURPOSES = {
    "educational": ContentPurpose.EDUCATIONAL,
    "product-update": ContentPurpose.PRODUCT_UPDATE,
    "product_update": ContentPurpose.PRODUCT_UPDATE,
    "builder-update": ContentPurpose.BUILDER_UPDATE,
    "builder_update": ContentPurpose.BUILDER_UPDATE,
    "promotional": ContentPurpose.PROMOTIONAL,
    "organic-discussion": ContentPurpose.ORGANIC_DISCUSSION,
    "organic_discussion": ContentPurpose.ORGANIC_DISCUSSION,
    "customer-support": ContentPurpose.CUSTOMER_SUPPORT,
    "customer_support": ContentPurpose.CUSTOMER_SUPPORT,
}


def parse_purpose(value: str | None) -> ContentPurpose | None:
    if value is None:
        return None
    normalized = value.strip().casefold()
    try:
        return _PURPOSES[normalized]
    except KeyError as error:
        raise CliInputError(
            "--purpose must be one of educational, product-update, builder-update, "
            "promotional, organic-discussion, or customer-support"
        ) from error


def resolve_generation_options(
    *,
    purpose: str | None = None,
    topic: str | None = None,
    goal: str | None = None,
    product_context: str | None = None,
    project_context: str | None = None,
    target_audience: str | None = None,
    tone: str | None = None,
    desired_length: str | None = None,
    call_to_action: str | None = None,
    required_facts: list[str] | None = None,
    forbidden_claims: list[str] | None = None,
    forbidden_phrases: list[str] | None = None,
    keywords: list[str] | None = None,
    additional_instructions: str | None = None,
    candidate_count: int | None = None,
    profile_name: str | None = None,
    campaign_id: str | None = None,
    generate_at: str | None = None,
    no_review: bool = False,
    bypass_approval: bool = False,
) -> GenerationOptionValues:
    desired_time = None
    if generate_at:
        candidate = generate_at.strip()
        if candidate.endswith(("Z", "z")):
            candidate = candidate[:-1] + "+00:00"
        try:
            desired_time = datetime.fromisoformat(candidate)
        except ValueError as error:
            raise CliInputError("--generate-at must be a timezone-aware ISO timestamp") from error
        if desired_time.tzinfo is None or desired_time.utcoffset() is None:
            raise CliInputError("--generate-at must include a UTC offset or Z")
        desired_time = desired_time.astimezone(UTC)
        if desired_time <= datetime.now(UTC):
            raise CliInputError("--generate-at must be in the future")
        if bypass_approval:
            raise CliInputError("--bypass-approval cannot be combined with --generate-at")
    return GenerationOptionValues(
        purpose=purpose,
        topic=topic,
        goal=goal,
        product_context=product_context,
        project_context=project_context,
        target_audience=target_audience,
        tone=tone,
        desired_length=desired_length,
        call_to_action=call_to_action,
        required_facts=tuple(required_facts or ()),
        forbidden_claims=tuple(forbidden_claims or ()),
        forbidden_phrases=tuple(forbidden_phrases or ()),
        keywords=tuple(keywords or ()),
        additional_instructions=additional_instructions,
        candidate_count=candidate_count,
        profile_name=profile_name,
        campaign_id=campaign_id,
        desired_generation_time=desired_time,
        no_review=no_review,
        bypass_approval=bypass_approval,
    )


account_context_for = configured_account_context


def build_generation_request(
    settings: Settings,
    generation_type: GenerationType,
    *,
    account: str,
    options: GenerationOptionValues,
    target_url: str | None = None,
    subreddit: str | None = None,
    example_selection_filters: Any = None,
) -> ContentRequest:
    platform = (
        Platform.X
        if generation_type in {GenerationType.X_POST, GenerationType.X_REPLY}
        else Platform.REDDIT
    )
    account_name = _validated_account(settings, platform, account)
    if target_url is not None:
        target_kind = None
        if platform is Platform.REDDIT:
            target_kind = (
                "comment"
                if generation_type is GenerationType.REDDIT_REPLY
                else "post"
            )
        target_url = validate_public_url(
            target_url,
            platform,
            target_kind=target_kind,
        )
    if subreddit is not None:
        subreddit = _normalize_subreddit(subreddit)
    purpose = parse_purpose(options.purpose)
    raw_parameters: dict[str, object] = {
        "content_purpose_explicit": options.purpose is not None,
    }
    return ContentRequest(
        generation_type=generation_type,
        platform=platform,
        account_name=account_name,
        content_purpose=purpose,
        topic=options.topic,
        goal=options.goal,
        product_context=options.product_context,
        project_context=options.project_context,
        target_audience=options.target_audience,
        tone=options.tone,
        desired_length=options.desired_length,
        call_to_action=options.call_to_action,
        subreddit=subreddit,
        target_url=target_url,
        required_facts=tuple(FactRequirement(statement=fact) for fact in options.required_facts),
        forbidden_claims=options.forbidden_claims,
        forbidden_phrases=options.forbidden_phrases,
        keywords=options.keywords,
        additional_instructions=options.additional_instructions,
        candidate_count=options.candidate_count or settings.content_generation.candidate_count,
        profile_name=options.profile_name,
        campaign_id=options.campaign_id,
        desired_generation_time=options.desired_generation_time,
        unattended_approval_requested=options.bypass_approval,
        account_context=configured_account_context(settings, platform, account_name),
        example_selection_filters=example_selection_filters,
        resolved_parameters=raw_parameters,
    )


def _common_kwargs(
    *,
    purpose: str | None,
    topic: str | None,
    goal: str | None,
    product_context: str | None,
    project_context: str | None,
    target_audience: str | None,
    tone: str | None,
    desired_length: str | None,
    call_to_action: str | None,
    fact: list[str] | None,
    forbidden_claim: list[str] | None,
    forbidden_phrase: list[str] | None,
    keyword: list[str] | None,
    additional_instructions: str | None,
    candidate_count: int | None,
    profile: str | None,
    campaign_id: str | None,
    generate_at: str | None,
    no_review: bool,
    bypass_approval: bool,
    **_ignored: Any,
) -> GenerationOptionValues:
    return resolve_generation_options(
        purpose=purpose,
        topic=topic,
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


async def _run_request(
    settings: Settings,
    request: ContentRequest,
    *,
    review: bool,
) -> None:
    from bot.commands.candidates import interactive_review

    async with content_runtime(settings) as runtime:
        result = await runtime.generation_service.create(request)
        typer.echo(f"request_id={result.request.id}")
        typer.echo(f"status={result.request.status.value}")
        if result.candidates:
            for candidate in result.candidates:
                typer.echo(
                    f"candidate_id={candidate.id} status={candidate.approval_status.value}"
                )
        if review and result.candidates:
            await interactive_review(runtime, result.request.id)


def _run_command(
    settings: Settings,
    request: ContentRequest,
    options: GenerationOptionValues,
) -> None:
    should_review = (
        not options.no_review
        and not options.bypass_approval
        and settings.manual_approval
        and _interactive_terminal()
    )
    _run_async(_run_request(settings, request, review=should_review))


generate_app = typer.Typer(
    help="Generate reviewable local candidates without publishing.",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
    rich_markup_mode=None,
)


@generate_app.command("x-post")
@_safe_command
def generate_x_post_command(
    account: Annotated[str, typer.Option("--account")],
    topic: Annotated[str, typer.Option("--topic")],
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
    """Generate X post candidates."""

    options = _common_kwargs(**locals())
    settings = _settings()
    _run_command(settings, build_generation_request(settings, GenerationType.X_POST, account=account, options=options), options)


@generate_app.command("x-reply")
@_safe_command
def generate_x_reply_command(
    account: Annotated[str, typer.Option("--account")],
    target: Annotated[str, typer.Option("--target")],
    purpose: Annotated[str | None, typer.Option("--purpose")] = None,
    topic: Annotated[str | None, typer.Option("--topic")] = None,
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
    """Generate candidates for an X post reply."""

    options = _common_kwargs(**locals())
    settings = _settings()
    _run_command(settings, build_generation_request(settings, GenerationType.X_REPLY, account=account, options=options, target_url=target), options)


@generate_app.command("reddit-post")
@_safe_command
def generate_reddit_post_command(
    account: Annotated[str, typer.Option("--account")],
    subreddit: Annotated[str, typer.Option("--subreddit")],
    topic: Annotated[str, typer.Option("--topic")],
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
    """Generate a Reddit post."""

    options = _common_kwargs(**locals())
    settings = _settings()
    _run_command(settings, build_generation_request(settings, GenerationType.REDDIT_POST, account=account, options=options, subreddit=subreddit), options)


@generate_app.command("reddit-comment")
@_safe_command
def generate_reddit_comment_command(
    account: Annotated[str, typer.Option("--account")],
    target: Annotated[str, typer.Option("--target")],
    purpose: Annotated[str | None, typer.Option("--purpose")] = None,
    topic: Annotated[str | None, typer.Option("--topic")] = None,
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
    """Generate a Reddit comment."""

    options = _common_kwargs(**locals())
    settings = _settings()
    _run_command(settings, build_generation_request(settings, GenerationType.REDDIT_COMMENT, account=account, options=options, target_url=target), options)


@generate_app.command("reddit-reply")
@_safe_command
def generate_reddit_reply_command(
    account: Annotated[str, typer.Option("--account")],
    target: Annotated[str, typer.Option("--target")],
    purpose: Annotated[str | None, typer.Option("--purpose")] = None,
    topic: Annotated[str | None, typer.Option("--topic")] = None,
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
    """Generate a Reddit reply."""

    options = _common_kwargs(**locals())
    settings = _settings()
    _run_command(settings, build_generation_request(settings, GenerationType.REDDIT_REPLY, account=account, options=options, target_url=target), options)


def register_generation_commands(app: typer.Typer) -> None:
    app.add_typer(generate_app, name="generate")


__all__ = [
    "CliGenerationType",
    "GenerationOptionValues",
    "account_context_for",
    "configured_account_context",
    "build_generation_request",
    "generate_app",
    "parse_purpose",
    "register_generation_commands",
    "resolve_generation_options",
]
