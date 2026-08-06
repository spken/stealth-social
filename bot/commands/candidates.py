"""Candidate inspection and approval commands."""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

import typer

from bot.commands.common import (
    _emit_json,
    _format_datetime,
    _interactive_terminal,
    _run_async,
    _safe_command,
    _settings,
)
from bot.content.models import (
    CandidateApprovalStatus,
    GenerationType,
    StoredCandidate,
)
from bot.content.runtime import ContentRuntime, content_runtime

candidates_app = typer.Typer(
    help="Review, revise, and promote persisted generated candidates.",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
    rich_markup_mode=None,
)


def _candidate_bucket(candidate: StoredCandidate) -> tuple[int, float, int]:
    if candidate.validation.has_errors:
        bucket = 2
    elif candidate.validation.has_warnings:
        bucket = 1
    else:
        bucket = 0
    score = candidate.ranking.score if candidate.ranking is not None else -1.0
    return bucket, -score if bucket == 0 else 0.0, candidate.ordinal


def _candidate_payload(candidate: StoredCandidate, *, include_content: bool) -> dict[str, object]:
    payload: dict[str, object] = {
        "id": str(candidate.id),
        "request_id": str(candidate.request_id),
        "ordinal": candidate.ordinal,
        "approval_status": candidate.approval_status.value,
        "revision_of_candidate_id": (
            str(candidate.revision_of_candidate_id)
            if candidate.revision_of_candidate_id
            else None
        ),
        "used_example_ids": [str(item) for item in candidate.draft.used_example_ids],
        "validation": candidate.validation.model_dump(mode="json"),
        "ranking": candidate.ranking.model_dump(mode="json") if candidate.ranking else None,
        "decision": candidate.decision.model_dump(mode="json") if candidate.decision else None,
        "model_name": candidate.model_name,
        "generated_at": _format_datetime(candidate.generated_at),
    }
    if include_content:
        payload.update({"title": candidate.draft.title, "body": candidate.draft.body})
    return payload


async def _list_candidates(settings, request_id: UUID):
    async with content_runtime(settings) as runtime:
        return await runtime.content_repository.list_candidates(request_id)


@candidates_app.command("list")
@_safe_command
def list_candidates_command(
    request_id: Annotated[UUID, typer.Argument(help="Generation request UUID.")],
    as_json: Annotated[bool, typer.Option("--json", help="Emit safe JSON rows.")] = False,
) -> None:
    """List persisted candidates with valid ranked candidates first."""

    candidates = sorted(
        _run_async(_list_candidates(_settings(), request_id)),
        key=_candidate_bucket,
    )
    rows = [_candidate_payload(candidate, include_content=False) for candidate in candidates]
    if as_json:
        _emit_json(rows)
        return
    if not rows:
        typer.echo("No candidates found.")
        return
    for row in rows:
        ranking = row["ranking"] or {}
        score = ranking.get("score", "-") if isinstance(ranking, dict) else "-"
        validation = row["validation"]
        findings = validation.get("findings", []) if isinstance(validation, dict) else []
        error_count = (
            sum(
                isinstance(item, dict) and item.get("severity") == "error"
                for item in findings
            )
            if isinstance(findings, list)
            else 0
        )
        typer.echo(
            f"{row['id']} ordinal={row['ordinal']} score={score} "
            f"status={row['approval_status']} errors="
            f"{error_count}"
        )


async def _show_candidate(settings, candidate_id: UUID):
    async with content_runtime(settings) as runtime:
        candidate = await runtime.content_repository.get_candidate(candidate_id)
        if candidate is None:
            raise ValueError(f"candidate {candidate_id} was not found")
        return candidate


@candidates_app.command("show")
@_safe_command
def show_candidate_command(
    candidate_id: Annotated[UUID, typer.Argument(help="Candidate UUID.")],
) -> None:
    """Show candidate text, provenance, validation, rank, and lifecycle state."""

    candidate = _run_async(_show_candidate(_settings(), candidate_id))
    _emit_json(_candidate_payload(candidate, include_content=True), allow_user_content=True)


async def _approve_candidate(settings, candidate_id: UUID, note: str | None):
    async with content_runtime(settings) as runtime:
        return await runtime.candidate_service.approve(candidate_id, note=note)


@candidates_app.command("approve")
@_safe_command
def approve_candidate_command(
    candidate_id: Annotated[UUID, typer.Argument(help="Candidate UUID.")],
    note: Annotated[str | None, typer.Option("--note")] = None,
) -> None:
    """Approve one valid candidate and create its unscheduled draft action."""

    promotion = _run_async(_approve_candidate(_settings(), candidate_id, note))
    _emit_json(
        {
            "candidate_id": str(promotion.candidate.id),
            "approval_status": promotion.candidate.approval_status.value,
            "social_action_id": str(promotion.action.id),
            "social_action_status": promotion.action.status.value,
        }
    )


async def _reject_candidate(settings, candidate_id: UUID, note: str | None):
    async with content_runtime(settings) as runtime:
        return await runtime.candidate_service.reject(candidate_id, note=note)


@candidates_app.command("reject")
@_safe_command
def reject_candidate_command(
    candidate_id: Annotated[UUID, typer.Argument(help="Candidate UUID.")],
    note: Annotated[str | None, typer.Option("--note")] = None,
) -> None:
    """Reject one pending candidate without changing its text."""

    candidate = _run_async(_reject_candidate(_settings(), candidate_id, note))
    _emit_json(
        {
            "candidate_id": str(candidate.id),
            "approval_status": candidate.approval_status.value,
        }
    )


def _render_editor(candidate: StoredCandidate, generation_type: GenerationType) -> tuple[str | None, str] | None:
    if generation_type is GenerationType.REDDIT_POST:
        initial = f"Title: {candidate.draft.title or ''}\n---BODY---\n{candidate.draft.body}"
        edited = typer.edit(text=initial)
        if edited is None:
            return None
        if edited.count("---BODY---") != 1:
            raise ValueError("Reddit post edits require exactly one ---BODY--- delimiter")
        title_part, body = edited.split("---BODY---", 1)
        if not title_part.strip().startswith("Title:"):
            raise ValueError("Reddit post edits must start with Title:")
        title = title_part.split(":", 1)[1].strip()
        if not title:
            raise ValueError("Reddit post edits require a nonblank Title:")
        return title, body.strip()
    edited = typer.edit(text=candidate.draft.body)
    return (None, edited) if edited is not None else None


async def _edit_candidate(runtime: ContentRuntime, candidate: StoredCandidate):
    stored_request = await runtime.content_repository.get_generation_request(candidate.request_id)
    if stored_request is None:
        raise ValueError(f"generation request {candidate.request_id} was not found")
    request = runtime.generation_service.request_from_stored(stored_request)
    replacement = _render_editor(candidate, request.generation_type)
    if replacement is None:
        return None
    title, body = replacement
    return await runtime.candidate_service.edit(
        candidate.id,
        title=title,
        body=body,
    )


def _print_candidate(candidate: StoredCandidate) -> None:
    score = f"{candidate.ranking.score:.1f}/10" if candidate.ranking else "unranked"
    errors = [item.message for item in candidate.validation.findings if item.severity.value == "error"]
    warnings = [item.message for item in candidate.validation.findings if item.severity.value == "warning"]
    typer.echo(f"Candidate {candidate.ordinal}")
    typer.echo(f"Score: {score}")
    typer.echo(f"Strategy: {candidate.draft.strategy}")
    typer.echo("Validation: " + ("blocked" if errors else "passed"))
    typer.echo("Warnings: " + ("; ".join(warnings) if warnings else "none"))
    if errors:
        typer.echo("Blocking findings: " + "; ".join(errors))
    if candidate.draft.title:
        typer.echo(candidate.draft.title)
    typer.echo(candidate.draft.body)
    typer.echo()


async def interactive_review(runtime: ContentRuntime, request_id: UUID) -> None:
    """Compare persisted candidates and apply only explicit candidate decisions."""

    index = 0
    while True:
        candidates = sorted(
            await runtime.content_repository.list_candidates(request_id),
            key=_candidate_bucket,
        )
        pending = [
            candidate
            for candidate in candidates
            if candidate.approval_status is CandidateApprovalStatus.PENDING
        ]
        if not pending:
            return
        candidate = pending[min(index, len(pending) - 1)]
        _print_candidate(candidate)
        options = "[r] Reject  [e] Edit  [n] Next  [q] Quit"
        if not candidate.validation.has_errors:
            options = "[a] Approve  " + options
        choice = typer.prompt(options, default="n").strip().casefold()
        if choice == "q":
            return
        if choice == "n":
            index = (index + 1) % len(pending)
            continue
        if choice == "a":
            if candidate.validation.has_errors:
                typer.echo("This candidate has blocking findings; approval is unavailable.", err=True)
                continue
            promotion = await runtime.candidate_service.approve(candidate.id)
            typer.echo(f"approved_candidate={promotion.candidate.id} draft_action={promotion.action.id}")
            return
        if choice == "r":
            rejected = await runtime.candidate_service.reject(candidate.id)
            typer.echo(f"rejected_candidate={rejected.id}")
            index = min(index, max(0, len(pending) - 2))
            continue
        if choice == "e":
            revision = await _edit_candidate(runtime, candidate)
            if revision is not None:
                error_count = sum(
                    item.severity.value == "error"
                    for item in revision.validation.findings
                )
                warning_count = sum(
                    item.severity.value == "warning"
                    for item in revision.validation.findings
                )
                typer.echo(
                    f"revision_id={revision.id} "
                    f"score={revision.ranking.score:.1f}/10"
                    if revision.ranking
                    else f"revision_id={revision.id} score=unranked"
                )
                typer.echo(
                    f"validation_errors={error_count} validation_warnings={warning_count}"
                )
            continue
        typer.echo("Choose a, r, e, n, or q.", err=True)


async def _regenerate(settings, request_id: UUID):
    async with content_runtime(settings) as runtime:
        result = await runtime.generation_service.regenerate(request_id)
        typer.echo(f"request_id={result.request.id}")
        for candidate in result.candidates:
            typer.echo(f"candidate_id={candidate.id} status={candidate.approval_status.value}")
        if settings.manual_approval and _interactive_terminal():
            await interactive_review(runtime, request_id)


@candidates_app.command("regenerate")
@_safe_command
def regenerate_candidates_command(
    request_id: Annotated[UUID, typer.Argument(help="Generation request UUID.")],
) -> None:
    """Generate additional candidates from the stored request snapshot."""

    _run_async(_regenerate(_settings(), request_id))


def register_candidate_commands(app: typer.Typer) -> None:
    app.add_typer(candidates_app, name="candidates")


__all__ = ["candidates_app", "interactive_review", "register_candidate_commands"]
