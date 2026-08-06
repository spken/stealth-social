"""End-to-end generation request orchestration."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol
from uuid import UUID, uuid4

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from bot.config import Settings
from bot.content.generator import ContentGenerator
from bot.content.models import (
    AccountContext,
    CandidateDraft,
    ContentRequest,
    GenerationStatus,
    GenerationType,
    PreparedGenerationData,
    RankingResult,
    SanitizedFailure,
    StoredCandidate,
    StoredGenerationRequest,
    ValidatedCandidate,
    ValidationResult,
)
from bot.content.prompt_builder import PROMPT_VERSION, SCHEMA_VERSION, select_strategies
from bot.content.ranking import CandidateRanker
from bot.content.validation import CandidateValidator, ValidationContext
from bot.examples.collectors.browser_common import validate_public_url
from bot.examples.models import ExampleType, TargetContext, TargetContextRequest
from bot.examples.service import ExampleService
from bot.models import ActionStatus, Platform
from bot.storage.content_repository import ContentRepository
from bot.storage.repositories import ActionRepository

logger = structlog.get_logger(__name__)

_FOREGROUND_LEASE_DURATION = timedelta(minutes=5)


class UnattendedApprovalService(Protocol):
    """Minimal approval surface used after a request has been generated."""

    async def approve_best_unattended(
        self, request_id: UUID, method: str = "configured_unattended"
    ) -> Any:
        ...

    async def approve_best_unattended_in_session(
        self,
        session: AsyncSession,
        request_id: UUID,
        *,
        method: str = "configured_unattended",
        owner: str | None = None,
        now: datetime | None = None,
    ) -> Any:
        ...


class UnattendedApprovalUnavailable(RuntimeError):
    """The request cannot be approved automatically and needs review."""


class GenerationLeaseLostError(RuntimeError):
    """Generation stopped because its active request lease was lost."""


@dataclass(frozen=True, slots=True)
class GenerationPipelineResult:
    request: StoredGenerationRequest
    candidates: tuple[StoredCandidate, ...]


class GenerationService:
    """Own request preparation, generation, validation, ranking, and persistence."""

    def __init__(
        self,
        settings: Settings,
        content_repository: ContentRepository,
        example_service: ExampleService,
        generator: ContentGenerator,
        validator: CandidateValidator,
        ranker: CandidateRanker,
        *,
        action_repository: ActionRepository | None = None,
        clock: Callable[[], datetime] | None = None,
        unattended_approval_service: UnattendedApprovalService | None = None,
    ) -> None:
        self._settings = settings
        self._content = content_repository
        self._examples = example_service
        self._generator = generator
        self._validator = validator
        self._ranker = ranker
        self._actions = action_repository
        self._clock = clock or (lambda: datetime.now(UTC))
        self._approvals = unattended_approval_service

    def attach_unattended_approval_service(
        self, service: UnattendedApprovalService
    ) -> None:
        """Attach promotion after the composition root builds both services."""

        self._approvals = service

    async def create(self, request: ContentRequest) -> GenerationPipelineResult:
        resolved = self._resolve_request(request)
        self._validate_request_policy(resolved)
        now = self._now()
        if (
            resolved.desired_generation_time is not None
            and resolved.desired_generation_time <= now
        ):
            raise ValueError("desired_generation_time must be in the future")
        future = resolved.desired_generation_time is not None and (
            resolved.desired_generation_time > now
        )
        if future and not self._settings.automation.allow_scheduled_generation:
            raise ValueError("scheduled generation is disabled by automation settings")
        stored = StoredGenerationRequest(
            id=resolved.id,
            generation_type=resolved.generation_type,
            platform=resolved.platform,
            account_name=resolved.account_name,
            campaign_id=resolved.campaign_id,
            status=GenerationStatus.SCHEDULED if future else GenerationStatus.PROCESSING,
            request_snapshot=_request_snapshot(resolved),
            resolved_profile=dict(resolved.resolved_parameters),
            prompt_version=PROMPT_VERSION,
            schema_version=SCHEMA_VERSION,
            desired_generation_time=resolved.desired_generation_time,
            unattended_approval_requested=resolved.unattended_approval_requested,
            claim_owner=None if future else _foreground_owner(),
            claim_expires_at=(
                None if future else now + _FOREGROUND_LEASE_DURATION
            ),
        )
        await self._content.create_generation_request(stored)
        if future:
            return GenerationPipelineResult(request=stored, candidates=())
        return await self.execute_request(
            stored.id,
            prepared_request=resolved,
            owner=stored.claim_owner,
        )

    async def schedule(self, request: ContentRequest) -> StoredGenerationRequest:
        resolved = self._resolve_request(request)
        if resolved.desired_generation_time is None:
            raise ValueError("scheduled generation requires desired_generation_time")
        if resolved.desired_generation_time <= self._now():
            raise ValueError("desired_generation_time must be in the future")
        if not self._settings.automation.allow_scheduled_generation:
            raise ValueError("scheduled generation is disabled by automation settings")
        self._validate_request_policy(resolved)
        stored = StoredGenerationRequest(
            id=resolved.id,
            generation_type=resolved.generation_type,
            platform=resolved.platform,
            account_name=resolved.account_name,
            campaign_id=resolved.campaign_id,
            status=GenerationStatus.SCHEDULED,
            request_snapshot=_request_snapshot(resolved),
            resolved_profile=dict(resolved.resolved_parameters),
            prompt_version=PROMPT_VERSION,
            schema_version=SCHEMA_VERSION,
            desired_generation_time=resolved.desired_generation_time,
            unattended_approval_requested=resolved.unattended_approval_requested,
        )
        return await self._content.create_generation_request(stored)

    async def execute_request(
        self,
        request_id: UUID,
        *,
        prepared_request: ContentRequest | None = None,
        owner: str | None = None,
        force_prepare: bool = False,
        renew_lease: bool = True,
    ) -> GenerationPipelineResult:
        stored = await self._content.get_generation_request(request_id)
        if stored is None:
            raise ValueError(f"generation request {request_id} was not found")
        owner = owner or stored.claim_owner
        if owner is None and stored.status is GenerationStatus.PROCESSING:
            owner = _foreground_owner()
            stored = await self._content.claim_foreground_generation_request(
                request_id,
                owner=owner,
                lease_duration=_FOREGROUND_LEASE_DURATION,
                now=self._now(),
            )
        if owner is None:
            raise ValueError("generation request must have an active owner before execution")

        if not renew_lease:
            return await self._execute_request(
                request_id,
                prepared_request=prepared_request,
                owner=owner,
                force_prepare=force_prepare,
            )

        lease_lost = asyncio.Event()
        renewer = asyncio.create_task(
            self._renew_generation_claim(request_id, owner, lease_lost),
            name=f"social-bot-generation-renew:{request_id}",
        )
        execution = asyncio.create_task(
            self._execute_request(
                request_id,
                prepared_request=prepared_request,
                owner=owner,
                force_prepare=force_prepare,
            ),
            name=f"social-bot-generation-execute:{request_id}",
        )
        try:
            done, _ = await asyncio.wait(
                (execution, renewer),
                return_when=asyncio.FIRST_COMPLETED,
            )
            if execution in done:
                return await execution
            if lease_lost.is_set():
                if not execution.done():
                    execution.cancel()
                try:
                    return await execution
                except asyncio.CancelledError as error:
                    raise GenerationLeaseLostError(
                        f"generation request {request_id} lease was lost"
                    ) from error
            return await execution
        except asyncio.CancelledError:
            if not execution.done():
                execution.cancel()
            await asyncio.gather(execution, return_exceptions=True)
            raise
        finally:
            if not execution.done():
                execution.cancel()
            await asyncio.gather(execution, return_exceptions=True)
            renewer.cancel()
            await asyncio.gather(renewer, return_exceptions=True)

    async def _execute_request(
        self,
        request_id: UUID,
        *,
        prepared_request: ContentRequest | None,
        owner: str,
        force_prepare: bool,
    ) -> GenerationPipelineResult:
        """Run one request; the caller owns any surrounding lease heartbeat."""

        try:
            stored = await self._content.ensure_active_generation_claim(
                request_id,
                owner,
                now=self._now(),
            )
            request = prepared_request or self._request_from_stored(stored)
            reuse_prepared = bool(stored.metadata.get("prepared")) and not force_prepare
            if not reuse_prepared:
                request, stored = await self._prepare_request(
                    stored,
                    request,
                    owner=owner,
                    now=self._now(),
                )
            else:
                request = request.model_copy(
                    update={
                        "target_context": _target_from_snapshot(stored.target_snapshot),
                        "selected_examples": stored.selected_examples,
                        "strategy_names": select_strategies(
                            request.generation_type, request.candidate_count
                        ),
                    }
                )
            result = await self._generator.generate(request)
            recent_published = await self._recent_published(request)
            write_time = self._now()
            async with self._content.transaction() as session:
                await self._content.ensure_active_generation_claim_in_session(
                    session,
                    request_id,
                    owner,
                    now=write_time,
                )
                stored_candidates = await self._persist_candidates(
                    request,
                    stored,
                    result_candidates=result.candidates,
                    result=result,
                    session=session,
                    owner=owner,
                    now=write_time,
                    recent_published=recent_published,
                )
                errors = sum(item.validation.has_errors for item in stored_candidates)
                warnings = sum(item.validation.has_warnings for item in stored_candidates)
                approval_outcome = await self._apply_approval_policy(
                    stored,
                    session=session,
                    owner=owner,
                    now=write_time,
                )
                completed = await self._content.complete_generation_request_in_session(
                    session,
                    request_id,
                    metadata={
                        **stored.metadata,
                        **result.metadata,
                        "model_name": result.model_name,
                        "latency_seconds": result.latency_seconds,
                        "prompt_tokens": result.prompt_tokens,
                        "completion_tokens": result.completion_tokens,
                        "selected_example_count": len(request.selected_examples),
                        "candidate_count": len(stored_candidates),
                        "validation_error_count": errors,
                        "validation_warning_count": warnings,
                        "rank_mode": request.resolved_parameters.get("ranking_mode", "heuristic"),
                        "approval_outcome": approval_outcome,
                        "regeneration_used_stored_examples": bool(
                            reuse_prepared and stored.metadata.get("regeneration")
                            and any(
                                not item.example.is_active
                                or item.example.is_quarantined
                                or (
                                    item.example.expires_at is not None
                                    and item.example.expires_at <= self._now()
                                )
                                for item in stored.selected_examples
                            )
                        ),
                    },
                    completed_at=write_time,
                    owner=owner,
                )
            final_candidates = tuple(await self._content.list_candidates(request_id))
            logger.info(
                "generation_completed",
                request_id=str(request_id),
                generation_type=request.generation_type.value,
                platform=request.platform.value,
                account_name=request.account_name,
                model=result.model_name,
                duration_seconds=result.latency_seconds,
                example_count=len(request.selected_examples),
                candidate_count=len(stored_candidates),
                validation_error_count=errors,
                validation_warning_count=warnings,
                status=completed.status.value,
            )
            return GenerationPipelineResult(request=completed, candidates=final_candidates)
        except asyncio.CancelledError:
            if owner is None:
                await self._safe_fail(
                    request_id,
                    "CancelledError",
                    "generation was cancelled",
                    retryable=False,
                    owner=None,
                )
            else:
                try:
                    await self._content.release_generation_claim(
                        request_id,
                        owner,
                        now=self._now(),
                    )
                except Exception:
                    logger.error(
                        "generation_claim_release_failed",
                        request_id=str(request_id),
                    )
            raise
        except Exception as error:
            await self._safe_fail(
                request_id,
                type(error).__name__,
                _safe_failure_message(error),
                retryable=_is_retryable(error),
                owner=owner,
                error=error,
            )
            raise

    async def regenerate(self, request_id: UUID) -> GenerationPipelineResult:
        stored = await self._content.get_generation_request(request_id)
        if stored is None:
            raise ValueError(f"generation request {request_id} was not found")
        owner = _foreground_owner()
        stored = await self._content.restart_generation_request(
            request_id,
            started_at=self._now(),
            owner=owner,
            lease_duration=_FOREGROUND_LEASE_DURATION,
        )
        return await self.execute_request(request_id, owner=stored.claim_owner)

    def request_from_stored(self, stored: StoredGenerationRequest) -> ContentRequest:
        """Reconstruct the trusted request snapshot for approval/revision services."""

        return self._request_from_stored(stored)

    async def rank_candidate(
        self,
        request: ContentRequest,
        candidate_id: UUID,
        ordinal: int,
        draft: CandidateDraft,
        validation: ValidationResult,
    ) -> RankingResult | None:
        ranked = await self._ranker.rank(
            [
                ValidatedCandidate(
                    id=candidate_id,
                    ordinal=ordinal,
                    draft=draft,
                    validation=validation,
                )
            ],
            request,
        )
        return ranked[0].ranking if ranked else None

    def evaluate_candidate(
        self,
        request: ContentRequest,
        draft: CandidateDraft,
        siblings: tuple[CandidateDraft, ...] = (),
        *,
        recent_published_contents: tuple[str, ...] = (),
    ) -> ValidationResult:
        context = ValidationContext(
            request=request,
            selected_examples=tuple(item.example for item in request.selected_examples),
            recent_published_contents=recent_published_contents,
            sibling_candidates=siblings,
            request_facts=tuple(item.statement for item in request.required_facts),
            allowed_subreddits=frozenset(
                self._allowed_subreddits(request.platform, request.account_name)
            ),
            community_rules=self._community_rules(request),
            supplied_urls=frozenset(
                _supplied_urls(request)
            ),
        )
        return self._validator.validate(draft, context)

    async def _prepare_request(
        self,
        stored: StoredGenerationRequest,
        request: ContentRequest,
        *,
        owner: str,
        now: datetime,
    ) -> tuple[ContentRequest, StoredGenerationRequest]:
        target = request.target_context
        if _needs_target(request):
            target = await self._examples.resolve_target(
                TargetContextRequest(
                    platform=request.platform,
                    account_name=request.account_name,
                    target_url=request.target_url,
                    target_kind=_target_kind(request.generation_type),
                    subreddit=request.subreddit,
                )
            )
            target = _inspect_target(target)
        prepared = request.model_copy(update={"target_context": target})
        selected = await self._examples.select_for_request(prepared)
        prepared = prepared.model_copy(
            update={
                "selected_examples": selected,
                "strategy_names": select_strategies(
                    prepared.generation_type, prepared.candidate_count
                ),
            }
        )
        stored = await self._content.prepare_generation_request(
            stored.id,
            PreparedGenerationData(
                target_context=target,
                selected_examples=selected,
                request_snapshot=_request_snapshot(prepared),
                resolved_profile=dict(prepared.resolved_parameters),
                prompt_version=PROMPT_VERSION,
                schema_version=SCHEMA_VERSION,
            ),
            owner=owner,
            now=now,
        )
        return prepared, stored

    async def _persist_candidates(
        self,
        request: ContentRequest,
        stored: StoredGenerationRequest,
        *,
        result_candidates: tuple[CandidateDraft, ...],
        result,
        session: AsyncSession,
        owner: str,
        now: datetime,
        recent_published: tuple[str, ...],
    ) -> list[StoredCandidate]:
        drafts = list(result_candidates)
        existing_candidates = await self._content.list_candidates_in_session(
            session, stored.id
        )
        ordinal_start = max(
            (candidate.ordinal for candidate in existing_candidates),
            default=0,
        ) + 1
        validated: list[ValidatedCandidate] = []
        for index, draft in enumerate(drafts):
            ordinal = ordinal_start + index
            siblings = tuple(
                sibling for sibling_index, sibling in enumerate(drafts)
                if sibling_index != index
            )
            validation = self.evaluate_candidate(
                request,
                draft,
                siblings,
                recent_published_contents=recent_published,
            )
            validated.append(
                ValidatedCandidate(
                    id=uuid4(),
                    ordinal=ordinal,
                    draft=draft,
                    validation=validation,
                )
            )
        ranked = await self._ranker.rank(validated, request)
        ranking_by_id = {item.candidate.id: item.ranking for item in ranked}
        candidates = [
            StoredCandidate(
                id=item.id,
                request_id=stored.id,
                ordinal=item.ordinal,
                draft=item.draft,
                model_name=result.model_name,
                generation_parameters=dict(result.resolved_parameters),
                generated_at=result.created_at,
                validation=item.validation,
                ranking=ranking_by_id.get(item.id),
                metadata={"prompt_version": PROMPT_VERSION, "schema_version": SCHEMA_VERSION},
            )
            for item in validated
        ]
        added_ids = {candidate.id for candidate in candidates}
        for candidate in candidates:
            await self._content.add_candidate_in_session(
                session,
                candidate,
                owner=owner,
                now=now,
            )
        persisted = await self._content.list_candidates_in_session(session, stored.id)
        return [candidate for candidate in persisted if candidate.id in added_ids]

    async def _recent_published(self, request: ContentRequest) -> tuple[str, ...]:
        if self._actions is None:
            return ()
        actions = await self._actions.list(
            statuses=[ActionStatus.PUBLISHED],
            platform=request.platform,
            account_name=request.account_name,
            limit=100,
        )
        return tuple(action.content for action in actions)

    def _resolve_request(self, request: ContentRequest) -> ContentRequest:
        profile = self._settings.generation_profiles.get(request.generation_type)
        values = dict(request.resolved_parameters)
        values.setdefault("model", self._settings.content_generation.model)
        values.setdefault("temperature", profile.temperature if profile and profile.temperature is not None else self._settings.content_generation.temperature)
        values.setdefault("top_p", profile.top_p if profile and profile.top_p is not None else self._settings.content_generation.top_p)
        values.setdefault("candidate_count", profile.candidate_count if profile and profile.candidate_count is not None else request.candidate_count)
        values.setdefault("ranking_mode", self._settings.content_generation.ranking_mode.value)
        values.setdefault("thinking", self._settings.content_generation.thinking)
        values.setdefault("maximum_context_examples", profile.maximum_context_examples if profile and profile.maximum_context_examples is not None else self._settings.content_generation.maximum_context_examples)
        values.setdefault("maximum_example_characters", profile.maximum_example_characters if profile and profile.maximum_example_characters is not None else self._settings.content_generation.maximum_example_characters)
        content_purpose = request.content_purpose
        if profile is not None and profile.content_purpose is not None and not values.get(
            "content_purpose_explicit", False
        ):
            content_purpose = profile.content_purpose
        return request.model_copy(
            update={
                "candidate_count": int(values["candidate_count"]),
                "content_purpose": content_purpose,
                "resolved_parameters": values,
            }
        )

    def _validate_request_policy(self, request: ContentRequest) -> None:
        if not self._settings.content_generation.enabled:
            raise ValueError("content generation is disabled by configuration")
        if (
            request.unattended_approval_requested
            and not self._settings.automation.allow_unattended_approval
        ):
            raise ValueError("approval bypass is disabled by automation settings")
        if request.unattended_approval_requested and request.desired_generation_time is not None:
            raise ValueError("approval bypass cannot be scheduled")
        accounts = self._settings.accounts.x if request.platform is Platform.X else self._settings.accounts.reddit
        account = accounts.get(request.account_name)
        if account is None or not account.enabled:
            raise ValueError(f"no enabled {request.platform.value} account named {request.account_name!r}")
        expected_context = AccountContext(
            account_name=request.account_name,
            platform=request.platform,
            identity=account.identity,
            products=tuple(account.products),
            verified_facts=tuple(account.verified_facts),
            forbidden_claims=tuple(account.forbidden_claims),
            required_disclosures=tuple(account.required_disclosures),
        )
        if request.account_context != expected_context:
            raise ValueError("account context does not match configured account ownership")
        if request.target_url is not None:
            validate_public_url(
                request.target_url,
                request.platform,
                target_kind=(
                    "comment"
                    if request.generation_type is GenerationType.REDDIT_REPLY
                    else "post"
                ),
            )
        if request.platform is Platform.REDDIT:
            reddit_account = self._settings.accounts.reddit.get(request.account_name)
            if reddit_account is None:
                raise ValueError("configured Reddit account disappeared during validation")
            if request.subreddit:
                allowed = {item.casefold() for item in reddit_account.allowed_subreddits}
                if request.subreddit.casefold() not in allowed:
                    raise ValueError("requested Reddit subreddit is not allowlisted")
            if request.content_purpose is None:
                raise ValueError("content purpose must be resolved before policy validation")
            if request.content_purpose.value == "promotional":
                rule = next(
                    (
                        value
                        for key, value in reddit_account.community_rules.items()
                        if request.subreddit
                        and key.casefold() == request.subreddit.casefold()
                    ),
                    None,
                )
                if rule is None or not rule.allow_promotional_content:
                    raise ValueError("promotional Reddit generation requires an explicit community rule")

    def _allowed_subreddits(self, platform: Platform, account_name: str) -> tuple[str, ...]:
        if platform is not Platform.REDDIT:
            return ()
        account = self._settings.accounts.reddit.get(account_name)
        return tuple(account.allowed_subreddits) if account else ()

    def _community_rules(self, request: ContentRequest):
        if request.platform is not Platform.REDDIT:
            return {}
        account = self._settings.accounts.reddit.get(request.account_name)
        return dict(account.community_rules) if account else {}

    def _request_from_stored(self, stored: StoredGenerationRequest) -> ContentRequest:
        values = dict(stored.request_snapshot)
        values.update(
            {
                "id": stored.id,
                "generation_type": stored.generation_type,
                "platform": stored.platform,
                "account_name": stored.account_name,
                "target_context": _target_from_snapshot(stored.target_snapshot),
                "selected_examples": stored.selected_examples,
                "resolved_parameters": stored.resolved_profile,
            }
        )
        return ContentRequest.model_validate(values)

    async def _safe_fail(
        self,
        request_id: UUID,
        error_type: str,
        message: str,
        *,
        retryable: bool,
        owner: str | None = None,
        error: BaseException | None = None,
    ) -> None:
        response_sha256 = getattr(error, "response_sha256", None)
        response_length = getattr(error, "response_length", None)
        response_excerpt = getattr(error, "response_excerpt", None)
        if not isinstance(response_sha256, str) or len(response_sha256) > 128:
            response_sha256 = None
        if not isinstance(response_length, int) or response_length < 0:
            response_length = None
        if not isinstance(response_excerpt, str):
            response_excerpt = None
        else:
            response_excerpt = " ".join(response_excerpt.split())[:240] or None
        try:
            await self._content.fail_generation_request(
                request_id,
                failure=SanitizedFailure(
                    error_type=error_type,
                    message=message[:500],
                    retryable=retryable,
                    response_sha256=response_sha256,
                    response_length=response_length,
                    response_excerpt=response_excerpt,
                ),
                retry_at=None,
                failed_at=self._now(),
                owner=owner,
            )
        except Exception:
            logger.error("generation_failure_persistence_failed", request_id=str(request_id))

    async def _apply_approval_policy(
        self,
        request: StoredGenerationRequest,
        *,
        session: AsyncSession,
        owner: str,
        now: datetime,
    ) -> str:
        await self._content.ensure_active_generation_claim_in_session(
            session,
            request.id,
            owner,
            now=now,
        )
        if self._approvals is None:
            if not self._settings.manual_approval:
                raise RuntimeError("unattended approval service is not configured")
            if request.unattended_approval_requested:
                raise RuntimeError("requested approval bypass is not configured")
            return "pending_manual_approval"
        if not self._settings.manual_approval:
            try:
                await self._approvals.approve_best_unattended_in_session(
                    session,
                    request.id,
                    method="configured_unattended",
                    owner=owner,
                    now=now,
                )
            except UnattendedApprovalUnavailable:
                return "pending_manual_approval"
            return "unattended_approved"
        elif request.unattended_approval_requested:
            if not self._settings.automation.allow_unattended_approval:
                raise RuntimeError("approval bypass is disabled by automation settings")
            try:
                await self._approvals.approve_best_unattended_in_session(
                    session,
                    request.id,
                    method="cli_bypass",
                    owner=owner,
                    now=now,
                )
            except UnattendedApprovalUnavailable:
                return "pending_manual_approval"
            return "cli_bypass_approved"
        return "pending_manual_approval"

    async def _renew_generation_claim(
        self,
        request_id: UUID,
        owner: str,
        lease_lost: asyncio.Event,
    ) -> None:
        """Renew only foreground work; queue workers own their own heartbeat."""

        interval = max(
            0.1,
            min(5.0, _FOREGROUND_LEASE_DURATION.total_seconds() / 3),
        )
        while True:
            try:
                await asyncio.sleep(interval)
                renewed = await self._content.renew_generation_claim(
                    request_id,
                    owner,
                    lease_duration=_FOREGROUND_LEASE_DURATION,
                    now=self._now(),
                )
            except asyncio.CancelledError:
                raise
            except Exception as error:
                logger.warning(
                    "foreground_generation_claim_renewal_failed",
                    request_id=str(request_id),
                    error_type=type(error).__name__,
                )
                lease_lost.set()
                return
            if not renewed:
                lease_lost.set()
                return

    def _now(self) -> datetime:
        value = self._clock()
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("generation clock must return an aware timestamp")
        return value.astimezone(UTC)


def _request_snapshot(request: ContentRequest) -> dict[str, Any]:
    return request.model_dump(
        mode="json",
        exclude={
            "selected_examples",
            "target_context",
            "source_post_text",
            "source_comment_text",
            "strategy_names",
        },
    )


def _foreground_owner() -> str:
    return f"foreground-{uuid4().hex}"


def _needs_target(request: ContentRequest) -> bool:
    return request.generation_type in {
        GenerationType.X_REPLY,
        GenerationType.REDDIT_COMMENT,
        GenerationType.REDDIT_REPLY,
    }


def _target_kind(generation_type: GenerationType) -> str:
    return "comment" if generation_type is GenerationType.REDDIT_REPLY else "post"


def _inspect_target(target: TargetContext) -> TargetContext:
    from bot.content.prompt_safety import inspect_untrusted_text

    findings = []
    for text in (target.title, target.body, target.parent_text, *target.discussion_comments):
        if text:
            findings.extend(
                finding.model_dump(mode="json")
                for finding in inspect_untrusted_text(text).findings
            )
    return target.model_copy(update={"injection_findings": tuple(findings)})


def _target_from_snapshot(snapshot: dict[str, Any] | None) -> TargetContext | None:
    return TargetContext.model_validate(snapshot) if snapshot is not None else None


def _safe_failure_message(error: Exception) -> str:
    return " ".join(str(error).split())[:500] or type(error).__name__


_URL = re.compile(r"https?://[^\s)]+", re.IGNORECASE)


def _supplied_urls(request: ContentRequest) -> set[str]:
    values = (
        request.target_url,
        request.call_to_action,
        request.product_context,
        request.project_context,
        request.target_context.canonical_url if request.target_context else None,
        *request.account_context.verified_facts,
        *(item.statement for item in request.required_facts),
    )
    return {
        url.rstrip(".,!?)]")
        for value in values
        if value
        for url in _URL.findall(value)
    }


def _is_retryable(error: Exception) -> bool:
    name = type(error).__name__
    return name in {
        "OllamaUnavailableError",
        "OllamaTimeoutError",
        "PlatformUnavailableError",
    }


__all__ = [
    "GenerationLeaseLostError",
    "GenerationPipelineResult",
    "GenerationService",
    "UnattendedApprovalUnavailable",
]
