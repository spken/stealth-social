"""Candidate decisions, immutable revisions, and promotion to draft actions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID, uuid4

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from bot.config import Settings
from bot.content.models import (
    CandidateApprovalStatus,
    CandidateDecision,
    CandidateDraft,
    ContentRequest,
    GenerationType,
    StoredCandidate,
    StoredGenerationRequest,
)
from bot.content.service import GenerationService, UnattendedApprovalUnavailable
from bot.models import ActionStatus, ActionType, Platform, SocialAction
from bot.storage.content_repository import (
    ContentNotFoundError,
    ContentRepository,
    ContentRepositoryError,
)
from bot.storage.repositories import ActionRepository

SessionFactory = async_sessionmaker[AsyncSession]


class CandidateApprovalRequiredError(UnattendedApprovalUnavailable):
    """No safe candidate can be approved by an unattended path."""


class CandidateValidationFailureError(RuntimeError):
    """A candidate has blocking deterministic validation findings."""


@dataclass(frozen=True, slots=True)
class CandidatePromotion:
    candidate: StoredCandidate
    action: SocialAction


def social_action_from_candidate(
    request: StoredGenerationRequest,
    candidate: StoredCandidate,
    *,
    action_id: UUID,
) -> SocialAction:
    """Map generation-domain types to the existing social-action representation."""

    snapshot = request.request_snapshot
    target = request.target_snapshot or {}
    target_url = snapshot.get("target_url") or target.get("canonical_url")
    parent_post_id = target.get("parent_post_id")
    parent_comment_id = target.get("parent_comment_id")
    generation_type = request.generation_type
    if generation_type is GenerationType.X_REPLY:
        action_type = ActionType.X_POST
        parent_post_id = parent_post_id or target.get("external_id")
        platform = Platform.X
        subreddit = None
        title = None
    elif generation_type is GenerationType.X_POST:
        action_type = ActionType.X_POST
        platform = Platform.X
        subreddit = None
        title = None
        target_url = None
        parent_post_id = None
    elif generation_type is GenerationType.REDDIT_POST:
        action_type = ActionType.REDDIT_POST
        platform = Platform.REDDIT
        subreddit = snapshot.get("subreddit")
        title = candidate.draft.title
        target_url = None
        parent_post_id = None
        parent_comment_id = None
    elif generation_type is GenerationType.REDDIT_COMMENT:
        action_type = ActionType.REDDIT_COMMENT
        platform = Platform.REDDIT
        subreddit = snapshot.get("subreddit") or target.get("subreddit")
        title = None
        parent_post_id = parent_post_id or target.get("external_id")
        parent_comment_id = None
    else:
        action_type = ActionType.REDDIT_REPLY
        platform = Platform.REDDIT
        subreddit = snapshot.get("subreddit") or target.get("subreddit")
        title = None
        parent_comment_id = parent_comment_id or target.get("external_id")

    provenance = {
        "generated_candidate_id": str(candidate.id),
        "generation_request_id": str(request.id),
        "selected_example_ids": [str(item.example_id) for item in request.selected_examples],
    }
    return SocialAction(
        id=action_id,
        action_type=action_type,
        platform=platform,
        account_name=request.account_name,
        content=candidate.draft.body,
        title=title,
        subreddit=subreddit,
        target_url=target_url,
        parent_post_id=parent_post_id,
        parent_comment_id=parent_comment_id,
        status=ActionStatus.DRAFT,
        metadata=provenance,
    )


class CandidateService:
    """Own candidate decisions while leaving publishing to existing workers."""

    def __init__(
        self,
        settings: Settings,
        content_repository: ContentRepository,
        action_repository: ActionRepository,
        generation_service: GenerationService,
        session_factory: SessionFactory,
    ) -> None:
        self._settings = settings
        self._content = content_repository
        self._actions = action_repository
        self._generation = generation_service
        self._sessions = session_factory

    async def approve(
        self,
        candidate_id: UUID,
        *,
        note: str | None = None,
        method: str = "manual",
        claim_owner: str | None = None,
        claim_now: datetime | None = None,
    ) -> CandidatePromotion:
        candidate, request = await self._candidate_and_request(candidate_id)
        if candidate.approval_status is not CandidateApprovalStatus.PENDING:
            raise CandidateApprovalRequiredError("candidate is no longer pending")
        if candidate.validation.has_errors:
            raise CandidateValidationFailureError("candidate has blocking validation errors")
        decision = CandidateDecision(method=method, note=(note or "")[:500])
        action_id = uuid4()
        action = social_action_from_candidate(request, candidate, action_id=action_id)
        async with self._sessions() as session:
            async with session.begin():
                if claim_owner is not None:
                    await self._content.ensure_active_generation_claim_in_session(
                        session,
                        request.id,
                        claim_owner,
                        now=claim_now or datetime.now(UTC),
                    )
                return await self._approve_in_session(
                    session,
                    request,
                    candidate,
                    action,
                    decision,
                )

    async def promote(self, candidate_id: UUID, *, note: str | None = None) -> CandidatePromotion:
        return await self.approve(candidate_id, note=note, method="manual")

    async def approve_best_unattended(
        self,
        request_id: UUID,
        method: str = "configured_unattended",
    ) -> CandidatePromotion:
        async with self._sessions() as session:
            async with session.begin():
                return await self.approve_best_unattended_in_session(
                    session,
                    request_id,
                    method=method,
                )

    async def approve_best_unattended_in_session(
        self,
        session: AsyncSession,
        request_id: UUID,
        *,
        method: str = "configured_unattended",
        owner: str | None = None,
        now: datetime | None = None,
    ) -> CandidatePromotion:
        if not self._settings.automation.allow_unattended_approval:
            raise CandidateApprovalRequiredError(
                "unattended approval is disabled by automation settings"
            )
        if owner is not None:
            await self._content.ensure_active_generation_claim_in_session(
                session,
                request_id,
                owner,
                now=now or datetime.now(UTC),
            )
        candidates = await self._content.list_candidates_in_session(session, request_id)
        eligible = [
            candidate
            for candidate in candidates
            if candidate.approval_status is CandidateApprovalStatus.PENDING
            and not candidate.validation.has_errors
            and not candidate.validation.has_warnings
            and candidate.ranking is not None
        ]
        if not eligible:
            raise CandidateApprovalRequiredError(
                "no candidate has zero validation errors and zero warnings"
            )
        selected = max(
            eligible,
            key=lambda candidate: (
                candidate.ranking.score if candidate.ranking is not None else -1,
                -candidate.ordinal,
            ),
        )
        candidate = await self._content.get_candidate_in_session(session, selected.id)
        request = await self._content.get_generation_request_in_session(session, request_id)
        if candidate is None or request is None:
            raise ContentNotFoundError("candidate approval context was not found")
        decision = CandidateDecision(method=method)
        action = social_action_from_candidate(request, candidate, action_id=uuid4())
        return await self._approve_in_session(
            session,
            request,
            candidate,
            action,
            decision,
        )

    async def reject(
        self,
        candidate_id: UUID,
        *,
        note: str | None = None,
    ) -> StoredCandidate:
        return await self._content.reject_candidate(
            candidate_id,
            CandidateDecision(method="manual", note=(note or "")[:500]),
        )

    async def edit(
        self,
        candidate_id: UUID,
        *,
        title: str | None,
        body: str,
    ) -> StoredCandidate:
        candidate, request_record = await self._candidate_and_request(candidate_id)
        if candidate.approval_status is not CandidateApprovalStatus.PENDING:
            raise CandidateApprovalRequiredError("only pending candidates can be edited")
        request = self._generation.request_from_stored(request_record)
        draft = CandidateDraft(
            title=title if request.generation_type is GenerationType.REDDIT_POST else None,
            body=body,
            strategy=f"manual revision of {candidate.id}",
            used_example_ids=candidate.draft.used_example_ids,
        )
        siblings = tuple(
            item.draft
            for item in await self._content.list_candidates(request_record.id)
            if item.id != candidate.id and item.approval_status is CandidateApprovalStatus.PENDING
        )
        validation = self._generation.evaluate_candidate(request, draft, siblings)
        ranking = await self._generation.rank_candidate(
            request,
            uuid4(),
            candidate.ordinal,
            draft,
            validation,
        ) if not validation.has_errors else None
        revision_id = uuid4()
        revision = StoredCandidate(
            id=revision_id,
            request_id=request_record.id,
            ordinal=max(
                [item.ordinal for item in await self._content.list_candidates(request_record.id)],
                default=candidate.ordinal,
            )
            + 1,
            revision_of_candidate_id=candidate.id,
            draft=draft,
            model_name=candidate.model_name,
            generation_parameters=candidate.generation_parameters,
            validation=validation,
            ranking=ranking,
            metadata={"revision_of_candidate_id": str(candidate.id)},
        )
        async with self._sessions() as session:
            async with session.begin():
                await self._content.add_candidate_in_session(session, revision)
                await self._content.supersede_candidate_in_session(
                    session,
                    candidate.id,
                    revision.id,
                    CandidateDecision(method="manual_edit"),
                )
        stored_revision = await self._content.get_candidate(revision.id)
        if stored_revision is None:
            raise ContentRepositoryError("edited candidate was not persisted")
        return stored_revision

    async def _candidate_and_request(
        self, candidate_id: UUID
    ) -> tuple[StoredCandidate, StoredGenerationRequest]:
        candidate = await self._content.get_candidate(candidate_id)
        if candidate is None:
            raise ContentNotFoundError(f"candidate {candidate_id} was not found")
        request = await self._content.get_generation_request(candidate.request_id)
        if request is None:
            raise ContentNotFoundError(f"generation request {candidate.request_id} was not found")
        return candidate, request

    async def _approve_in_session(
        self,
        session: AsyncSession,
        request: StoredGenerationRequest,
        candidate: StoredCandidate,
        action: SocialAction,
        decision: CandidateDecision,
    ) -> CandidatePromotion:
        created_action = await self._actions.create_in_session(session, action)
        approved = await self._content.approve_candidate_in_session(
            session,
            candidate.id,
            action.id,
            decision,
        )
        return CandidatePromotion(candidate=approved, action=created_action)


__all__ = [
    "CandidateApprovalRequiredError",
    "CandidatePromotion",
    "CandidateService",
    "CandidateValidationFailureError",
    "social_action_from_candidate",
]
