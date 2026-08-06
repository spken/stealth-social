"""Campaign-scoped autopost recovery and cooldown orchestration."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Protocol
from uuid import UUID, uuid4

from bot.autopost.models import AutopostOutcome, AutopostResult
from bot.config import AutopostCampaignSettings, Settings
from bot.autopost.requests import build_autopost_request
from bot.autopost.topics import select_campaign_topic
from bot.content.models import (
    ContentRequest,
    GenerationStatus,
    StoredCandidate,
    StoredGenerationRequest,
)
from bot.content.service import GenerationPipelineResult
from bot.models import ActionStatus, SocialAction
from bot.storage.content_repository import ContentRepository
from bot.storage.repositories import AccountStateRepository, ActionRepository
from bot.worker import ActionExecutionReport, ExecutionDisposition


class GenerationGateway(Protocol):
    async def create(self, request: ContentRequest) -> GenerationPipelineResult: ...

    async def regenerate(self, request_id: UUID) -> GenerationPipelineResult: ...

    async def execute_request(
        self,
        request_id: UUID,
        *,
        owner: str | None = None,
    ) -> GenerationPipelineResult: ...


class SchedulerGateway(Protocol):
    async def schedule(
        self,
        action_id: UUID | str,
        *,
        scheduled_at: datetime | None = None,
    ) -> SocialAction: ...


class WorkerGateway(Protocol):
    async def execute_now(self, action_id: UUID | str) -> ActionExecutionReport: ...


@dataclass(frozen=True, slots=True)
class _CampaignItem:
    request: StoredGenerationRequest
    candidate: StoredCandidate | None
    action: SocialAction | None


class AutopostService:
    """Run one validated campaign while reusing persisted work safely."""

    def __init__(
        self,
        settings: Settings,
        *,
        content_repository: ContentRepository,
        action_repository: ActionRepository,
        account_state_repository: AccountStateRepository,
        generation_service: GenerationGateway,
        scheduler: SchedulerGateway,
        worker: WorkerGateway,
        clock: Callable[[], datetime] | None = None,
        sleeper: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        self._settings = settings
        self._content = content_repository
        self._actions = action_repository
        self._account_states = account_state_repository
        self._generation = generation_service
        self._scheduler = scheduler
        self._worker = worker
        self._clock = clock or (lambda: datetime.now(UTC))
        self._sleeper = sleeper or self._default_sleep

    async def run(self, campaign_id: str) -> AutopostResult:
        campaign = self._settings.autopost_campaigns.get(campaign_id)
        if campaign is None:
            return self._result(
                campaign_id,
                None,
                "configuration_error",
                attention_reason="campaign_not_found",
            )
        preflight = await self._preflight(campaign_id, campaign)
        if preflight is not None:
            return preflight

        items_or_result = await self._load_campaign_items(campaign_id, campaign)
        if isinstance(items_or_result, AutopostResult):
            return items_or_result
        items = items_or_result

        ambiguous = next(
            (
                item
                for item in items
                if item.action is not None
                and item.action.status is not ActionStatus.PUBLISHED
                and item.action.external_dispatch_started_at is not None
            ),
            None,
        )
        if ambiguous is not None:
            return self._result(
                campaign_id,
                campaign,
                "attention_required",
                item=ambiguous,
                attention_reason="ambiguous_external_dispatch",
            )

        resumable = [
            item for item in items if item.action is not None and self._is_resumable(item.action)
        ]
        if len(resumable) > 1:
            return self._result(
                campaign_id,
                campaign,
                "attention_required",
                attention_reason="multiple_resumable_actions",
            )
        if resumable:
            return await self._resume_item(campaign_id, campaign, resumable[0])

        generation_outcome = await self._recover_generation(
            campaign_id,
            campaign,
            items,
        )
        if generation_outcome is not None:
            return generation_outcome

        cooldown = self._cooldown_result(campaign_id, campaign, items)
        if cooldown is not None:
            return cooldown

        return await self._start_generation(campaign_id, campaign, items)

    async def _preflight(
        self,
        campaign_id: str,
        campaign: AutopostCampaignSettings,
    ) -> AutopostResult | None:
        if not campaign.enabled:
            return self._result(
                campaign_id,
                campaign,
                "configuration_error",
                attention_reason="campaign_disabled",
            )
        if self._settings.dry_run:
            return self._result(
                campaign_id,
                campaign,
                "configuration_error",
                attention_reason="dry_run_enabled",
            )
        if not self._settings.content_generation.enabled:
            return self._result(
                campaign_id,
                campaign,
                "configuration_error",
                attention_reason="content_generation_disabled",
            )
        if not self._settings.automation.allow_unattended_approval:
            return self._result(
                campaign_id,
                campaign,
                "configuration_error",
                attention_reason="unattended_approval_disabled",
            )
        if not self._settings.automation.allow_unattended_publishing:
            return self._result(
                campaign_id,
                campaign,
                "configuration_error",
                attention_reason="unattended_publishing_disabled",
            )
        if self._settings.global_pause:
            return self._result(
                campaign_id,
                campaign,
                "attention_required",
                attention_reason="global_pause",
            )
        if await self._account_states.is_paused(
            campaign.platform,
            campaign.account,
            now=self._now(),
        ):
            return self._result(
                campaign_id,
                campaign,
                "attention_required",
                attention_reason="account_paused",
            )
        return None

    async def _load_campaign_items(
        self,
        campaign_id: str,
        campaign: AutopostCampaignSettings,
    ) -> list[_CampaignItem] | AutopostResult:
        requests = await self._content.list_generation_requests_for_campaign(
            campaign_id,
            limit=1000,
        )
        items: list[_CampaignItem] = []
        for request in requests:
            candidate = await self._content.get_approved_candidate_for_request(request.id)
            action = None
            if candidate is not None and candidate.social_action_id is not None:
                action = await self._actions.get(candidate.social_action_id)
                if action is None:
                    return self._result(
                        campaign_id,
                        campaign,
                        "attention_required",
                        request=request,
                        candidate=candidate,
                        attention_reason="approved_action_missing",
                    )
            items.append(_CampaignItem(request, candidate, action))
        return items

    async def _recover_generation(
        self,
        campaign_id: str,
        campaign: AutopostCampaignSettings,
        items: list[_CampaignItem],
    ) -> AutopostResult | None:
        """Recover the newest unpromoted request before any cooldown or replacement."""

        pending = next((item for item in items if item.candidate is None), None)
        if pending is None:
            return None
        request = pending.request
        if request.status is GenerationStatus.PROCESSING:
            if request.claim_expires_at is None or request.claim_expires_at > self._now():
                return self._result(
                    campaign_id,
                    campaign,
                    "temporary_failure",
                    request=request,
                    attention_reason="generation_in_progress",
                    retry_at=request.claim_expires_at,
                )
            owner = f"autopost-{uuid4().hex}"
            claimed = await self._content.reclaim_expired_generation_request(
                request.id,
                owner=owner,
                lease_duration=timedelta(minutes=5),
                now=self._now(),
            )
            if claimed is None:
                return self._result(
                    campaign_id,
                    campaign,
                    "temporary_failure",
                    request=request,
                    attention_reason="generation_in_progress",
                    retry_at=request.claim_expires_at,
                )
            return await self._execute_claimed_generation(
                campaign_id,
                campaign,
                claimed,
            )

        if request.status is GenerationStatus.FAILED:
            if request.failure is not None and request.failure.retryable:
                if request.attempt_count >= self._settings.content_generation.maximum_retries:
                    return self._result(
                        campaign_id,
                        campaign,
                        "attention_required",
                        request=request,
                        attention_reason="generation_attempts_exhausted",
                    )
                try:
                    pipeline = await self._generation.regenerate(request.id)
                except Exception:
                    stored = await self._content.get_generation_request(request.id)
                    return self._generation_failure_result(
                        campaign_id,
                        campaign,
                        request=stored or request,
                    )
                return await self._continue_generated(
                    campaign_id,
                    campaign,
                    pipeline.request,
                )

        if request.status is GenerationStatus.SCHEDULED:
            return self._result(
                campaign_id,
                campaign,
                "attention_required",
                request=request,
                attention_reason="unexpected_scheduled_generation",
            )
        if request.status is GenerationStatus.PROCESSING:
            return self._result(
                campaign_id,
                campaign,
                "temporary_failure",
                request=request,
                attention_reason="generation_in_progress",
                retry_at=request.claim_expires_at,
            )
        return None

    async def _start_generation(
        self,
        campaign_id: str,
        campaign: AutopostCampaignSettings,
        items: list[_CampaignItem],
    ) -> AutopostResult:
        requests = [item.request for item in items]
        topic = select_campaign_topic(campaign.topics, requests)
        request = build_autopost_request(
            settings=self._settings,
            campaign_id=campaign_id,
            campaign=campaign,
            topic=topic,
        )
        try:
            pipeline = await self._generation.create(request)
        except Exception:
            stored = await self._content.get_generation_request(request.id)
            return self._generation_failure_result(
                campaign_id,
                campaign,
                request=stored,
            )
        return await self._continue_generated(
            campaign_id,
            campaign,
            pipeline.request,
        )

    async def _execute_claimed_generation(
        self,
        campaign_id: str,
        campaign: AutopostCampaignSettings,
        request: StoredGenerationRequest,
    ) -> AutopostResult:
        try:
            pipeline = await self._generation.execute_request(
                request.id,
                owner=request.claim_owner,
            )
        except Exception:
            stored = await self._content.get_generation_request(request.id)
            return self._generation_failure_result(
                campaign_id,
                campaign,
                request=stored or request,
            )
        return await self._continue_generated(
            campaign_id,
            campaign,
            pipeline.request,
        )

    async def _continue_generated(
        self,
        campaign_id: str,
        campaign: AutopostCampaignSettings,
        request: StoredGenerationRequest,
        *,
        topic: str | None = None,
    ) -> AutopostResult:
        candidate = await self._content.get_approved_candidate_for_request(request.id)
        if candidate is None or candidate.social_action_id is None:
            return self._result(
                campaign_id,
                campaign,
                "attention_required",
                request=request,
                candidate=candidate,
                attention_reason="no_safe_candidate",
            )
        action = await self._actions.get(candidate.social_action_id)
        if action is None:
            return self._result(
                campaign_id,
                campaign,
                "attention_required",
                request=request,
                candidate=candidate,
                attention_reason="approved_action_missing",
            )
        return await self._resume_item(
            campaign_id,
            campaign,
            _CampaignItem(request, candidate, action),
        )

    def _generation_failure_result(
        self,
        campaign_id: str,
        campaign: AutopostCampaignSettings,
        *,
        request: StoredGenerationRequest | None,
        topic: str | None = None,
    ) -> AutopostResult:
        if request is None:
            return self._result(
                campaign_id,
                campaign,
                "attention_required",
                attention_reason="generation_failed",
            )
        failure = request.failure
        if (
            failure is not None
            and failure.retryable
            and request.attempt_count < self._settings.content_generation.maximum_retries
        ):
            return self._result(
                campaign_id,
                campaign,
                "temporary_failure",
                request=request,
                retry_at=failure.retry_at,
                attention_reason="generation_failed",
            )
        reason = (
            "generation_attempts_exhausted"
            if failure is not None
            and failure.retryable
            and request.attempt_count >= self._settings.content_generation.maximum_retries
            else "generation_failed"
        )
        return self._result(
            campaign_id,
            campaign,
            "attention_required",
            request=request,
            attention_reason=reason,
        )

    async def _resume_item(
        self,
        campaign_id: str,
        campaign: AutopostCampaignSettings,
        item: _CampaignItem,
    ) -> AutopostResult:
        action = item.action
        if action is None:
            return self._result(
                campaign_id,
                campaign,
                "attention_required",
                item=item,
                attention_reason="approved_action_missing",
            )

        if action.status is ActionStatus.PROCESSING:
            if action.claim_expires_at is None or action.claim_expires_at > self._now():
                return self._result(
                    campaign_id,
                    campaign,
                    "temporary_failure",
                    item=item,
                    retry_at=action.claim_expires_at,
                    attention_reason="action_claim_active",
                )
            await self._actions.recover_stale_claims(now=self._now())
            recovered = await self._actions.get(action.id)
            if recovered is None:
                return self._result(
                    campaign_id,
                    campaign,
                    "attention_required",
                    item=item,
                    attention_reason="approved_action_missing",
                )
            if recovered.external_dispatch_started_at is not None:
                return self._result(
                    campaign_id,
                    campaign,
                    "attention_required",
                    item=item,
                    attention_reason="ambiguous_external_dispatch",
                )
            if recovered.status is ActionStatus.PROCESSING:
                return self._result(
                    campaign_id,
                    campaign,
                    "temporary_failure",
                    item=item,
                    retry_at=recovered.claim_expires_at,
                    attention_reason="action_claim_active",
                )
            if not self._is_resumable(recovered):
                return self._result(
                    campaign_id,
                    campaign,
                    "attention_required",
                    item=_CampaignItem(item.request, item.candidate, recovered),
                    attention_reason="terminal_action_failure",
                )
            item = _CampaignItem(item.request, item.candidate, recovered)
            action = recovered

        if action.status is ActionStatus.DRAFT:
            action = await self._scheduler.schedule(action.id)
        due_at = self._due_at(action)
        delay = max(0.0, (due_at - self._now()).total_seconds())
        if delay > 0:
            await self._sleeper(delay)

        report = await self._worker.execute_now(action.id)
        if report.disposition is ExecutionDisposition.PUBLISHED:
            return self._published_result(campaign_id, campaign, item, report)
        if report.disposition is ExecutionDisposition.RETRY_SCHEDULED:
            return self._result(
                campaign_id,
                campaign,
                "temporary_failure",
                item=item,
                action_status=report.status,
                retry_at=report.retry_at,
            )
        if report.reason == "already_published":
            current = await self._actions.get(action.id)
            if current is not None and current.status is ActionStatus.PUBLISHED:
                return self._result(
                    campaign_id,
                    campaign,
                    "published",
                    item=item,
                    action=current,
                    action_status=current.status,
                    published_url=(
                        current.external_content_url or report.external_content_url
                    ),
                )
        return self._result(
            campaign_id,
            campaign,
            "attention_required",
            item=item,
            action_status=report.status,
            attention_reason=report.reason,
        )

    def _cooldown_result(
        self,
        campaign_id: str,
        campaign: AutopostCampaignSettings,
        items: list[_CampaignItem],
    ) -> AutopostResult | None:
        published: list[tuple[datetime, _CampaignItem, SocialAction]] = []
        for item in items:
            action = item.action
            if (
                action is not None
                and action.status is ActionStatus.PUBLISHED
                and action.published_at is not None
            ):
                published.append((action.published_at, item, action))
        if not published:
            return None
        published_at, latest, action = max(published, key=lambda entry: entry[0])
        if self._now() - published_at < timedelta(
            hours=campaign.minimum_interval_hours
        ):
            return self._result(
                campaign_id,
                campaign,
                "skipped_recent_success",
                item=latest,
                action_status=action.status,
                published_url=action.external_content_url,
            )
        return None

    @staticmethod
    def _is_resumable(action: SocialAction) -> bool:
        if action.external_dispatch_started_at is not None:
            return False
        if action.status in {
            ActionStatus.DRAFT,
            ActionStatus.SCHEDULED,
            ActionStatus.PROCESSING,
        }:
            return True
        return (
            action.status is ActionStatus.FAILED
            and action.retry_available_at is not None
            and action.attempts < action.max_attempts
        )

    @staticmethod
    async def _default_sleep(seconds: float) -> None:
        import asyncio

        await asyncio.sleep(seconds)

    def _now(self) -> datetime:
        value = self._clock()
        if value.tzinfo is None or value.utcoffset() is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)

    def _due_at(self, action: SocialAction) -> datetime:
        if action.status is ActionStatus.FAILED:
            return action.retry_available_at or self._now()
        return action.scheduled_at or self._now()

    def _published_result(
        self,
        campaign_id: str,
        campaign: AutopostCampaignSettings,
        item: _CampaignItem,
        report: ActionExecutionReport,
    ) -> AutopostResult:
        return self._result(
            campaign_id,
            campaign,
            "published",
            item=item,
            action_status=report.status,
            published_url=report.external_content_url,
        )

    def _result(
        self,
        campaign_id: str,
        campaign: AutopostCampaignSettings | None,
        outcome: str,
        *,
        item: _CampaignItem | None = None,
        request: StoredGenerationRequest | None = None,
        candidate: StoredCandidate | None = None,
        action: SocialAction | None = None,
        action_status: ActionStatus | None = None,
        published_url: str | None = None,
        retry_at: datetime | None = None,
        attention_reason: str | None = None,
    ) -> AutopostResult:
        if item is not None:
            request = request or item.request
            candidate = candidate or item.candidate
            action = action or item.action
        topic = None
        if request is not None:
            raw_topic = request.request_snapshot.get("topic")
            if isinstance(raw_topic, str):
                topic = raw_topic
        return AutopostResult(
            campaign_id=campaign_id,
            outcome=AutopostOutcome(outcome),
            platform=campaign.platform if campaign is not None else None,
            account=campaign.account if campaign is not None else None,
            topic=topic,
            request_id=request.id if request is not None else None,
            candidate_id=candidate.id if candidate is not None else None,
            action_id=action.id if action is not None else None,
            action_status=action_status or (action.status if action is not None else None),
            published_url=published_url
            or (action.external_content_url if action is not None else None),
            retry_at=retry_at,
            attention_reason=attention_reason,
        )
