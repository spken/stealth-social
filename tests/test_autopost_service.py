"""Existing-work recovery and cooldown tests for autopost."""

from __future__ import annotations

import unittest
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

from bot.autopost.models import AutopostOutcome
from bot.autopost.service import AutopostService
from bot.content.models import (
    CandidateApprovalStatus,
    GenerationStatus,
    SanitizedFailure,
)
from bot.models import ActionStatus
from bot.worker import ActionExecutionReport, ExecutionDisposition
from tests.support import make_action, make_candidate, make_generation_request, make_settings


class AutopostServiceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.now = datetime(2030, 1, 1, tzinfo=UTC)
        self.settings = make_settings()
        self.content = AsyncMock()
        self.actions = AsyncMock()
        self.accounts = AsyncMock()
        self.generation = AsyncMock()
        self.scheduler = AsyncMock()
        self.worker = AsyncMock()
        self.sleeper = AsyncMock()
        self.content.list_generation_requests_for_campaign.return_value = []
        self.content.get_approved_candidate_for_request.return_value = None
        self.accounts.is_paused.return_value = False

    def service(self, settings=None) -> AutopostService:
        return AutopostService(
            settings or self.settings,
            content_repository=self.content,
            action_repository=self.actions,
            account_state_repository=self.accounts,
            generation_service=self.generation,
            scheduler=self.scheduler,
            worker=self.worker,
            clock=lambda: self.now,
            sleeper=self.sleeper,
        )

    def configure_action(self, action, *, request=None) -> None:
        request = request or make_generation_request()
        candidate = make_candidate(
            request.id,
            action_id=action.id,
            approval_status=CandidateApprovalStatus.APPROVED,
        )
        self.content.list_generation_requests_for_campaign.return_value = [request]
        self.content.get_approved_candidate_for_request.return_value = candidate
        self.actions.get.return_value = action

    def published_report(self, action) -> ActionExecutionReport:
        return ActionExecutionReport(
            action_id=action.id,
            platform=action.platform,
            account_name=action.account_name,
            disposition=ExecutionDisposition.PUBLISHED,
            status=ActionStatus.PUBLISHED,
            attempt=action.attempts + 1,
            reason="published",
            external_content_url="https://example.invalid/published",
        )

    async def test_unknown_campaign_returns_configuration_error(self) -> None:
        result = await self.service().run("missing")
        self.assertEqual(result.outcome, AutopostOutcome.CONFIGURATION_ERROR)
        self.assertEqual(result.attention_reason, "campaign_not_found")
        self.assertEqual(result.exit_code, 2)

    async def test_disabled_campaign_returns_configuration_error(self) -> None:
        campaign = self.settings.autopost_campaigns["daily-x"].model_copy(
            update={"enabled": False}
        )
        settings = self.settings.model_copy(
            update={"autopost_campaigns": {"daily-x": campaign}}
        )
        result = await self.service(settings).run("daily-x")
        self.assertEqual(result.attention_reason, "campaign_disabled")

    async def test_dry_run_is_rejected_before_generation(self) -> None:
        result = await self.service(self.settings.model_copy(update={"dry_run": True})).run(
            "daily-x"
        )
        self.assertEqual(result.attention_reason, "dry_run_enabled")
        self.generation.create.assert_not_awaited()

    async def test_content_generation_disabled_is_rejected_before_work(self) -> None:
        generation = self.settings.content_generation.model_copy(update={"enabled": False})
        settings = self.settings.model_copy(update={"content_generation": generation})
        result = await self.service(settings).run("daily-x")
        self.assertEqual(result.outcome, AutopostOutcome.CONFIGURATION_ERROR)
        self.assertEqual(result.attention_reason, "content_generation_disabled")
        self.generation.create.assert_not_awaited()
        self.worker.execute_now.assert_not_awaited()

    async def test_unattended_approval_capability_is_required(self) -> None:
        automation = self.settings.automation.model_copy(
            update={"allow_unattended_approval": False}
        )
        result = await self.service(
            self.settings.model_copy(update={"automation": automation})
        ).run("daily-x")
        self.assertEqual(result.attention_reason, "unattended_approval_disabled")

    async def test_unattended_publishing_capability_is_required(self) -> None:
        automation = self.settings.automation.model_copy(
            update={"allow_unattended_publishing": False}
        )
        result = await self.service(
            self.settings.model_copy(update={"automation": automation})
        ).run("daily-x")
        self.assertEqual(result.attention_reason, "unattended_publishing_disabled")

    async def test_global_pause_blocks_work(self) -> None:
        result = await self.service(self.settings.model_copy(update={"global_pause": True})).run(
            "daily-x"
        )
        self.assertEqual(result.outcome, AutopostOutcome.ATTENTION_REQUIRED)
        self.assertEqual(result.attention_reason, "global_pause")
        self.generation.create.assert_not_awaited()

    async def test_account_pause_blocks_work(self) -> None:
        self.accounts.is_paused.return_value = True
        result = await self.service().run("daily-x")
        self.assertEqual(result.attention_reason, "account_paused")
        self.generation.create.assert_not_awaited()

    async def test_draft_is_scheduled_and_published(self) -> None:
        action = make_action()
        scheduled = action.model_copy(
            update={"status": ActionStatus.SCHEDULED, "scheduled_at": self.now}
        )
        self.configure_action(action)
        self.scheduler.schedule.return_value = scheduled
        self.worker.execute_now.return_value = self.published_report(scheduled)

        result = await self.service().run("daily-x")

        self.assertEqual(result.outcome, AutopostOutcome.PUBLISHED)
        self.scheduler.schedule.assert_awaited_once_with(action.id)
        self.worker.execute_now.assert_awaited_once_with(action.id)

    async def test_scheduled_action_waits_until_persisted_time(self) -> None:
        action = make_action(
            status=ActionStatus.SCHEDULED,
            scheduled_at=self.now + timedelta(minutes=10),
        )
        self.configure_action(action)
        self.worker.execute_now.return_value = self.published_report(action)

        result = await self.service().run("daily-x")

        self.assertEqual(result.outcome, AutopostOutcome.PUBLISHED)
        self.sleeper.assert_awaited_once_with(600.0)

    async def test_retryable_action_reuses_existing_action(self) -> None:
        action = make_action(
            status=ActionStatus.FAILED,
            retry_available_at=self.now + timedelta(minutes=5),
        )
        self.configure_action(action)
        self.worker.execute_now.return_value = self.published_report(action)

        result = await self.service().run("daily-x")

        self.assertEqual(result.outcome, AutopostOutcome.PUBLISHED)
        self.worker.execute_now.assert_awaited_once_with(action.id)
        self.sleeper.assert_awaited_once_with(300.0)
        self.scheduler.schedule.assert_not_awaited()

    async def test_active_processing_claim_returns_temporary(self) -> None:
        action = make_action(
            status=ActionStatus.PROCESSING,
            claim_owner="other",
            claim_expires_at=self.now + timedelta(minutes=4),
        )
        self.configure_action(action)

        result = await self.service().run("daily-x")

        self.assertEqual(result.outcome, AutopostOutcome.TEMPORARY_FAILURE)
        self.assertEqual(result.retry_at, action.claim_expires_at)
        self.worker.execute_now.assert_not_awaited()

    async def test_expired_processing_claim_is_recovered(self) -> None:
        expired = make_action(
            status=ActionStatus.PROCESSING,
            claim_owner="old",
            claim_expires_at=self.now - timedelta(minutes=1),
        )
        recovered = make_action(
            action_id=expired.id,
            status=ActionStatus.FAILED,
            retry_available_at=self.now,
        )
        self.configure_action(expired)
        self.actions.get.side_effect = [expired, recovered]
        self.worker.execute_now.return_value = self.published_report(recovered)

        result = await self.service().run("daily-x")

        self.assertEqual(result.outcome, AutopostOutcome.PUBLISHED)
        self.actions.recover_stale_claims.assert_awaited_once_with(now=self.now)
        self.worker.execute_now.assert_awaited_once_with(recovered.id)

    async def test_ambiguous_external_dispatch_requires_attention(self) -> None:
        action = make_action(
            status=ActionStatus.PROCESSING,
            claim_owner="worker",
            claim_expires_at=self.now - timedelta(minutes=1),
            external_dispatch_started_at=self.now - timedelta(minutes=2),
        )
        self.configure_action(action)

        result = await self.service().run("daily-x")

        self.assertEqual(result.outcome, AutopostOutcome.ATTENTION_REQUIRED)
        self.assertEqual(result.attention_reason, "ambiguous_external_dispatch")
        self.worker.execute_now.assert_not_awaited()

    async def test_multiple_resumable_actions_fail_closed(self) -> None:
        first = make_action()
        second = make_action()
        first_request = make_generation_request(topic="First topic")
        second_request = make_generation_request(topic="Second topic")
        first_candidate = make_candidate(
            first_request.id,
            action_id=first.id,
            approval_status=CandidateApprovalStatus.APPROVED,
        )
        second_candidate = make_candidate(
            second_request.id,
            action_id=second.id,
            approval_status=CandidateApprovalStatus.APPROVED,
        )
        self.content.list_generation_requests_for_campaign.return_value = [
            first_request,
            second_request,
        ]
        self.content.get_approved_candidate_for_request.side_effect = [
            first_candidate,
            second_candidate,
        ]
        self.actions.get.side_effect = [first, second]

        result = await self.service().run("daily-x")

        self.assertEqual(result.attention_reason, "multiple_resumable_actions")
        self.worker.execute_now.assert_not_awaited()

    async def test_terminal_failed_action_is_historical_not_resumed(self) -> None:
        action = make_action(status=ActionStatus.FAILED)
        request = make_generation_request(topic="First topic")
        candidate = make_candidate(
            request.id,
            action_id=action.id,
            approval_status=CandidateApprovalStatus.APPROVED,
        )
        generated = make_generation_request(topic="Second topic")
        self.content.list_generation_requests_for_campaign.return_value = [request]
        self.content.get_approved_candidate_for_request.side_effect = [candidate, None]
        self.actions.get.return_value = action
        self.generation.create.return_value = self.pipeline(generated)

        result = await self.service().run("daily-x")

        self.assertEqual(result.attention_reason, "no_safe_candidate")
        self.generation.create.assert_awaited_once()
        self.scheduler.schedule.assert_not_awaited()
        self.worker.execute_now.assert_not_awaited()

    async def test_cancelled_action_is_historical_not_resumed(self) -> None:
        action = make_action(status=ActionStatus.CANCELLED)
        request = make_generation_request(topic="First topic")
        candidate = make_candidate(
            request.id,
            action_id=action.id,
            approval_status=CandidateApprovalStatus.APPROVED,
        )
        generated = make_generation_request(topic="Second topic")
        self.content.list_generation_requests_for_campaign.return_value = [request]
        self.content.get_approved_candidate_for_request.side_effect = [candidate, None]
        self.actions.get.return_value = action
        self.generation.create.return_value = self.pipeline(generated)

        result = await self.service().run("daily-x")

        self.assertEqual(result.attention_reason, "no_safe_candidate")
        self.generation.create.assert_awaited_once()
        self.scheduler.schedule.assert_not_awaited()
        self.worker.execute_now.assert_not_awaited()

    async def test_approved_candidate_with_missing_action_requires_attention(self) -> None:
        request = make_generation_request()
        candidate = make_candidate(
            request.id,
            action_id=make_action().id,
            approval_status=CandidateApprovalStatus.APPROVED,
        )
        self.content.list_generation_requests_for_campaign.return_value = [request]
        self.content.get_approved_candidate_for_request.return_value = candidate
        self.actions.get.return_value = None

        result = await self.service().run("daily-x")

        self.assertEqual(result.attention_reason, "approved_action_missing")
        self.scheduler.schedule.assert_not_awaited()
        self.worker.execute_now.assert_not_awaited()

    async def test_worker_retry_is_reported_as_temporary(self) -> None:
        action = make_action(status=ActionStatus.SCHEDULED, scheduled_at=self.now)
        self.configure_action(action)
        retry_at = self.now + timedelta(minutes=5)
        self.worker.execute_now.return_value = ActionExecutionReport(
            action_id=action.id,
            platform=action.platform,
            account_name=action.account_name,
            disposition=ExecutionDisposition.RETRY_SCHEDULED,
            status=ActionStatus.FAILED,
            attempt=1,
            reason="platform_unavailable",
            retry_at=retry_at,
        )

        result = await self.service().run("daily-x")

        self.assertEqual(result.outcome, AutopostOutcome.TEMPORARY_FAILURE)
        self.assertEqual(result.retry_at, retry_at)

    async def test_already_published_requires_authoritative_repository_state(self) -> None:
        action = make_action(status=ActionStatus.SCHEDULED, scheduled_at=self.now)
        published = action.model_copy(
            update={
                "status": ActionStatus.PUBLISHED,
                "published_at": self.now,
                "external_content_url": "https://example.invalid/existing",
            }
        )
        self.configure_action(action)
        self.actions.get.side_effect = [action, published]
        self.worker.execute_now.return_value = ActionExecutionReport(
            action_id=action.id,
            platform=action.platform,
            account_name=action.account_name,
            disposition=ExecutionDisposition.SKIPPED,
            status=ActionStatus.PUBLISHED,
            attempt=1,
            reason="already_published",
        )

        result = await self.service().run("daily-x")

        self.assertEqual(result.outcome, AutopostOutcome.PUBLISHED)
        self.assertEqual(result.published_url, "https://example.invalid/existing")

    async def test_recent_publication_skips_generation(self) -> None:
        action = make_action(
            status=ActionStatus.PUBLISHED,
            published_at=self.now - timedelta(hours=1),
        ).model_copy(update={"external_content_url": "https://example.invalid/existing"})
        self.configure_action(action)

        result = await self.service().run("daily-x")

        self.assertEqual(result.outcome, AutopostOutcome.SKIPPED_RECENT_SUCCESS)
        self.assertEqual(result.published_url, "https://example.invalid/existing")
        self.generation.create.assert_not_awaited()

    def pipeline(self, request):
        from bot.content.service import GenerationPipelineResult

        return GenerationPipelineResult(request=request, candidates=())

    async def test_empty_history_starts_first_topic_with_immediate_unattended_request(self) -> None:
        generated = make_generation_request(topic="First topic")
        self.generation.create.return_value = self.pipeline(generated)

        await self.service().run("daily-x")

        request = self.generation.create.await_args.args[0]
        self.assertEqual(request.topic, "First topic")
        self.assertTrue(request.unattended_approval_requested)
        self.assertIsNone(request.desired_generation_time)

    async def test_prior_topic_usage_selects_least_recently_attempted_topic(self) -> None:
        used = make_generation_request(topic="First topic")
        self.content.list_generation_requests_for_campaign.return_value = [used]
        generated = make_generation_request(topic="Second topic")
        self.generation.create.return_value = self.pipeline(generated)

        await self.service().run("daily-x")

        request = self.generation.create.await_args.args[0]
        self.assertEqual(request.topic, "Second topic")

    async def test_approved_draft_continues_through_scheduler_and_worker(self) -> None:
        generated = make_generation_request(topic="First topic")
        action = make_action()
        candidate = make_candidate(
            generated.id,
            action_id=action.id,
            approval_status=CandidateApprovalStatus.APPROVED,
        )
        self.generation.create.return_value = self.pipeline(generated)
        self.content.get_approved_candidate_for_request.return_value = candidate
        self.actions.get.return_value = action
        scheduled = action.model_copy(
            update={"status": ActionStatus.SCHEDULED, "scheduled_at": self.now}
        )
        self.scheduler.schedule.return_value = scheduled
        self.worker.execute_now.return_value = self.published_report(scheduled)

        result = await self.service().run("daily-x")

        self.assertEqual(result.outcome, AutopostOutcome.PUBLISHED)
        self.assertEqual(result.request_id, generated.id)
        self.assertEqual(result.candidate_id, candidate.id)
        self.assertEqual(result.action_id, action.id)
        self.assertEqual(result.published_url, "https://example.invalid/published")

    async def test_no_safe_candidate_never_schedules(self) -> None:
        generated = make_generation_request()
        self.generation.create.return_value = self.pipeline(generated)
        self.content.get_approved_candidate_for_request.return_value = None

        result = await self.service().run("daily-x")

        self.assertEqual(result.outcome, AutopostOutcome.ATTENTION_REQUIRED)
        self.assertEqual(result.attention_reason, "no_safe_candidate")
        self.scheduler.schedule.assert_not_awaited()
        self.worker.execute_now.assert_not_awaited()

    async def test_retryable_failed_generation_reuses_request_id(self) -> None:
        failed = make_generation_request(
            status=GenerationStatus.FAILED,
            attempt_count=0,
            failure=SanitizedFailure(
                error_type="OllamaUnavailable",
                message="retryable failure",
                retryable=True,
                retry_at=self.now,
            ),
        )
        self.content.list_generation_requests_for_campaign.return_value = [failed]
        self.generation.regenerate.return_value = self.pipeline(failed)

        await self.service().run("daily-x")

        self.generation.regenerate.assert_awaited_once_with(failed.id)

    async def test_failed_generation_at_bound_requires_attention(self) -> None:
        failed = make_generation_request(
            status=GenerationStatus.FAILED,
            attempt_count=self.settings.content_generation.maximum_retries,
            failure=SanitizedFailure(
                error_type="OllamaUnavailable",
                message="retryable failure",
                retryable=True,
            ),
        )
        self.content.list_generation_requests_for_campaign.return_value = [failed]

        result = await self.service().run("daily-x")

        self.assertEqual(result.attention_reason, "generation_attempts_exhausted")
        self.generation.regenerate.assert_not_awaited()

    async def test_active_generation_claim_returns_temporary(self) -> None:
        active = make_generation_request(
            status=GenerationStatus.PROCESSING,
            claim_owner="other",
            claim_expires_at=self.now + timedelta(minutes=5),
        )
        self.content.list_generation_requests_for_campaign.return_value = [active]

        result = await self.service().run("daily-x")

        self.assertEqual(result.outcome, AutopostOutcome.TEMPORARY_FAILURE)
        self.assertEqual(result.attention_reason, "generation_in_progress")
        self.assertEqual(result.retry_at, active.claim_expires_at)

    async def test_expired_generation_claim_is_adopted_and_executed(self) -> None:
        expired = make_generation_request(
            status=GenerationStatus.PROCESSING,
            claim_owner="old",
            claim_expires_at=self.now - timedelta(minutes=1),
        )
        claimed = expired.model_copy(
            update={
                "claim_owner": "autopost-test-owner",
                "claim_expires_at": self.now + timedelta(minutes=5),
                "attempt_count": expired.attempt_count + 1,
            }
        )
        self.content.list_generation_requests_for_campaign.return_value = [expired]
        self.content.reclaim_expired_generation_request.return_value = claimed
        self.generation.execute_request.return_value = self.pipeline(claimed)

        await self.service().run("daily-x")

        self.content.reclaim_expired_generation_request.assert_awaited_once()
        self.generation.execute_request.assert_awaited_once_with(
            expired.id,
            owner="autopost-test-owner",
        )

    async def test_scheduled_generation_requires_attention(self) -> None:
        scheduled = make_generation_request(
            status=GenerationStatus.SCHEDULED,
        ).model_copy(update={"desired_generation_time": self.now + timedelta(hours=1)})
        self.content.list_generation_requests_for_campaign.return_value = [scheduled]

        result = await self.service().run("daily-x")

        self.assertEqual(result.attention_reason, "unexpected_scheduled_generation")
        self.generation.execute_request.assert_not_awaited()

    async def test_generation_exception_uses_sanitized_retry_metadata(self) -> None:
        failed = make_generation_request(
            status=GenerationStatus.FAILED,
            attempt_count=0,
            failure=SanitizedFailure(
                error_type="OllamaUnavailable",
                message="safe persisted message",
                retryable=True,
                retry_at=self.now + timedelta(minutes=2),
            ),
        )
        self.generation.create.side_effect = RuntimeError("secret prompt and model output")
        self.content.get_generation_request.return_value = failed

        result = await self.service().run("daily-x")

        self.assertEqual(result.outcome, AutopostOutcome.TEMPORARY_FAILURE)
        self.assertEqual(result.attention_reason, "generation_failed")
        self.assertEqual(result.retry_at, failed.failure.retry_at)
        self.assertNotIn("secret", result.model_dump_json())

    async def test_non_retryable_generation_exception_requires_attention(self) -> None:
        failed = make_generation_request(
            status=GenerationStatus.FAILED,
            attempt_count=0,
            failure=SanitizedFailure(
                error_type="InvalidModelResponse",
                message="safe persisted message",
                retryable=False,
            ),
        )
        self.generation.create.side_effect = RuntimeError("secret response")
        self.content.get_generation_request.return_value = failed

        result = await self.service().run("daily-x")

        self.assertEqual(result.outcome, AutopostOutcome.ATTENTION_REQUIRED)
        self.assertEqual(result.attention_reason, "generation_failed")
        self.assertNotIn("secret", result.model_dump_json())
