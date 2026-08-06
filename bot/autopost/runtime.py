"""Runtime composition for one locked autopost invocation."""

from __future__ import annotations

from pathlib import Path

from bot.autopost.locking import AutopostLockBusyError, hold_autopost_lock
from bot.autopost.models import AutopostOutcome, AutopostResult
from bot.autopost.service import AutopostService
from bot.commands.worker import build_worker
from bot.config import Settings
from bot.content.runtime import content_runtime
from bot.scheduler import SchedulerService


def _configuration_outcome(
    settings: Settings,
    campaign_id: str,
) -> AutopostResult | None:
    campaign = settings.autopost_campaigns.get(campaign_id)
    if campaign is None:
        return AutopostResult(
            campaign_id=campaign_id,
            outcome=AutopostOutcome.CONFIGURATION_ERROR,
            attention_reason="campaign_not_found",
        )
    if not campaign.enabled:
        reason = "campaign_disabled"
    elif settings.dry_run:
        reason = "dry_run_enabled"
    elif not settings.content_generation.enabled:
        reason = "content_generation_disabled"
    elif not settings.automation.allow_unattended_approval:
        reason = "unattended_approval_disabled"
    elif not settings.automation.allow_unattended_publishing:
        reason = "unattended_publishing_disabled"
    elif settings.global_pause:
        return AutopostResult(
            campaign_id=campaign_id,
            outcome=AutopostOutcome.ATTENTION_REQUIRED,
            platform=campaign.platform,
            account=campaign.account,
            attention_reason="global_pause",
        )
    else:
        return None
    return AutopostResult(
        campaign_id=campaign_id,
        outcome=AutopostOutcome.CONFIGURATION_ERROR,
        platform=campaign.platform,
        account=campaign.account,
        attention_reason=reason,
    )


async def run_autopost(
    settings: Settings,
    campaign_id: str,
    *,
    lock_path: Path | None = None,
) -> AutopostResult:
    """Run one campaign with shared content and publishing resources."""

    try:
        async with hold_autopost_lock(
            settings.database_url,
            explicit_path=lock_path,
        ):
            early = _configuration_outcome(settings, campaign_id)
            if early is not None:
                return early

            async with content_runtime(settings) as runtime:
                publishing = build_worker(
                    settings,
                    database=runtime.database,
                    browser_manager=runtime.browser_manager,
                    action_repository=runtime.action_repository,
                    account_state_repository=runtime.account_state_repository,
                    close_shared_resources=False,
                ).worker
                try:
                    scheduler = SchedulerService(
                        settings,
                        runtime.action_repository,
                        worker=publishing,
                        approved_generated_action_lookup=runtime.content_repository,
                    )
                    service = AutopostService(
                        settings,
                        content_repository=runtime.content_repository,
                        action_repository=runtime.action_repository,
                        account_state_repository=runtime.account_state_repository,
                        generation_service=runtime.generation_service,
                        scheduler=scheduler,
                        worker=publishing,
                    )
                    return await service.run(campaign_id)
                finally:
                    await publishing.close()
    except AutopostLockBusyError:
        return AutopostResult(
            campaign_id=campaign_id,
            outcome=AutopostOutcome.TEMPORARY_FAILURE,
            attention_reason="autopost_lock_busy",
        )


__all__ = ["run_autopost"]
