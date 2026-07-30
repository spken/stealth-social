"""Persistent claimed-action execution runtime."""

from __future__ import annotations

import asyncio
import math
import os
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from random import SystemRandom
from typing import Protocol
from urllib.parse import urlsplit, urlunsplit
from uuid import UUID, uuid4

import structlog

from bot.browser.manager import BrowserManager
from bot.config import SafetyLimitOverrides, Settings
from bot.models import ActionResult, ActionStatus, Platform, SocialAction
from bot.platforms.base import (
    PlatformAuthenticationError,
    PlatformError,
    PlatformRateLimitError,
    SocialPlatform,
)
from bot.platforms.reddit import BrowserInteractionError, RedditAdapter
from bot.platforms.x import XAdapter
from bot.storage.database import Database
from bot.storage.repositories import (
    AccountStateRepository,
    ActionNotFoundError,
    ActionRepository,
    ClaimConflictError,
)


logger = structlog.get_logger(__name__)


class JitterSource(Protocol):
    """Injectable random source for retry jitter."""

    def uniform(self, a: float, b: float) -> float:
        """Return a floating-point value in the inclusive configured range."""

        ...


class AdapterProvider(Protocol):
    """Resolve one account-bound platform adapter."""

    async def get(
        self,
        platform: Platform,
        account_name: str,
    ) -> SocialPlatform:
        """Return a cached adapter for the exact platform/account pair."""

        ...


class ExecutionDisposition(StrEnum):
    """Stable CLI-facing outcome categories for one action."""

    PUBLISHED = "published"
    RETRY_SCHEDULED = "retry_scheduled"
    FAILED = "failed"
    SKIPPED = "skipped"
    DRY_RUN = "dry_run"


@dataclass(frozen=True, slots=True)
class EffectiveSafetyLimits:
    """Fully resolved global, platform, and account safety settings."""

    minimum_seconds_between_actions: float
    maximum_actions_per_hour: int
    maximum_actions_per_day: int
    duplicate_window_hours: int
    failure_threshold: int


@dataclass(frozen=True, slots=True)
class ActionExecutionReport:
    """Safe execution result suitable for structured CLI output."""

    action_id: UUID
    platform: Platform
    account_name: str
    disposition: ExecutionDisposition
    status: ActionStatus
    attempt: int
    reason: str
    retry_at: datetime | None = None
    external_content_id: str | None = None
    external_content_url: str | None = None

    @property
    def published(self) -> bool:
        return self.disposition is ExecutionDisposition.PUBLISHED

    @property
    def retry_scheduled(self) -> bool:
        return self.disposition is ExecutionDisposition.RETRY_SCHEDULED


@dataclass(frozen=True, slots=True)
class RunOnceReport:
    """One worker polling-cycle result."""

    worker_id: str
    started_at: datetime
    finished_at: datetime
    due_count: int
    claimed_count: int
    processed_count: int
    published_count: int
    failed_count: int
    retry_scheduled_count: int
    skipped_count: int
    recovered_stale_count: int
    dry_run: bool
    global_paused: bool
    results: tuple[ActionExecutionReport, ...]


@dataclass(frozen=True, slots=True)
class WorkerReport:
    """Aggregate returned when the persistent worker stops normally."""

    worker_id: str
    started_at: datetime
    stopped_at: datetime
    cycles: int
    claimed_count: int
    processed_count: int
    published_count: int
    failed_count: int
    retry_scheduled_count: int
    skipped_count: int
    recovered_stale_count: int
    last_run: RunOnceReport | None


@dataclass(frozen=True, slots=True)
class _SafetyDecision:
    allowed: bool
    reason: str
    retry_at: datetime | None = None
    settle_claim: bool = True
    temporary: bool = False


class _InvalidAdapterResult(RuntimeError):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class AdapterFactory:
    """Construct and safely cache exact account-bound built-in adapters."""

    def __init__(self, browser_manager: BrowserManager, settings: Settings) -> None:
        self._browser_manager = browser_manager
        self._settings = settings
        self._cache: dict[tuple[Platform, str], SocialPlatform] = {}
        self._lock = asyncio.Lock()

    async def get(
        self,
        platform: Platform,
        account_name: str,
    ) -> SocialPlatform:
        key = (Platform(platform), account_name)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        async with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                return cached
            if key[0] is Platform.X:
                adapter: SocialPlatform = XAdapter(
                    self._browser_manager,
                    self._settings,
                    key[1],
                )
            elif key[0] is Platform.REDDIT:
                adapter = RedditAdapter(
                    self._browser_manager,
                    self._settings,
                    key[1],
                )
            else:
                raise ValueError(f"unsupported platform: {key[0]}")
            self._cache[key] = adapter
            return adapter

    def clear(self) -> None:
        self._cache.clear()


class Worker:
    """Poll, atomically claim, safety-check, and execute persistent actions."""

    def __init__(
        self,
        settings: Settings,
        action_repository: ActionRepository | None = None,
        account_state_repository: AccountStateRepository | None = None,
        browser_manager: BrowserManager | None = None,
        *,
        database: Database | None = None,
        adapter_factory: AdapterProvider | None = None,
        polling_interval: float = 5.0,
        batch_size: int = 1,
        lease_duration: timedelta = timedelta(minutes=5),
        retry_base_delay: timedelta = timedelta(seconds=30),
        retry_max_delay: timedelta = timedelta(hours=1),
        retry_jitter_fraction: float = 0.25,
        random_source: JitterSource | None = None,
        clock: Callable[[], datetime] | None = None,
        worker_id: str | None = None,
    ) -> None:
        if polling_interval <= 0:
            raise ValueError("polling_interval must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if lease_duration <= timedelta(0):
            raise ValueError("lease_duration must be positive")
        if retry_base_delay <= timedelta(0):
            raise ValueError("retry_base_delay must be positive")
        if retry_max_delay < retry_base_delay:
            raise ValueError("retry_max_delay must be at least retry_base_delay")
        if not 0 <= retry_jitter_fraction <= 1:
            raise ValueError("retry_jitter_fraction must be between 0 and 1")

        resolved_worker_id = worker_id or (
            f"worker-{os.getpid()}-{uuid4().hex}"
        )
        if not resolved_worker_id.strip():
            raise ValueError("worker_id cannot be empty")

        if database is None and (
            action_repository is None or account_state_repository is None
        ):
            database = Database(settings.database_url)
        if action_repository is None:
            if database is None:
                raise ValueError("database is required when action_repository is omitted")
            action_repository = ActionRepository(database.session_factory)
        if account_state_repository is None:
            if database is None:
                raise ValueError(
                    "database is required when account_state_repository is omitted"
                )
            account_state_repository = AccountStateRepository(database.session_factory)

        resolved_browser_manager = browser_manager or BrowserManager(settings)
        self._settings = settings
        self._actions = action_repository
        self._account_states = account_state_repository
        self._browser_manager = resolved_browser_manager
        self._database = database
        self._adapter_factory = adapter_factory or AdapterFactory(
            resolved_browser_manager,
            settings,
        )
        self._polling_interval = float(polling_interval)
        self._batch_size = batch_size
        self._lease_duration = lease_duration
        self._retry_base_delay = retry_base_delay
        self._retry_max_delay = retry_max_delay
        self._retry_jitter_fraction = retry_jitter_fraction
        self._random = random_source or SystemRandom()
        self._clock = clock or (lambda: datetime.now(UTC))
        self._worker_id = resolved_worker_id.strip()
        self._stop_event = asyncio.Event()
        self._active_tasks: set[asyncio.Task[ActionExecutionReport]] = set()
        self._initialize_lock = asyncio.Lock()
        self._close_lock = asyncio.Lock()
        self._initialized = database is None
        self._closed = False
        self._closing = False
        self._running = False
        self._log = logger.bind(worker_id=self._worker_id)

    @property
    def worker_id(self) -> str:
        return self._worker_id

    @property
    def dry_run(self) -> bool:
        return self._settings.dry_run

    @property
    def global_paused(self) -> bool:
        return self._settings.global_pause

    @property
    def stop_event(self) -> asyncio.Event:
        return self._stop_event

    def stop(self) -> None:
        """Request a graceful stop and wake the polling wait immediately."""

        self._stop_event.set()

    async def run(self) -> WorkerReport:
        """Poll until stopped, closing browser and database on every exit path."""

        self._ensure_open()
        if self._running:
            raise RuntimeError("worker is already running")
        self._running = True
        started_at = self._now()
        cycles = 0
        claimed_count = 0
        processed_count = 0
        published_count = 0
        failed_count = 0
        retry_scheduled_count = 0
        skipped_count = 0
        recovered_stale_count = 0
        last_run: RunOnceReport | None = None

        try:
            await self._ensure_initialized()
            while not self._stop_event.is_set():
                last_run = await self.run_once()
                cycles += 1
                claimed_count += last_run.claimed_count
                processed_count += last_run.processed_count
                published_count += last_run.published_count
                failed_count += last_run.failed_count
                retry_scheduled_count += last_run.retry_scheduled_count
                skipped_count += last_run.skipped_count
                recovered_stale_count += last_run.recovered_stale_count
                if self._stop_event.is_set():
                    break
                try:
                    await asyncio.wait_for(
                        self._stop_event.wait(),
                        timeout=self._polling_interval,
                    )
                except TimeoutError:
                    pass
        finally:
            self._running = False
            await self.close()

        return WorkerReport(
            worker_id=self._worker_id,
            started_at=started_at,
            stopped_at=self._now(),
            cycles=cycles,
            claimed_count=claimed_count,
            processed_count=processed_count,
            published_count=published_count,
            failed_count=failed_count,
            retry_scheduled_count=retry_scheduled_count,
            skipped_count=skipped_count,
            recovered_stale_count=recovered_stale_count,
            last_run=last_run,
        )

    async def run_once(self) -> RunOnceReport:
        """Recover leases and process one atomically claimed batch."""

        self._ensure_open()
        await self._ensure_initialized()
        started_at = self._now()

        if self._settings.dry_run:
            return await self._blocked_run_report(
                started_at,
                reason="dry_run",
                disposition=ExecutionDisposition.DRY_RUN,
                recovered_stale_count=0,
            )

        recovered = await self._actions.recover_stale_claims(now=started_at)
        await self._account_states.clear_expired_pauses(now=started_at)

        if self._settings.global_pause:
            return await self._blocked_run_report(
                started_at,
                reason="global_pause",
                disposition=ExecutionDisposition.SKIPPED,
                recovered_stale_count=len(recovered),
            )

        claimed = await self._actions.claim_due_actions(
            self._worker_id,
            limit=self._batch_size,
            lease_duration=self._lease_duration,
            now=started_at,
        )
        tasks = tuple(
            asyncio.create_task(
                self.process_claimed_action(action),
                name=f"social-bot:{action.id}",
            )
            for action in claimed
        )
        self._active_tasks.update(tasks)
        try:
            results = tuple(await asyncio.gather(*tasks)) if tasks else ()
        finally:
            self._active_tasks.difference_update(tasks)

        report = self._build_run_report(
            started_at=started_at,
            due_count=len(claimed),
            claimed_count=len(claimed),
            recovered_stale_count=len(recovered),
            results=results,
        )
        self._log.info(
            "worker_cycle_completed",
            due_count=report.due_count,
            claimed_count=report.claimed_count,
            published_count=report.published_count,
            failed_count=report.failed_count,
            retry_scheduled_count=report.retry_scheduled_count,
            skipped_count=report.skipped_count,
            recovered_stale_count=report.recovered_stale_count,
            dry_run=False,
            global_paused=False,
        )
        return report

    async def execute_now(self, action_id: UUID | str) -> ActionExecutionReport:
        """Claim and execute one due action without bypassing runtime gates."""

        async with self._close_lock:
            self._ensure_open()
            task = asyncio.create_task(
                self._execute_now(action_id),
                name=f"social-bot-execute-now:{action_id}",
            )
            self._active_tasks.add(task)
        try:
            return await task
        finally:
            self._active_tasks.discard(task)

    async def _execute_now(self, action_id: UUID | str) -> ActionExecutionReport:
        await self._ensure_initialized()
        action = await self._actions.get(action_id)
        if action is None:
            raise ActionNotFoundError(f"action {action_id} was not found")

        if self._settings.dry_run:
            return self._report(
                action,
                ExecutionDisposition.DRY_RUN,
                reason="dry_run",
            )
        if self._settings.global_pause:
            return self._report(
                action,
                ExecutionDisposition.SKIPPED,
                reason="global_pause",
            )
        if action.status is ActionStatus.PUBLISHED:
            return self._report(
                action,
                ExecutionDisposition.SKIPPED,
                reason="already_published",
                external_content_id=action.external_content_id,
                external_content_url=action.external_content_url,
            )
        if action.status not in {ActionStatus.SCHEDULED, ActionStatus.FAILED}:
            return self._report(
                action,
                ExecutionDisposition.SKIPPED,
                reason="not_claimable",
            )
        now = self._now()
        if (
            action.status is ActionStatus.SCHEDULED
            and (action.scheduled_at is None or action.scheduled_at > now)
        ):
            return self._report(
                action,
                ExecutionDisposition.SKIPPED,
                reason="not_due",
            )

        claimed = await self._actions.claim_action(
            action.id,
            self._worker_id,
            lease_duration=self._lease_duration,
            now=now,
        )
        if claimed is None:
            current = await self._actions.get(action.id)
            if current is not None and current.status is ActionStatus.PUBLISHED:
                return self._report(
                    current,
                    ExecutionDisposition.SKIPPED,
                    reason="already_published",
                    external_content_id=current.external_content_id,
                    external_content_url=current.external_content_url,
                )
            return self._report(
                current or action,
                ExecutionDisposition.SKIPPED,
                reason="not_due_or_claimed_elsewhere",
            )
        return await self.process_claimed_action(claimed)

    async def process_claimed_action(
        self,
        action: SocialAction,
    ) -> ActionExecutionReport:
        """Execute only an owned processing claim under an account-wide lease."""

        if action.status is not ActionStatus.PROCESSING:
            return self._report(
                action,
                ExecutionDisposition.SKIPPED,
                reason="action_not_claimed",
            )
        if self._settings.dry_run:
            return self._report(
                action,
                ExecutionDisposition.DRY_RUN,
                reason="dry_run",
            )

        execution_owner = f"{self._worker_id}:{action.id.hex}"
        acquired_at = self._now()
        try:
            account_acquired = await self._account_states.acquire_execution_lease(
                action.platform,
                action.account_name,
                execution_owner,
                lease_duration=self._lease_duration,
                now=acquired_at,
            )
        except Exception as error:
            self._log.error(
                "account_execution_lease_acquisition_failed",
                action_id=str(action.id),
                platform=action.platform.value,
                account_name=action.account_name,
                error_type=type(error).__name__,
            )
            retry_at = self._retry_at(
                action,
                now=acquired_at,
                allow_exhausted=True,
            ) or self._add_delay(acquired_at, self._retry_base_delay)
            return await self._defer_claim(
                action,
                reason="account_execution_lease_acquisition_failed",
                retry_at=retry_at,
            )

        if not account_acquired:
            retry_at = self._add_delay(
                acquired_at,
                min(
                    self._lease_duration,
                    timedelta(seconds=self._polling_interval),
                ),
            )
            return await self._defer_claim(
                action,
                reason="account_execution_busy",
                retry_at=retry_at,
            )

        finished = asyncio.Event()
        heartbeat = asyncio.create_task(
            self._renew_ownership_loop(
                action,
                execution_owner,
                finished,
            ),
            name=f"social-bot-lease:{action.id}",
        )
        execution = asyncio.create_task(
            self._execute_claimed_action(action, execution_owner),
            name=f"social-bot-dispatch:{action.id}",
        )
        try:
            done, _ = await asyncio.wait(
                {execution, heartbeat},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if execution in done:
                return await execution

            loss_reason = await heartbeat
            execution.cancel()
            with suppress(asyncio.CancelledError):
                await execution
            current = await self._actions.get(action.id)
            return self._report(
                current or action,
                ExecutionDisposition.SKIPPED,
                reason=loss_reason or "ownership_heartbeat_stopped",
                external_content_id=(
                    current.external_content_id if current is not None else None
                ),
                external_content_url=(
                    current.external_content_url if current is not None else None
                ),
            )
        except asyncio.CancelledError:
            execution.cancel()
            with suppress(asyncio.CancelledError):
                await execution
            raise
        finally:
            finished.set()
            heartbeat.cancel()
            with suppress(asyncio.CancelledError):
                await heartbeat
            try:
                await self._account_states.release_execution_lease(
                    action.platform,
                    action.account_name,
                    execution_owner,
                    now=self._now(),
                )
            except Exception as error:
                self._log.error(
                    "account_execution_lease_release_failed",
                    action_id=str(action.id),
                    platform=action.platform.value,
                    account_name=action.account_name,
                    error_type=type(error).__name__,
                )

    async def _execute_claimed_action(
        self,
        action: SocialAction,
        execution_owner: str,
    ) -> ActionExecutionReport:
        external_dispatch_started = False
        dispatch_boundary: asyncio.Task[SocialAction] | None = None
        try:
            now = self._now()
            decision = await self._evaluate_safety(
                action,
                now=now,
                require_processing=True,
            )
            if not decision.allowed:
                if not decision.settle_claim:
                    current = await self._actions.get(action.id)
                    return self._report(
                        current or action,
                        ExecutionDisposition.SKIPPED,
                        reason=decision.reason,
                        retry_at=decision.retry_at,
                    )
                return await self._settle_safety_decision(action, decision, now=now)

            adapter = await self._adapter_factory.get(
                action.platform,
                action.account_name,
            )
            ownership_at = self._now()
            account_owned = await self._account_states.renew_execution_lease(
                action.platform,
                action.account_name,
                execution_owner,
                lease_duration=self._lease_duration,
                now=ownership_at,
            )
            if not account_owned:
                return await self._defer_claim(
                    action,
                    reason="account_execution_lease_lost_before_dispatch",
                    retry_at=self._add_delay(
                        ownership_at,
                        timedelta(seconds=self._polling_interval),
                    ),
                )
            if not await self._actions.verify_claim(
                action.id,
                self._worker_id,
                now=ownership_at,
            ):
                current = await self._actions.get(action.id)
                return self._report(
                    current or action,
                    ExecutionDisposition.SKIPPED,
                    reason="claim_lost_before_dispatch",
                    external_content_id=(
                        current.external_content_id if current is not None else None
                    ),
                    external_content_url=(
                        current.external_content_url if current is not None else None
                    ),
                )

            dispatch_boundary = asyncio.create_task(
                self._actions.begin_external_dispatch(
                    action.id,
                    self._worker_id,
                    now=self._now(),
                ),
                name=f"social-bot-dispatch-boundary:{action.id}",
            )
            dispatched = await asyncio.shield(dispatch_boundary)
            external_dispatch_started = True
            result = await adapter.execute(dispatched)
            self._validate_result(dispatched, result)

            current = await self._actions.get(action.id)
            if current is None or current.status is not ActionStatus.PROCESSING:
                return self._report(
                    current or action,
                    ExecutionDisposition.SKIPPED,
                    reason=(
                        "already_published"
                        if current is not None
                        and current.status is ActionStatus.PUBLISHED
                        else "claim_lost_after_execution"
                    ),
                    external_content_id=(
                        current.external_content_id if current is not None else None
                    ),
                    external_content_url=(
                        current.external_content_url if current is not None else None
                    ),
                )

            try:
                published = await self._actions.mark_published(
                    action.id,
                    self._worker_id,
                    external_content_id=result.external_content_id,
                    external_content_url=result.external_content_url,
                    published_at=self._now(),
                )
            except ClaimConflictError:
                latest = await self._actions.get(action.id)
                return self._report(
                    latest or action,
                    ExecutionDisposition.SKIPPED,
                    reason=(
                        "already_published"
                        if latest is not None
                        and latest.status is ActionStatus.PUBLISHED
                        else "publication_record_conflict"
                    ),
                    external_content_id=(
                        latest.external_content_id if latest is not None else None
                    ),
                    external_content_url=(
                        latest.external_content_url if latest is not None else None
                    ),
                )

            try:
                await self._account_states.record_success(
                    action.platform,
                    action.account_name,
                    now=self._now(),
                )
            except Exception as error:
                self._log.error(
                    "account_success_state_failed",
                    action_id=str(action.id),
                    platform=action.platform.value,
                    account_name=action.account_name,
                    error_type=type(error).__name__,
                )

            report = self._report(
                published,
                ExecutionDisposition.PUBLISHED,
                reason="published",
                external_content_id=published.external_content_id,
                external_content_url=published.external_content_url,
            )
            self._log.info(
                "action_published",
                action_id=str(action.id),
                platform=action.platform.value,
                account_name=action.account_name,
                attempt=action.attempts,
            )
            return report
        except asyncio.CancelledError as cancellation:
            if dispatch_boundary is not None:
                while not dispatch_boundary.done():
                    try:
                        await asyncio.shield(dispatch_boundary)
                    except asyncio.CancelledError:
                        continue
                if not dispatch_boundary.cancelled():
                    try:
                        dispatch_boundary.result()
                    except Exception:
                        pass
                    else:
                        external_dispatch_started = True
            await self._settle_cancellation(
                action,
                execution_started=external_dispatch_started,
            )
            raise cancellation
        except ClaimConflictError:
            current = await self._actions.get(action.id)
            return self._report(
                current or action,
                ExecutionDisposition.SKIPPED,
                reason="claim_lost_before_dispatch",
                external_content_id=(
                    current.external_content_id if current is not None else None
                ),
                external_content_url=(
                    current.external_content_url if current is not None else None
                ),
            )
        except _InvalidAdapterResult as error:
            return await self._mark_claim_failed(
                action,
                error=f"invalid adapter result: {error.reason}",
                reason=error.reason,
                retry_at=None,
                page_url=None,
                screenshot_path=None,
                context={
                    "error_type": type(error).__name__,
                    "attempt": action.attempts,
                },
                failed_at=self._now(),
            )
        except PlatformError as error:
            return await self._handle_platform_error(action, error)
        except Exception as error:
            failed_at = self._now()
            retry_at = (
                None
                if external_dispatch_started
                else self._retry_at(action, now=failed_at)
            )
            self._log.error(
                (
                    "action_execution_uncertain"
                    if external_dispatch_started
                    else "action_pre_dispatch_failed"
                ),
                action_id=str(action.id),
                platform=action.platform.value,
                account_name=action.account_name,
                attempt=action.attempts,
                error_type=type(error).__name__,
            )
            return await self._mark_claim_failed(
                action,
                error=f"unexpected execution failure: {type(error).__name__}",
                reason=(
                    "uncertain_execution_failure"
                    if external_dispatch_started
                    else "unexpected_pre_dispatch_failure"
                ),
                retry_at=retry_at,
                page_url=None,
                screenshot_path=None,
                context={
                    "error_type": type(error).__name__,
                    "attempt": action.attempts,
                    "uncertain": external_dispatch_started,
                },
                failed_at=failed_at,
            )

    async def close(self) -> None:
        """Stop work and close BrowserManager and the owned/passed Database."""

        async with self._close_lock:
            if self._closed:
                return
            self._closing = True
            self._stop_event.set()
            current_task = asyncio.current_task()
            active = tuple(
                task
                for task in self._active_tasks
                if task is not current_task and not task.done()
            )
            if active:
                await asyncio.gather(*active, return_exceptions=True)
            try:
                await self._browser_manager.shutdown()
            finally:
                try:
                    if self._database is not None:
                        await self._database.close()
                finally:
                    if isinstance(self._adapter_factory, AdapterFactory):
                        self._adapter_factory.clear()
                    self._closed = True

    async def __aenter__(self) -> Worker:
        self._ensure_open()
        await self._ensure_initialized()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()

    async def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        async with self._initialize_lock:
            if self._initialized:
                return
            if self._database is not None:
                await self._database.initialize()
            self._initialized = True

    def _ensure_open(self) -> None:
        if self._closed or self._closing:
            raise RuntimeError("worker is closing or closed")

    async def _blocked_run_report(
        self,
        started_at: datetime,
        *,
        reason: str,
        disposition: ExecutionDisposition,
        recovered_stale_count: int,
    ) -> RunOnceReport:
        due_count, due_actions = await self._preview_due_actions(started_at)
        results = tuple(
            self._report(action, disposition, reason=reason)
            for action in due_actions
        )
        report = RunOnceReport(
            worker_id=self._worker_id,
            started_at=started_at,
            finished_at=self._now(),
            due_count=due_count,
            claimed_count=0,
            processed_count=0,
            published_count=0,
            failed_count=0,
            retry_scheduled_count=0,
            skipped_count=due_count,
            recovered_stale_count=recovered_stale_count,
            dry_run=self._settings.dry_run,
            global_paused=self._settings.global_pause,
            results=results,
        )
        self._log.info(
            "worker_cycle_skipped",
            due_count=due_count,
            claimed_count=0,
            skipped_count=due_count,
            recovered_stale_count=recovered_stale_count,
            dry_run=self._settings.dry_run,
            global_paused=self._settings.global_pause,
            reason=reason,
        )
        return report

    async def _preview_due_actions(
        self,
        now: datetime,
    ) -> tuple[int, tuple[SocialAction, ...]]:
        due = await self._actions.list_due_actions(
            limit=self._batch_size,
            now=now,
        )
        return len(due), tuple(due)

    def _build_run_report(
        self,
        *,
        started_at: datetime,
        due_count: int,
        claimed_count: int,
        recovered_stale_count: int,
        results: tuple[ActionExecutionReport, ...],
    ) -> RunOnceReport:
        return RunOnceReport(
            worker_id=self._worker_id,
            started_at=started_at,
            finished_at=self._now(),
            due_count=due_count,
            claimed_count=claimed_count,
            processed_count=len(results),
            published_count=sum(result.published for result in results),
            failed_count=sum(
                result.disposition is ExecutionDisposition.FAILED
                for result in results
            ),
            retry_scheduled_count=sum(
                result.retry_scheduled for result in results
            ),
            skipped_count=sum(
                result.disposition
                in {ExecutionDisposition.SKIPPED, ExecutionDisposition.DRY_RUN}
                for result in results
            ),
            recovered_stale_count=recovered_stale_count,
            dry_run=self._settings.dry_run,
            global_paused=self._settings.global_pause,
            results=results,
        )

    async def _evaluate_safety(
        self,
        action: SocialAction,
        *,
        now: datetime,
        require_processing: bool,
    ) -> _SafetyDecision:
        current = await self._actions.get(action.id)
        if current is None:
            return _SafetyDecision(
                allowed=False,
                reason="action_missing",
                settle_claim=False,
            )
        if current.status is ActionStatus.PUBLISHED:
            return _SafetyDecision(
                allowed=False,
                reason="already_published",
                settle_claim=False,
            )
        if require_processing and current.status is not ActionStatus.PROCESSING:
            return _SafetyDecision(
                allowed=False,
                reason="claim_lost",
                settle_claim=False,
            )
        if self._settings.global_pause:
            return _SafetyDecision(
                allowed=False,
                reason="global_pause",
                retry_at=self._add_delay(
                    now,
                    timedelta(seconds=self._polling_interval),
                ),
                temporary=True,
            )

        state = await self._account_states.get(
            action.platform,
            action.account_name,
        )
        if state is not None and state.is_paused(now):
            retry_at = state.paused_until or self._add_delay(
                now,
                timedelta(seconds=self._polling_interval),
            )
            return _SafetyDecision(
                allowed=False,
                reason="account_paused",
                retry_at=retry_at,
                temporary=True,
            )

        limits = resolve_effective_limits(
            self._settings,
            action.platform,
            action.account_name,
        )
        if limits.duplicate_window_hours > 0:
            duplicate = await self._actions.has_published_duplicate(
                action,
                window=timedelta(hours=limits.duplicate_window_hours),
                now=now,
            )
            if duplicate:
                return _SafetyDecision(
                    allowed=False,
                    reason="duplicate_publication",
                )

        usage = await self._actions.get_rate_usage(
            action.platform,
            action.account_name,
            now=now,
        )
        gates: list[tuple[str, datetime]] = []
        if usage.last_published_at is not None:
            earliest = self._add_delay(
                usage.last_published_at,
                timedelta(seconds=limits.minimum_seconds_between_actions),
            )
            if earliest > now:
                gates.append(("minimum_interval", earliest))
        if usage.hourly >= limits.maximum_actions_per_hour:
            gates.append(("hourly_limit", self._add_delay(now, timedelta(hours=1))))
        if usage.daily >= limits.maximum_actions_per_day:
            gates.append(("daily_limit", self._add_delay(now, timedelta(days=1))))
        if gates:
            reason, retry_at = max(gates, key=lambda gate: gate[1])
            return _SafetyDecision(
                allowed=False,
                reason=reason,
                retry_at=retry_at,
                temporary=True,
            )
        return _SafetyDecision(allowed=True, reason="allowed")

    async def _settle_safety_decision(
        self,
        action: SocialAction,
        decision: _SafetyDecision,
        *,
        now: datetime,
    ) -> ActionExecutionReport:
        if decision.temporary:
            retry_at = (
                decision.retry_at
                if decision.retry_at is not None and decision.retry_at > now
                else self._add_delay(
                    now,
                    timedelta(seconds=self._polling_interval),
                )
            )
            return await self._defer_claim(
                action,
                reason=decision.reason,
                retry_at=retry_at,
            )

        return await self._mark_claim_failed(
            action,
            error=f"execution blocked by safety gate: {decision.reason}",
            reason=decision.reason,
            retry_at=None,
            page_url=None,
            screenshot_path=None,
            context={
                "safety_gate": decision.reason,
                "attempt": action.attempts,
                "permanent": True,
            },
            failed_at=now,
        )

    async def _handle_platform_error(
        self,
        action: SocialAction,
        error: PlatformError,
    ) -> ActionExecutionReport:
        now = self._now()
        retry_at = self._platform_retry_at(action, error, now=now)
        page_url, screenshot_path = _safe_diagnostics(error)
        safe_error = _safe_platform_error_message(error)
        report = await self._mark_claim_failed(
            action,
            error=safe_error,
            reason=(
                "platform_retryable"
                if retry_at is not None
                else "platform_permanent"
            ),
            retry_at=retry_at,
            page_url=page_url,
            screenshot_path=screenshot_path,
            context={
                "error_type": type(error).__name__,
                "retryable": bool(error.retryable),
                "attempt": action.attempts,
                "retry_at": retry_at.isoformat() if retry_at is not None else None,
            },
            failed_at=now,
        )

        if isinstance(error, PlatformAuthenticationError):
            limits = resolve_effective_limits(
                self._settings,
                action.platform,
                action.account_name,
            )
            try:
                state = await self._account_states.record_failure(
                    action.platform,
                    action.account_name,
                    safe_error,
                    failure_threshold=limits.failure_threshold,
                    now=now,
                )
                self._log.warning(
                    "account_authentication_failure",
                    action_id=str(action.id),
                    platform=action.platform.value,
                    account_name=action.account_name,
                    consecutive_failures=state.consecutive_failures,
                    paused=state.is_paused(now),
                    error_type=type(error).__name__,
                )
            except Exception as state_error:
                self._log.error(
                    "account_failure_state_failed",
                    action_id=str(action.id),
                    platform=action.platform.value,
                    account_name=action.account_name,
                    error_type=type(state_error).__name__,
                )

        self._log.warning(
            "action_platform_failure",
            action_id=str(action.id),
            platform=action.platform.value,
            account_name=action.account_name,
            attempt=action.attempts,
            error_type=type(error).__name__,
            retry_scheduled=retry_at is not None,
            page_url_recorded=page_url is not None,
            screenshot_recorded=screenshot_path is not None,
        )
        return report

    def _platform_retry_at(
        self,
        action: SocialAction,
        error: PlatformError,
        *,
        now: datetime,
    ) -> datetime | None:
        if not error.retryable:
            return None

        retry_at = self._retry_at(action, now=now)
        if retry_at is None or not isinstance(error, PlatformRateLimitError):
            return retry_at

        retry_after = error.retry_after_seconds
        if retry_after is None:
            return retry_at
        if not math.isfinite(retry_after) or retry_after < 0:
            return None
        try:
            retry_delay = timedelta(seconds=retry_after)
        except OverflowError:
            return None
        latest = datetime.max.replace(tzinfo=UTC)
        if retry_delay > latest - now:
            return None
        return max(retry_at, now + retry_delay)

    def _retry_at(
        self,
        action: SocialAction,
        *,
        now: datetime,
        allow_exhausted: bool = False,
    ) -> datetime | None:
        if action.attempts >= action.max_attempts and not allow_exhausted:
            return None

        exponent = min(max(action.attempts - 1, 0), 30)
        base_seconds = self._retry_base_delay.total_seconds()
        maximum_seconds = self._retry_max_delay.total_seconds()
        exponential_seconds = min(base_seconds * (2**exponent), maximum_seconds)
        jitter_capacity = max(
            min(
                exponential_seconds * self._retry_jitter_fraction,
                maximum_seconds - exponential_seconds,
            ),
            0.0,
        )
        jitter_seconds = (
            min(
                max(float(self._random.uniform(0.0, jitter_capacity)), 0.0),
                jitter_capacity,
            )
            if jitter_capacity > 0
            else 0.0
        )
        delay_seconds = min(
            exponential_seconds + jitter_seconds,
            maximum_seconds,
        )
        try:
            delay = timedelta(seconds=delay_seconds)
        except OverflowError:
            delay = self._retry_max_delay
        return self._add_delay(now, min(delay, self._retry_max_delay))


    @staticmethod
    def _add_delay(now: datetime, delay: timedelta) -> datetime:
        latest = datetime.max.replace(tzinfo=UTC)
        remaining = latest - now
        return latest if delay >= remaining else now + delay

    async def _defer_claim(
        self,
        action: SocialAction,
        *,
        reason: str,
        retry_at: datetime,
    ) -> ActionExecutionReport:
        try:
            deferred = await self._actions.defer_claim(
                action.id,
                self._worker_id,
                retry_at,
                reason,
            )
        except ClaimConflictError:
            current = await self._actions.get(action.id)
            return self._report(
                current or action,
                ExecutionDisposition.SKIPPED,
                reason=(
                    "already_published"
                    if current is not None
                    and current.status is ActionStatus.PUBLISHED
                    else "claim_deferral_conflict"
                ),
                external_content_id=(
                    current.external_content_id if current is not None else None
                ),
                external_content_url=(
                    current.external_content_url if current is not None else None
                ),
            )
        except Exception as persistence_error:
            self._log.error(
                "claim_deferral_failed",
                action_id=str(action.id),
                platform=action.platform.value,
                account_name=action.account_name,
                error_type=type(persistence_error).__name__,
            )
            return self._report(
                action,
                ExecutionDisposition.SKIPPED,
                reason="claim_deferral_failed",
            )
        return self._report(
            deferred,
            ExecutionDisposition.RETRY_SCHEDULED,
            reason=reason,
            retry_at=retry_at,
        )

    async def _mark_claim_failed(
        self,
        action: SocialAction,
        *,
        error: str,
        reason: str,
        retry_at: datetime | None,
        page_url: str | None,
        screenshot_path: str | None,
        context: dict[str, object],
        failed_at: datetime,
    ) -> ActionExecutionReport:
        try:
            failed = await self._actions.mark_failed(
                action.id,
                self._worker_id,
                error,
                retry_at=retry_at,
                page_url=page_url,
                screenshot_path=screenshot_path,
                context=context,
                failed_at=failed_at,
            )
        except ClaimConflictError:
            current = await self._actions.get(action.id)
            return self._report(
                current or action,
                ExecutionDisposition.SKIPPED,
                reason=(
                    "already_published"
                    if current is not None
                    and current.status is ActionStatus.PUBLISHED
                    else "claim_settlement_conflict"
                ),
                external_content_id=(
                    current.external_content_id if current is not None else None
                ),
                external_content_url=(
                    current.external_content_url if current is not None else None
                ),
            )
        except Exception as persistence_error:
            self._log.error(
                "claim_settlement_failed",
                action_id=str(action.id),
                platform=action.platform.value,
                account_name=action.account_name,
                error_type=type(persistence_error).__name__,
            )
            return self._report(
                action,
                ExecutionDisposition.SKIPPED,
                reason="claim_settlement_failed",
            )

        has_retry = retry_at is not None and failed.attempts < failed.max_attempts
        return self._report(
            failed,
            (
                ExecutionDisposition.RETRY_SCHEDULED
                if has_retry
                else ExecutionDisposition.FAILED
            ),
            reason=reason,
            retry_at=retry_at if has_retry else None,
        )

    async def _settle_cancellation(
        self,
        action: SocialAction,
        *,
        execution_started: bool,
    ) -> None:
        now = self._now()
        retry_at = (
            self._retry_at(action, now=now)
            if not execution_started
            else None
        )
        settlement = asyncio.create_task(
            self._mark_claim_failed(
                action,
                error=(
                    "worker cancelled before platform execution"
                    if not execution_started
                    else "worker cancelled during platform execution; outcome uncertain"
                ),
                reason=(
                    "cancelled_before_execution"
                    if not execution_started
                    else "cancelled_during_execution"
                ),
                retry_at=retry_at,
                page_url=None,
                screenshot_path=None,
                context={
                    "cancelled": True,
                    "execution_started": execution_started,
                    "attempt": action.attempts,
                },
                failed_at=now,
            ),
            name=f"social-bot-settle:{action.id}",
        )
        try:
            await asyncio.shield(settlement)
        except asyncio.CancelledError:
            with suppress(asyncio.CancelledError):
                await settlement

    async def _renew_ownership_loop(
        self,
        action: SocialAction,
        execution_owner: str,
        finished: asyncio.Event,
    ) -> str | None:
        interval = max(min(self._lease_duration.total_seconds() / 3, 30.0), 0.05)
        while not finished.is_set():
            try:
                await asyncio.wait_for(finished.wait(), timeout=interval)
                return None
            except TimeoutError:
                pass

            now = self._now()
            try:
                claim_renewed = await self._actions.renew_claim(
                    action.id,
                    self._worker_id,
                    lease_duration=self._lease_duration,
                    now=now,
                )
            except Exception as error:
                self._log.error(
                    "claim_renewal_failed",
                    action_id=str(action.id),
                    error_type=type(error).__name__,
                )
                return "claim_renewal_failed"
            if not claim_renewed:
                self._log.warning(
                    "claim_renewal_lost",
                    action_id=str(action.id),
                )
                return "claim_lease_lost"

            try:
                account_renewed = (
                    await self._account_states.renew_execution_lease(
                        action.platform,
                        action.account_name,
                        execution_owner,
                        lease_duration=self._lease_duration,
                        now=now,
                    )
                )
            except Exception as error:
                self._log.error(
                    "account_execution_lease_renewal_failed",
                    action_id=str(action.id),
                    platform=action.platform.value,
                    account_name=action.account_name,
                    error_type=type(error).__name__,
                )
                return "account_execution_lease_renewal_failed"
            if not account_renewed:
                self._log.warning(
                    "account_execution_lease_renewal_lost",
                    action_id=str(action.id),
                    platform=action.platform.value,
                    account_name=action.account_name,
                )
                return "account_execution_lease_lost"

    @staticmethod
    def _validate_result(action: SocialAction, result: ActionResult) -> None:
        if result.action_id != action.id:
            raise _InvalidAdapterResult("action_id_mismatch")
        if not result.success:
            raise _InvalidAdapterResult("unsuccessful_result")
        if not result.external_content_id:
            raise _InvalidAdapterResult("missing_external_content_id")
        if not result.external_content_url:
            raise _InvalidAdapterResult("missing_external_content_url")
        if not _is_safe_external_url(result.external_content_url):
            raise _InvalidAdapterResult("invalid_external_content_url")

    @staticmethod
    def _report(
        action: SocialAction,
        disposition: ExecutionDisposition,
        *,
        reason: str,
        retry_at: datetime | None = None,
        external_content_id: str | None = None,
        external_content_url: str | None = None,
    ) -> ActionExecutionReport:
        return ActionExecutionReport(
            action_id=action.id,
            platform=action.platform,
            account_name=action.account_name,
            disposition=disposition,
            status=action.status,
            attempt=action.attempts,
            reason=reason,
            retry_at=retry_at,
            external_content_id=external_content_id,
            external_content_url=external_content_url,
        )

    def _now(self) -> datetime:
        return _as_utc(self._clock(), name="clock")


def resolve_effective_limits(
    settings: Settings,
    platform: Platform,
    account_name: str,
) -> EffectiveSafetyLimits:
    """Layer global defaults, platform overrides, then account overrides."""

    resolved_platform = Platform(platform)
    platform_overrides = (
        settings.limits.platforms.x
        if resolved_platform is Platform.X
        else settings.limits.platforms.reddit
    )
    account = (
        settings.accounts.x.get(account_name)
        if resolved_platform is Platform.X
        else settings.accounts.reddit.get(account_name)
    )
    account_overrides = account.limits if account is not None else None

    return EffectiveSafetyLimits(
        minimum_seconds_between_actions=float(
            _effective_value(
                settings.limits.minimum_seconds_between_actions,
                platform_overrides,
                account_overrides,
                "minimum_seconds_between_actions",
            )
        ),
        maximum_actions_per_hour=int(
            _effective_value(
                settings.limits.maximum_actions_per_hour,
                platform_overrides,
                account_overrides,
                "maximum_actions_per_hour",
            )
        ),
        maximum_actions_per_day=int(
            _effective_value(
                settings.limits.maximum_actions_per_day,
                platform_overrides,
                account_overrides,
                "maximum_actions_per_day",
            )
        ),
        duplicate_window_hours=int(
            _effective_value(
                settings.limits.duplicate_window_hours,
                platform_overrides,
                account_overrides,
                "duplicate_window_hours",
            )
        ),
        failure_threshold=int(
            _effective_value(
                settings.limits.failure_threshold,
                platform_overrides,
                account_overrides,
                "failure_threshold",
            )
        ),
    )


def _effective_value(
    default: int | float,
    platform_overrides: SafetyLimitOverrides | None,
    account_overrides: SafetyLimitOverrides | None,
    field_name: str,
) -> int | float:
    account_value = (
        getattr(account_overrides, field_name)
        if account_overrides is not None
        else None
    )
    if account_value is not None:
        return account_value
    platform_value = (
        getattr(platform_overrides, field_name)
        if platform_overrides is not None
        else None
    )
    return platform_value if platform_value is not None else default


def _as_utc(value: datetime, *, name: str) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"{name} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must include timezone information")
    return value.astimezone(UTC)


def _safe_platform_error_message(error: PlatformError) -> str:
    message = " ".join(str(error).split())
    if not message:
        return type(error).__name__
    return f"{type(error).__name__}: {message[:900]}"


def _safe_diagnostics(error: PlatformError) -> tuple[str | None, str | None]:
    if not isinstance(error, BrowserInteractionError):
        return None, None
    page_url = _safe_page_url(error.current_url)
    screenshot_path = (
        str(error.screenshot_path)
        if isinstance(error.screenshot_path, Path)
        else None
    )
    return page_url, screenshot_path


def _safe_page_url(value: str | None) -> str | None:
    if not value:
        return None
    try:
        parsed = urlsplit(value)
        if parsed.scheme.casefold() not in {"http", "https"} or not parsed.hostname:
            return None
        hostname = parsed.hostname.casefold()
        authority = f"[{hostname}]" if ":" in hostname else hostname
        if parsed.port is not None:
            is_default = (
                parsed.scheme.casefold() == "http" and parsed.port == 80
            ) or (
                parsed.scheme.casefold() == "https" and parsed.port == 443
            )
            if not is_default:
                authority += f":{parsed.port}"
        return urlunsplit(
            (parsed.scheme.casefold(), authority, parsed.path or "/", "", "")
        )
    except ValueError:
        return None


def _is_safe_external_url(value: str) -> bool:
    try:
        parsed = urlsplit(value)
        port = parsed.port
        hostname = parsed.hostname
    except ValueError:
        return False
    return (
        parsed.scheme.casefold() in {"http", "https"}
        and bool(hostname)
        and parsed.username is None
        and parsed.password is None
        and (port is None or 1 <= port <= 65535)
    )


__all__ = [
    "ActionExecutionReport",
    "AdapterFactory",
    "AdapterProvider",
    "EffectiveSafetyLimits",
    "ExecutionDisposition",
    "JitterSource",
    "RunOnceReport",
    "Worker",
    "WorkerReport",
    "resolve_effective_limits",
]
