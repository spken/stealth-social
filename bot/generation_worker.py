"""Leased scheduled-generation worker kept separate from publishing work."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from random import SystemRandom
from typing import Protocol
from uuid import UUID, uuid4

import structlog

from bot.config import Settings
from bot.content.models import GenerationStatus, SanitizedFailure, StoredGenerationRequest
from bot.content.service import GenerationLeaseLostError, GenerationService
from bot.examples.models import ExampleRateLimitedError, ExampleTargetUnavailableError
from bot.ollama.errors import OllamaTimeoutError, OllamaUnavailableError
from bot.storage.content_repository import ContentRepository

logger = structlog.get_logger(__name__)


class GenerationJitterSource(Protocol):
    """Small injectable retry-jitter surface."""

    def uniform(self, a: float, b: float) -> float:
        ...


@dataclass(frozen=True, slots=True)
class GenerationExecutionReport:
    request_id: UUID
    status: str
    attempt: int
    error_type: str | None = None
    retry_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class GenerationRunOnceReport:
    worker_id: str
    started_at: datetime
    finished_at: datetime
    enabled: bool
    paused: bool
    due_count: int
    claimed_count: int
    completed_count: int
    failed_count: int
    retry_scheduled_count: int
    results: tuple[GenerationExecutionReport, ...]


@dataclass(frozen=True, slots=True)
class GenerationWorkerReport:
    worker_id: str
    started_at: datetime
    stopped_at: datetime
    cycles: int
    claimed_count: int
    completed_count: int
    failed_count: int
    retry_scheduled_count: int
    last_run: GenerationRunOnceReport | None


class GenerationWorker:
    """Claim, execute, renew, and settle scheduled generation requests."""

    def __init__(
        self,
        settings: Settings,
        content_repository: ContentRepository,
        generation_service: GenerationService,
        *,
        clock: Callable[[], datetime] | None = None,
        polling_interval: float = 5.0,
        batch_size: int = 1,
        lease_duration: timedelta = timedelta(minutes=5),
        retry_base_delay: timedelta = timedelta(seconds=30),
        retry_max_delay: timedelta = timedelta(hours=1),
        retry_jitter_fraction: float = 0.25,
        random_source: GenerationJitterSource | None = None,
        worker_id: str | None = None,
    ) -> None:
        if polling_interval <= 0 or batch_size <= 0:
            raise ValueError("polling_interval and batch_size must be positive")
        if lease_duration <= timedelta(0):
            raise ValueError("lease_duration must be positive")
        if retry_base_delay <= timedelta(0) or retry_max_delay < retry_base_delay:
            raise ValueError("retry delays must be positive and ordered")
        if not 0 <= retry_jitter_fraction <= 1:
            raise ValueError("retry_jitter_fraction must be between 0 and 1")

        identifier = worker_id or f"generation-{os.getpid()}-{uuid4().hex}"
        if not identifier.strip():
            raise ValueError("worker_id cannot be empty")
        self._settings = settings
        self._content = content_repository
        self._generation = generation_service
        self._clock = clock or (lambda: datetime.now(UTC))
        self._polling_interval = float(polling_interval)
        self._batch_size = batch_size
        self._lease_duration = lease_duration
        self._retry_base_delay = retry_base_delay
        self._retry_max_delay = retry_max_delay
        self._retry_jitter_fraction = retry_jitter_fraction
        self._random = random_source or SystemRandom()
        self._worker_id = identifier.strip()
        self._stop_event = asyncio.Event()
        self._active_tasks: set[asyncio.Task[GenerationExecutionReport]] = set()
        self._close_lock = asyncio.Lock()
        self._closed = False
        self._running = False
        self._log = logger.bind(worker_id=self._worker_id, queue="generation")

    @property
    def worker_id(self) -> str:
        return self._worker_id

    @property
    def stop_event(self) -> asyncio.Event:
        return self._stop_event

    def stop(self) -> None:
        self._stop_event.set()

    async def run_once(self) -> GenerationRunOnceReport:
        self._ensure_open()
        started_at = self._now()
        if not self._settings.automation.allow_scheduled_generation:
            return self._report(
                started_at,
                enabled=False,
                paused=False,
                requests=(),
            )
        if self._settings.global_pause:
            return self._report(
                started_at,
                enabled=True,
                paused=True,
                requests=(),
            )

        claimed = await self._content.claim_due_generation_requests(
            self._worker_id,
            limit=self._batch_size,
            lease_duration=self._lease_duration,
            now=started_at,
        )
        tasks = tuple(
            asyncio.create_task(
                self._process_claimed(request),
                name=f"social-bot-generation:{request.id}",
            )
            for request in claimed
        )
        self._active_tasks.update(tasks)
        try:
            results = tuple(await asyncio.gather(*tasks)) if tasks else ()
        except asyncio.CancelledError:
            for task in tasks:
                if not task.done():
                    task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            raise
        finally:
            self._active_tasks.difference_update(tasks)
        report = self._report(
            started_at,
            enabled=True,
            paused=False,
            requests=results,
        )
        self._log.info(
            "generation_worker_cycle_completed",
            due_count=report.due_count,
            claimed_count=report.claimed_count,
            completed_count=report.completed_count,
            failed_count=report.failed_count,
            retry_scheduled_count=report.retry_scheduled_count,
        )
        return report

    async def run(self) -> GenerationWorkerReport:
        self._ensure_open()
        if self._running:
            raise RuntimeError("generation worker is already running")
        self._running = True
        started_at = self._now()
        cycles = claimed = completed = failed = retries = 0
        last_run: GenerationRunOnceReport | None = None
        try:
            while not self._stop_event.is_set():
                last_run = await self.run_once()
                cycles += 1
                claimed += last_run.claimed_count
                completed += last_run.completed_count
                failed += last_run.failed_count
                retries += last_run.retry_scheduled_count
                if self._stop_event.is_set():
                    break
                try:
                    await asyncio.wait_for(
                        self._stop_event.wait(), timeout=self._polling_interval
                    )
                except TimeoutError:
                    continue
        finally:
            self._running = False
            await self.close()
        return GenerationWorkerReport(
            worker_id=self._worker_id,
            started_at=started_at,
            stopped_at=self._now(),
            cycles=cycles,
            claimed_count=claimed,
            completed_count=completed,
            failed_count=failed,
            retry_scheduled_count=retries,
            last_run=last_run,
        )

    async def close(self) -> None:
        async with self._close_lock:
            if self._closed:
                return
            self._stop_event.set()
            current = asyncio.current_task()
            active = tuple(
                task
                for task in self._active_tasks
                if task is not current and not task.done()
            )
            for task in active:
                task.cancel()
            try:
                if active:
                    await asyncio.gather(*active, return_exceptions=True)
            finally:
                self._closed = True

    async def _process_claimed(
        self, request: StoredGenerationRequest
    ) -> GenerationExecutionReport:
        lease_lost = asyncio.Event()
        renewer = asyncio.create_task(
            self._renew_claim(request.id, lease_lost),
            name=f"social-bot-generation-renew:{request.id}",
        )
        execution = asyncio.create_task(
            self._generation.execute_request(
                request.id,
                owner=self._worker_id,
                force_prepare=True,
                renew_lease=False,
            ),
            name=f"social-bot-generation-execute:{request.id}",
        )
        try:
            done, _ = await asyncio.wait(
                (execution, renewer),
                return_when=asyncio.FIRST_COMPLETED,
            )
            if execution in done:
                await execution
                return GenerationExecutionReport(
                    request_id=request.id,
                    status=GenerationStatus.COMPLETED.value,
                    attempt=request.attempt_count,
                )
            if renewer in done and lease_lost.is_set():
                if not execution.done():
                    execution.cancel()
                try:
                    await execution
                except asyncio.CancelledError as error:
                    raise GenerationLeaseLostError(
                        f"generation request {request.id} lease was lost"
                    ) from error
                return GenerationExecutionReport(
                    request_id=request.id,
                    status=GenerationStatus.COMPLETED.value,
                    attempt=request.attempt_count,
                )
            await execution
            return GenerationExecutionReport(
                request_id=request.id,
                status=GenerationStatus.COMPLETED.value,
                attempt=request.attempt_count,
            )
        except asyncio.CancelledError:
            try:
                await self._content.release_generation_claim(
                    request.id, self._worker_id, now=self._now()
                )
            except Exception as error:
                self._log.warning(
                    "generation_claim_release_failed",
                    request_id=str(request.id),
                    error_type=type(error).__name__,
                )
            raise
        except GenerationLeaseLostError:
            try:
                await self._content.release_generation_claim(
                    request.id, self._worker_id, now=self._now()
                )
            except Exception as error:
                self._log.warning(
                    "generation_claim_release_failed",
                    request_id=str(request.id),
                    error_type=type(error).__name__,
                )
            return GenerationExecutionReport(
                request_id=request.id,
                status=GenerationStatus.FAILED.value,
                attempt=request.attempt_count,
                error_type="GenerationLeaseLostError",
            )
        except Exception as error:
            retry_at = self._retry_at(error, request)
            if retry_at is not None:
                failure = SanitizedFailure(
                    error_type=type(error).__name__,
                    message="transient generation failure; retry scheduled",
                    retryable=True,
                    retry_at=retry_at,
                )
                scheduled = await self._content.reschedule_failed_generation_request(
                    request.id,
                    attempt_count=request.attempt_count,
                    failure=failure,
                    retry_at=retry_at,
                    failed_at=self._now(),
                )
                if scheduled:
                    return GenerationExecutionReport(
                        request_id=request.id,
                        status="retry_scheduled",
                        attempt=request.attempt_count,
                        error_type=type(error).__name__,
                        retry_at=retry_at,
                    )
            return GenerationExecutionReport(
                request_id=request.id,
                status=GenerationStatus.FAILED.value,
                attempt=request.attempt_count,
                error_type=type(error).__name__,
            )
        finally:
            if not execution.done():
                execution.cancel()
            await asyncio.gather(execution, return_exceptions=True)
            renewer.cancel()
            await asyncio.gather(renewer, return_exceptions=True)

    async def _renew_claim(self, request_id: UUID, lease_lost: asyncio.Event) -> None:
        interval = max(
            0.1,
            min(self._polling_interval, self._lease_duration.total_seconds() / 3),
        )
        while not self._stop_event.is_set():
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=interval)
            except TimeoutError:
                try:
                    renewed = await self._content.renew_generation_claim(
                        request_id,
                        self._worker_id,
                        lease_duration=self._lease_duration,
                        now=self._now(),
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as error:
                    self._log.warning(
                        "generation_claim_renewal_failed",
                        request_id=str(request_id),
                        error_type=type(error).__name__,
                    )
                    lease_lost.set()
                    return
                if not renewed:
                    lease_lost.set()
                    return
        lease_lost.set()

    def _retry_at(
        self, error: Exception, request: StoredGenerationRequest
    ) -> datetime | None:
        if isinstance(error, ExampleRateLimitedError):
            retry_after = error.retry_after_seconds
            if retry_after is None or retry_after < 0 or retry_after > 86400:
                return None
            delay = timedelta(seconds=retry_after)
        elif isinstance(
            error,
            (OllamaUnavailableError, OllamaTimeoutError, ExampleTargetUnavailableError),
        ):
            exponent = max(0, request.attempt_count - 1)
            seconds = min(
                self._retry_max_delay.total_seconds(),
                self._retry_base_delay.total_seconds() * (2**exponent),
            )
            jitter = seconds * self._retry_jitter_fraction
            delay = timedelta(
                seconds=min(
                    self._retry_max_delay.total_seconds(),
                    seconds + self._random.uniform(0.0, jitter),
                )
            )
        else:
            return None
        return self._now() + min(delay, self._retry_max_delay)

    def _report(
        self,
        started_at: datetime,
        *,
        enabled: bool,
        paused: bool,
        requests: tuple[GenerationExecutionReport, ...],
    ) -> GenerationRunOnceReport:
        return GenerationRunOnceReport(
            worker_id=self._worker_id,
            started_at=started_at,
            finished_at=self._now(),
            enabled=enabled,
            paused=paused,
            due_count=len(requests),
            claimed_count=len(requests),
            completed_count=sum(item.status == GenerationStatus.COMPLETED.value for item in requests),
            failed_count=sum(item.status == GenerationStatus.FAILED.value for item in requests),
            retry_scheduled_count=sum(item.status == "retry_scheduled" for item in requests),
            results=requests,
        )

    def _now(self) -> datetime:
        value = self._clock()
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("generation worker clock must return an aware timestamp")
        return value.astimezone(UTC)

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("generation worker is closed")


__all__ = [
    "GenerationExecutionReport",
    "GenerationJitterSource",
    "GenerationLeaseLostError",
    "GenerationRunOnceReport",
    "GenerationWorker",
    "GenerationWorkerReport",
]
