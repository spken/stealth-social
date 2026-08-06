"""Coordinate publishing and generation queues without merging their leases."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum

from bot.generation_worker import GenerationRunOnceReport, GenerationWorker
from bot.worker import RunOnceReport, Worker


class WorkerQueue(StrEnum):
    ALL = "all"
    PUBLISHING = "publishing"
    GENERATION = "generation"


@dataclass(frozen=True, slots=True)
class SupervisorRunOnceReport:
    queue: WorkerQueue
    started_at: datetime
    finished_at: datetime
    publishing: RunOnceReport | None
    generation: GenerationRunOnceReport | None


@dataclass(frozen=True, slots=True)
class SupervisorReport:
    queue: WorkerQueue
    started_at: datetime
    stopped_at: datetime
    cycles: int
    last_run: SupervisorRunOnceReport | None


class WorkerSupervisor:
    """Own one polling loop and one shutdown path for both queues."""

    def __init__(
        self,
        publishing_worker: Worker,
        generation_worker: GenerationWorker,
        *,
        queue: WorkerQueue = WorkerQueue.ALL,
        polling_interval: float = 5.0,
    ) -> None:
        if polling_interval <= 0:
            raise ValueError("polling_interval must be positive")
        self._publishing = publishing_worker
        self._generation = generation_worker
        self._queue = WorkerQueue(queue)
        self._polling_interval = float(polling_interval)
        self._stop_event = asyncio.Event()
        self._close_lock = asyncio.Lock()
        self._closed = False
        self._running = False

    @property
    def queue(self) -> WorkerQueue:
        return self._queue

    def stop(self) -> None:
        self._stop_event.set()
        self._publishing.stop()
        self._generation.stop()

    async def run_once(self) -> SupervisorRunOnceReport:
        self._ensure_open()
        started_at = datetime.now(UTC)
        publishing: RunOnceReport | None = None
        generation: GenerationRunOnceReport | None = None
        if self._queue is WorkerQueue.ALL:
            publishing, generation = await asyncio.gather(
                self._publishing.run_once(), self._generation.run_once()
            )
        elif self._queue is WorkerQueue.PUBLISHING:
            publishing = await self._publishing.run_once()
        else:
            generation = await self._generation.run_once()
        return SupervisorRunOnceReport(
            queue=self._queue,
            started_at=started_at,
            finished_at=datetime.now(UTC),
            publishing=publishing,
            generation=generation,
        )

    async def run(self) -> SupervisorReport:
        self._ensure_open()
        if self._running:
            raise RuntimeError("worker supervisor is already running")
        self._running = True
        started_at = datetime.now(UTC)
        cycles = 0
        last_run: SupervisorRunOnceReport | None = None
        try:
            while not self._stop_event.is_set():
                last_run = await self.run_once()
                cycles += 1
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
        return SupervisorReport(
            queue=self._queue,
            started_at=started_at,
            stopped_at=datetime.now(UTC),
            cycles=cycles,
            last_run=last_run,
        )

    async def close(self) -> None:
        async with self._close_lock:
            if self._closed:
                return
            self.stop()
            try:
                await asyncio.gather(
                    self._publishing.close(),
                    self._generation.close(),
                )
            finally:
                self._closed = True

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("worker supervisor is closed")


__all__ = [
    "SupervisorReport",
    "SupervisorRunOnceReport",
    "WorkerQueue",
    "WorkerSupervisor",
]
