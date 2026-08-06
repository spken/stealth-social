"""Publishing-worker construction used by direct execution and the CLI worker."""

from __future__ import annotations

import asyncio
import signal
from dataclasses import asdict, dataclass
from typing import Annotated
from uuid import UUID

import typer

from bot.browser.manager import BrowserManager
from bot.commands.common import _emit_json, _run_async, _safe_command, _settings
from bot.config import Settings
from bot.content.runtime import content_runtime
from bot.storage.database import Database
from bot.storage.repositories import AccountStateRepository, ActionRepository
from bot.worker import ActionExecutionReport, RunOnceReport, Worker, WorkerReport
from bot.worker_supervisor import (
    SupervisorReport,
    SupervisorRunOnceReport,
    WorkerQueue,
    WorkerSupervisor,
)

worker_app = typer.Typer(add_completion=False)


@dataclass(frozen=True, slots=True)
class WorkerRuntime:
    """Resources composed for a publishing-only worker."""

    worker: Worker
    database: Database
    actions: ActionRepository


def build_worker(
    settings: Settings,
    *,
    database: Database | None = None,
    browser_manager: BrowserManager | None = None,
    action_repository: ActionRepository | None = None,
    account_state_repository: AccountStateRepository | None = None,
    close_shared_resources: bool = True,
) -> WorkerRuntime:
    """Build a publishing worker with optionally shared runtime resources."""

    resolved_database = database or Database(settings.database_url)
    actions = action_repository or ActionRepository(resolved_database.session_factory)
    account_states = account_state_repository or AccountStateRepository(
        resolved_database.session_factory
    )
    resolved_browser = browser_manager or BrowserManager(settings)
    worker = Worker(
        settings,
        action_repository=actions,
        account_state_repository=account_states,
        browser_manager=resolved_browser,
        database=resolved_database,
        close_shared_resources=close_shared_resources,
    )
    return WorkerRuntime(
        worker=worker,
        database=resolved_database,
        actions=actions,
    )


async def execute_action(settings: Settings, action_id: UUID) -> ActionExecutionReport:
    """Execute one action through the existing publishing safety gates."""

    runtime = build_worker(settings)
    try:
        await runtime.database.initialize()
        from bot.scheduler import SchedulerService

        scheduler = SchedulerService(settings, runtime.actions, worker=runtime.worker)
        return await scheduler.execute_now(action_id)
    finally:
        await runtime.worker.close()


async def run_publishing_worker(
    settings: Settings,
    *,
    once: bool,
) -> RunOnceReport | WorkerReport:
    """Run the publishing worker with its historical direct-worker lifecycle."""

    runtime = build_worker(settings)
    worker = runtime.worker
    loop = asyncio.get_running_loop()
    installed: list[signal.Signals] = []
    try:
        for interrupt_signal in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(interrupt_signal, worker.stop)
            except (NotImplementedError, RuntimeError, ValueError):
                continue
            installed.append(interrupt_signal)
        if once:
            return await worker.run_once()
        return await worker.run()
    finally:
        for interrupt_signal in installed:
            loop.remove_signal_handler(interrupt_signal)
        await worker.close()


async def run_supervised_worker(
    settings: Settings,
    *,
    once: bool,
    queue: WorkerQueue,
) -> SupervisorRunOnceReport | SupervisorReport:
    """Run selected publishing/generation queues over one shared runtime."""

    async with content_runtime(settings) as runtime:
        publishing = build_worker(
            settings,
            database=runtime.database,
            browser_manager=runtime.browser_manager,
            action_repository=runtime.action_repository,
            account_state_repository=runtime.account_state_repository,
            close_shared_resources=False,
        ).worker
        supervisor = WorkerSupervisor(
            publishing,
            runtime.generation_worker,
            queue=queue,
        )
        loop = asyncio.get_running_loop()
        installed: list[signal.Signals] = []
        try:
            for interrupt_signal in (signal.SIGINT, signal.SIGTERM):
                try:
                    loop.add_signal_handler(interrupt_signal, supervisor.stop)
                except (NotImplementedError, RuntimeError, ValueError):
                    continue
                installed.append(interrupt_signal)
            if once:
                return await supervisor.run_once()
            return await supervisor.run()
        finally:
            for interrupt_signal in installed:
                loop.remove_signal_handler(interrupt_signal)
            await supervisor.close()


@worker_app.callback(invoke_without_command=True)
@_safe_command
def worker_command(
    once: Annotated[bool, typer.Option("--once", help="Process one selected batch and exit.")] = False,
    queue: Annotated[
        WorkerQueue,
        typer.Option("--queue", help="Queue to run: all, publishing, or generation."),
    ] = WorkerQueue.ALL,
) -> None:
    """Run the selected publishing and/or scheduled-generation queue."""

    if not once:
        typer.echo("Worker running; press Ctrl-C to stop gracefully.", err=True)
    report = _run_async(
        run_supervised_worker(_settings(), once=once, queue=queue)
    )
    _emit_json(
        {
            "mode": "once" if once else "persistent",
            "report": asdict(report),
        }
    )


def register_worker_command(app: typer.Typer) -> None:
    app.add_typer(worker_app, name="worker")


__all__ = [
    "WorkerQueue",
    "WorkerRuntime",
    "build_worker",
    "execute_action",
    "run_publishing_worker",
    "run_supervised_worker",
    "register_worker_command",
    "worker_app",
    "worker_command",
]
