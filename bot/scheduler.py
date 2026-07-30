"""Domain-facing action lifecycle and scheduling service."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from random import SystemRandom
from typing import TYPE_CHECKING, Any, Protocol
from uuid import UUID

from pydantic import TypeAdapter, ValidationError
import structlog

from bot.config import Settings
from bot.models import ActionStatus, Platform, SocialAction
from bot.storage.repositories import (
    ActionNotFoundError,
    ActionRepository,
    InvalidStateTransitionError,
    RepositoryError,
)

if TYPE_CHECKING:
    from bot.worker import ActionExecutionReport, Worker


logger = structlog.get_logger(__name__)
_DATETIME_ADAPTER = TypeAdapter(datetime)


class RandomSource(Protocol):
    """Small injectable surface used for deterministic scheduling checks."""

    def uniform(self, a: float, b: float) -> float:
        """Return a floating-point value in the inclusive configured range."""

        ...


@dataclass(frozen=True, slots=True)
class SchedulePreview:
    """Non-persistent view of the lifecycle and time a schedule request would use."""

    action_id: UUID
    platform: Platform
    account_name: str
    current_status: ActionStatus
    resulting_status: ActionStatus
    scheduled_at: datetime
    delay_seconds: float
    requires_approval: bool
    dry_run: bool


@dataclass(frozen=True, slots=True)
class ImportFailure:
    """Safe validation or persistence failure for one imported mapping."""

    index: int
    error_type: str
    message: str


@dataclass(frozen=True, slots=True)
class ImportReport:
    """Result of importing independent JSON-ready action mappings."""

    created: tuple[SocialAction, ...]
    failures: tuple[ImportFailure, ...]

    @property
    def total_count(self) -> int:
        return len(self.created) + len(self.failures)

    @property
    def created_count(self) -> int:
        return len(self.created)

    @property
    def failure_count(self) -> int:
        return len(self.failures)


class SchedulerService:
    """Validate input and expose guarded action lifecycle operations."""

    def __init__(
        self,
        settings: Settings,
        action_repository: ActionRepository,
        *,
        worker: Worker | None = None,
        random_source: RandomSource | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._settings = settings
        self._actions = action_repository
        self._worker = worker
        self._random = random_source or SystemRandom()
        self._clock = clock or (lambda: datetime.now(UTC))
        self._log = logger.bind(component="scheduler")

    @property
    def worker(self) -> Worker | None:
        return self._worker

    def attach_worker(self, worker: Worker) -> None:
        """Attach the execution runtime used by :meth:`execute_now`."""

        self._worker = worker

    async def create_action(
        self,
        action: SocialAction | Mapping[str, Any],
        *,
        scheduled_at: datetime | None = None,
    ) -> SocialAction:
        """Validate and persist a new action under the configured approval gate."""

        validated = self._new_action(action, scheduled_at=scheduled_at)
        self._require_enabled_account(validated)
        requested_at = scheduled_at or validated.scheduled_at
        initial_status = (
            ActionStatus.PENDING_APPROVAL
            if self._settings.manual_approval
            else ActionStatus.DRAFT
        )
        initial = self._reconstruct(
            validated,
            status=initial_status,
            scheduled_at=requested_at if self._settings.manual_approval else None,
            attempts=0,
            last_error=None,
            external_content_id=None,
            external_content_url=None,
        )
        created = await self._actions.create(initial)

        if self._settings.manual_approval:
            self._log.info(
                "action_created",
                action_id=str(created.id),
                platform=created.platform.value,
                account_name=created.account_name,
                status=created.status.value,
            )
            return created

        when, _ = self._schedule_time(requested_at)
        scheduled = await self._actions.schedule(created.id, when)
        self._log.info(
            "action_created",
            action_id=str(scheduled.id),
            platform=scheduled.platform.value,
            account_name=scheduled.account_name,
            status=scheduled.status.value,
            scheduled_at=(
                scheduled.scheduled_at.isoformat()
                if scheduled.scheduled_at is not None
                else None
            ),
        )
        return scheduled

    async def create(
        self,
        action: SocialAction | Mapping[str, Any],
        *,
        scheduled_at: datetime | None = None,
    ) -> SocialAction:
        """CRUD-friendly alias for :meth:`create_action`."""

        return await self.create_action(action, scheduled_at=scheduled_at)

    async def import_actions(
        self,
        actions: Iterable[Mapping[str, Any]],
    ) -> ImportReport:
        """Validate and persist independent JSON-ready mappings in input order."""

        created: list[SocialAction] = []
        failures: list[ImportFailure] = []
        for index, item in enumerate(actions):
            if not isinstance(item, Mapping):
                failures.append(
                    ImportFailure(
                        index=index,
                        error_type="TypeError",
                        message="action must be a JSON object",
                    )
                )
                continue
            try:
                created.append(await self.create_action(item))
            except ValidationError as error:
                failures.append(
                    ImportFailure(
                        index=index,
                        error_type=type(error).__name__,
                        message=_validation_message(error),
                    )
                )
            except (RepositoryError, TypeError, ValueError) as error:
                failures.append(
                    ImportFailure(
                        index=index,
                        error_type=type(error).__name__,
                        message=_safe_exception_message(error),
                    )
                )

        report = ImportReport(created=tuple(created), failures=tuple(failures))
        self._log.info(
            "actions_imported",
            total_count=report.total_count,
            created_count=report.created_count,
            failure_count=report.failure_count,
        )
        return report

    async def get(self, action_id: UUID | str) -> SocialAction:
        """Return an action or raise the repository's domain not-found error."""

        return await self._require_action(action_id)

    async def list(
        self,
        *,
        statuses: tuple[ActionStatus, ...] | list[ActionStatus] | None = None,
        platform: Platform | None = None,
        account_name: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[SocialAction]:
        """List actions through repository filtering and pagination."""

        return await self._actions.list(
            statuses=statuses,
            platform=platform,
            account_name=account_name,
            limit=limit,
            offset=offset,
        )

    async def update(
        self,
        action: SocialAction | Mapping[str, Any],
    ) -> SocialAction:
        """Validate editable action data without bypassing repository guards."""

        if isinstance(action, Mapping):
            values = dict(action)
            _reject_naive_datetimes(values)
            validated = SocialAction.model_validate(values)
        else:
            validated = SocialAction.model_validate(
                action.model_dump(exclude={"fingerprint"})
            )
        self._require_enabled_account(validated)
        if self._settings.manual_approval:
            current = await self._require_action(validated.id)
            if current.status is ActionStatus.FAILED:
                raise InvalidStateTransitionError(
                    f"failed action {current.id} cannot be edited while manual "
                    "approval is enabled"
                )
        return await self._actions.update(validated)

    async def delete(self, action_id: UUID | str) -> bool:
        """Delete a repository-approved non-processing action."""

        return await self._actions.delete(action_id)

    async def preview(
        self,
        action: SocialAction | Mapping[str, Any] | UUID | str,
        *,
        scheduled_at: datetime | None = None,
    ) -> SchedulePreview:
        """Preview approval and randomized scheduling without persistence."""

        stored = isinstance(action, (UUID, str))
        if stored:
            validated = await self._require_action(action)
        else:
            validated = self._new_action(action, scheduled_at=scheduled_at)

        prospective_statuses = {
            ActionStatus.DRAFT,
            ActionStatus.PENDING_APPROVAL,
        }
        if stored and validated.status not in prospective_statuses:
            if scheduled_at is not None:
                _as_utc(scheduled_at, name="scheduled_at")
            return SchedulePreview(
                action_id=validated.id,
                platform=validated.platform,
                account_name=validated.account_name,
                current_status=validated.status,
                resulting_status=validated.status,
                scheduled_at=validated.scheduled_at or validated.created_at,
                delay_seconds=0.0,
                requires_approval=False,
                dry_run=self._settings.dry_run,
            )

        requested_at = scheduled_at or validated.scheduled_at
        when, delay_seconds = self._schedule_time(requested_at)
        requires_approval = (
            validated.status is ActionStatus.PENDING_APPROVAL
            or (
                self._settings.manual_approval
                and validated.status is ActionStatus.DRAFT
            )
        )
        resulting_status = (
            ActionStatus.PENDING_APPROVAL
            if requires_approval
            else ActionStatus.SCHEDULED
        )
        return SchedulePreview(
            action_id=validated.id,
            platform=validated.platform,
            account_name=validated.account_name,
            current_status=validated.status,
            resulting_status=resulting_status,
            scheduled_at=when,
            delay_seconds=delay_seconds,
            requires_approval=requires_approval,
            dry_run=self._settings.dry_run,
        )

    async def approve(
        self,
        action_id: UUID | str,
        *,
        scheduled_at: datetime | None = None,
    ) -> SocialAction:
        """Approve a pending action at its stored time or a new explicit time."""

        action = await self._require_action(action_id)
        if action.status is not ActionStatus.PENDING_APPROVAL:
            raise InvalidStateTransitionError(
                f"cannot approve action {action.id} while {action.status.value}"
            )
        requested_at = scheduled_at or action.scheduled_at
        when, _ = self._schedule_time(requested_at)
        approved = await self._actions.approve(
            action.id,
            scheduled_at=when,
            approved_at=self._now(),
        )
        self._log.info(
            "action_approved",
            action_id=str(approved.id),
            platform=approved.platform.value,
            account_name=approved.account_name,
            scheduled_at=(
                approved.scheduled_at.isoformat()
                if approved.scheduled_at is not None
                else None
            ),
        )
        return approved

    async def schedule(
        self,
        action_id: UUID | str,
        *,
        scheduled_at: datetime | None = None,
    ) -> SocialAction:
        """Request a UTC schedule while preserving the manual approval gate."""

        action = await self._require_action(action_id)
        when, _ = self._schedule_time(scheduled_at)

        if self._settings.manual_approval:
            if action.status is not ActionStatus.PENDING_APPROVAL:
                raise InvalidStateTransitionError(
                    "manual approval is enabled; only pending-approval actions can "
                    "store a requested schedule"
                )
            requested = self._reconstruct(action, scheduled_at=when)
            stored = await self._actions.update(requested)
            self._log.info(
                "approval_schedule_requested",
                action_id=str(stored.id),
                platform=stored.platform.value,
                account_name=stored.account_name,
                scheduled_at=when.isoformat(),
            )
            return stored

        scheduled = await self._actions.schedule(action.id, when)
        self._log.info(
            "action_scheduled",
            action_id=str(scheduled.id),
            platform=scheduled.platform.value,
            account_name=scheduled.account_name,
            scheduled_at=when.isoformat(),
        )
        return scheduled

    async def cancel(
        self,
        action_id: UUID | str,
        *,
        reason: str | None = None,
    ) -> SocialAction:
        """Cancel an action through the repository state guard."""

        cancelled = await self._actions.cancel(
            action_id,
            reason=reason,
            cancelled_at=self._now(),
        )
        self._log.info(
            "action_cancelled",
            action_id=str(cancelled.id),
            platform=cancelled.platform.value,
            account_name=cancelled.account_name,
        )
        return cancelled

    async def execute_now(self, action_id: UUID | str) -> ActionExecutionReport:
        """Request immediate claimed execution from an attached live worker."""

        if self._worker is None:
            raise RuntimeError(
                "execute_now requires an attached Worker; construct SchedulerService "
                "with worker=... or call attach_worker()"
            )

        action = await self._require_action(action_id)
        if action.status is ActionStatus.PENDING_APPROVAL:
            raise RuntimeError(
                f"action {action.id} requires approval before immediate execution"
            )
        if self._settings.manual_approval and action.status is ActionStatus.DRAFT:
            raise RuntimeError(
                f"action {action.id} has not passed the configured manual approval gate"
            )
        if action.status is ActionStatus.CANCELLED:
            raise RuntimeError(f"cancelled action {action.id} cannot be executed")

        if self._worker.dry_run or self._worker.global_paused:
            return await self._worker.execute_now(action.id)

        now = self._now()
        if action.status is ActionStatus.DRAFT:
            action = await self._actions.schedule(action.id, now)
        elif (
            action.status is ActionStatus.SCHEDULED
            and action.scheduled_at is not None
            and action.scheduled_at > now
        ):
            raise RuntimeError(
                f"action {action.id} is scheduled for {action.scheduled_at.isoformat()}; "
                "the repository does not permit bypassing an existing future schedule"
            )

        return await self._worker.execute_now(action.id)

    async def _require_action(self, action_id: UUID | str) -> SocialAction:
        action = await self._actions.get(action_id)
        if action is None:
            raise ActionNotFoundError(f"action {action_id} was not found")
        return action
    def _require_enabled_account(self, action: SocialAction) -> None:
        accounts = (
            self._settings.accounts.x
            if action.platform is Platform.X
            else self._settings.accounts.reddit
        )
        configured = accounts.get(action.account_name)
        if configured is None:
            raise ValueError(
                f"no configured {action.platform.value} account named "
                f"'{action.account_name}'"
            )
        if not configured.enabled:
            raise ValueError(
                f"configured {action.platform.value} account "
                f"'{action.account_name}' is disabled"
            )


    def _new_action(
        self,
        action: SocialAction | Mapping[str, Any],
        *,
        scheduled_at: datetime | None,
    ) -> SocialAction:
        if isinstance(action, SocialAction):
            values = action.model_dump(exclude={"fingerprint"})
        elif isinstance(action, Mapping):
            values = dict(action)
            values.pop("fingerprint", None)
        else:
            raise TypeError("action must be a SocialAction or JSON-ready mapping")

        if scheduled_at is not None:
            values["scheduled_at"] = scheduled_at
        _reject_naive_datetimes(values)
        values.update(
            {
                "status": (
                    ActionStatus.PENDING_APPROVAL
                    if self._settings.manual_approval
                    else ActionStatus.DRAFT
                ),
                "attempts": 0,
                "last_error": None,
                "external_content_id": None,
                "external_content_url": None,
            }
        )
        return SocialAction.model_validate(values)

    @staticmethod
    def _reconstruct(action: SocialAction, **changes: Any) -> SocialAction:
        values = action.model_dump(exclude={"fingerprint"})
        values.update(changes)
        return SocialAction.model_validate(values)

    def _schedule_time(self, requested_at: datetime | None) -> tuple[datetime, float]:
        if requested_at is not None:
            return _as_utc(requested_at, name="scheduled_at"), 0.0
        delay = self._random.uniform(
            float(self._settings.randomized_delay.minimum_seconds),
            float(self._settings.randomized_delay.maximum_seconds),
        )
        return self._now() + timedelta(seconds=delay), delay

    def _now(self) -> datetime:
        return _as_utc(self._clock(), name="clock")


def _as_utc(value: datetime, *, name: str) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"{name} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must include timezone information")
    return value.astimezone(UTC)

def _reject_naive_datetimes(values: Mapping[str, Any]) -> None:
    for name in ("created_at", "scheduled_at"):
        value = values.get(name)
        if value is None:
            continue
        try:
            parsed = _DATETIME_ADAPTER.validate_python(value)
        except ValidationError:
            continue
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError(f"{name} must include timezone information")


def _validation_message(error: ValidationError) -> str:
    details: list[str] = []
    for issue in error.errors(include_input=False, include_url=False):
        location = ".".join(str(part) for part in issue.get("loc", ())) or "action"
        details.append(f"{location}: {issue.get('msg', 'invalid value')}")
    return "; ".join(details) or "action validation failed"


def _safe_exception_message(error: Exception) -> str:
    message = " ".join(str(error).split())
    return message[:500] if message else type(error).__name__


__all__ = [
    "ImportFailure",
    "ImportReport",
    "RandomSource",
    "SchedulePreview",
    "SchedulerService",
]
