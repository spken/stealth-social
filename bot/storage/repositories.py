"""Domain-facing async repositories with atomic SQLite state transitions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

from sqlalchemy import and_, case, delete, func, or_, select, update
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from bot.models import (
    ActionStatus,
    ActionType,
    Platform,
    SocialAction,
    canonical_target_scope,
)
from bot.storage.models import (
    AccountStateRecord,
    ActionEventRecord,
    SocialActionRecord,
)

SessionFactory = async_sessionmaker[AsyncSession]


class RepositoryError(RuntimeError):
    """Base class for persistence operation failures."""


class ActionNotFoundError(RepositoryError):
    """Raised when an action id is not present."""


class ActionAlreadyExistsError(RepositoryError):
    """Raised when an action id already exists."""


class InvalidStateTransitionError(RepositoryError):
    """Raised when an action cannot enter the requested state."""


class ClaimConflictError(RepositoryError):
    """Raised when a worker no longer owns an execution claim."""


@dataclass(frozen=True, slots=True)
class RateUsage:
    """Published-action usage for rolling safety windows."""

    hourly: int
    daily: int
    last_published_at: datetime | None


@dataclass(frozen=True, slots=True)
class AccountState:
    """Domain-safe view of stored account safety state."""

    platform: Platform
    account_name: str
    paused: bool
    paused_at: datetime | None
    paused_until: datetime | None
    pause_reason: str | None
    consecutive_failures: int
    total_failures: int
    failure_threshold: int | None
    last_failure_at: datetime | None
    last_success_at: datetime | None
    last_error: str | None
    created_at: datetime
    updated_at: datetime
    execution_owner: str | None = None
    execution_expires_at: datetime | None = None

    def is_paused(self, at: datetime | None = None) -> bool:
        """Return whether the pause is active at an aware UTC timestamp."""
        instant = _as_utc(at or datetime.now(UTC), name="at")
        return self.paused and (
            self.paused_until is None or self.paused_until > instant
        )


@dataclass(frozen=True, slots=True)
class ActionEvent:
    """Domain-safe view of one persisted state-transition event."""

    id: int
    action_id: UUID
    event_type: str
    from_status: ActionStatus | None
    to_status: ActionStatus | None
    message: str | None
    page_url: str | None
    screenshot_path: str | None
    context: dict[str, Any]
    created_at: datetime


def _as_utc(value: datetime, *, name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return value.astimezone(UTC)


def _now(value: datetime | None = None) -> datetime:
    return _as_utc(value or datetime.now(UTC), name="now")


def _platform_value(platform: Platform | str) -> str:
    return Platform(platform).value


def _status_value(status: ActionStatus | str) -> str:
    return ActionStatus(status).value


def _json_object(value: dict[str, Any] | None) -> dict[str, Any]:
    """Validate JSON serializability and detach caller-owned mutable data."""
    candidate = value or {}
    try:
        encoded = json.dumps(candidate, allow_nan=False, separators=(",", ":"))
        decoded = json.loads(encoded)
    except (TypeError, ValueError) as error:
        raise ValueError("metadata/context must contain only JSON values") from error
    if not isinstance(decoded, dict):
        raise ValueError("metadata/context must be a JSON object")
    return decoded

def _revalidate_action(action: SocialAction) -> SocialAction:
    """Re-run domain validation and derive the fingerprint from current fields."""
    return SocialAction.model_validate(
        action.model_dump(exclude={"fingerprint"})
    )


def target_scope_for(action: SocialAction) -> str:
    """Build the semantic destination used by duplicate protection."""
    return canonical_target_scope(action)


def _action_values(action: SocialAction) -> dict[str, Any]:
    return {
        "action_type": ActionType(action.action_type).value,
        "platform": Platform(action.platform).value,
        "account_name": action.account_name,
        "content": action.content,
        "title": action.title,
        "subreddit": action.subreddit,
        "target_url": str(action.target_url) if action.target_url is not None else None,
        "parent_post_id": action.parent_post_id,
        "parent_comment_id": action.parent_comment_id,
        "scheduled_at": (
            _as_utc(action.scheduled_at, name="scheduled_at")
            if action.scheduled_at is not None
            else None
        ),
        "max_attempts": action.max_attempts,
        "last_error": action.last_error,
        "external_content_id": action.external_content_id,
        "external_content_url": (
            str(action.external_content_url)
            if action.external_content_url is not None
            else None
        ),
        "metadata_json": _json_object(action.metadata),
        "fingerprint": action.fingerprint,
        "target_scope": target_scope_for(action),
    }


def _to_action(record: SocialActionRecord) -> SocialAction:
    return SocialAction.model_validate(
        {
            "id": UUID(record.id),
            "action_type": record.action_type,
            "platform": record.platform,
            "account_name": record.account_name,
            "content": record.content,
            "title": record.title,
            "subreddit": record.subreddit,
            "target_url": record.target_url,
            "parent_post_id": record.parent_post_id,
            "parent_comment_id": record.parent_comment_id,
            "created_at": record.created_at,
            "scheduled_at": record.scheduled_at,
            "status": record.status,
            "attempts": record.attempts,
            "max_attempts": record.max_attempts,
            "last_error": record.last_error,
            "external_content_id": record.external_content_id,
            "external_content_url": record.external_content_url,
            "metadata": _json_object(record.metadata_json),
            "fingerprint": record.fingerprint,
        }
    )


def _to_account_state(record: AccountStateRecord) -> AccountState:
    return AccountState(
        platform=Platform(record.platform),
        account_name=record.account_name,
        paused=record.paused,
        paused_at=record.paused_at,
        paused_until=record.paused_until,
        pause_reason=record.pause_reason,
        consecutive_failures=record.consecutive_failures,
        total_failures=record.total_failures,
        failure_threshold=record.failure_threshold,
        last_failure_at=record.last_failure_at,
        last_success_at=record.last_success_at,
        last_error=record.last_error,
        created_at=record.created_at,
        updated_at=record.updated_at,
        execution_owner=record.execution_owner,
        execution_expires_at=record.execution_expires_at,
    )


def _to_event(record: ActionEventRecord) -> ActionEvent:
    return ActionEvent(
        id=record.id,
        action_id=UUID(record.action_id),
        event_type=record.event_type,
        from_status=(
            ActionStatus(record.from_status) if record.from_status is not None else None
        ),
        to_status=(
            ActionStatus(record.to_status) if record.to_status is not None else None
        ),
        message=record.message,
        page_url=record.page_url,
        screenshot_path=record.screenshot_path,
        context=_json_object(record.context_json),
        created_at=record.created_at,
    )


def _event(
    action_id: str,
    event_type: str,
    *,
    from_status: ActionStatus | str | None = None,
    to_status: ActionStatus | str | None = None,
    message: str | None = None,
    page_url: str | None = None,
    screenshot_path: str | None = None,
    context: dict[str, Any] | None = None,
    created_at: datetime,
) -> ActionEventRecord:
    return ActionEventRecord(
        action_id=action_id,
        event_type=event_type,
        from_status=_status_value(from_status) if from_status is not None else None,
        to_status=_status_value(to_status) if to_status is not None else None,
        message=message,
        page_url=page_url,
        screenshot_path=screenshot_path,
        context_json=_json_object(context),
        created_at=created_at,
    )


class ActionRepository:
    """Persist actions and enforce their claim/state-transition invariants."""

    def __init__(self, session_factory: SessionFactory) -> None:
        self._sessions = session_factory

    async def create(self, action: SocialAction) -> SocialAction:
        action = _revalidate_action(action)
        status = ActionStatus(action.status)
        if status not in {
            ActionStatus.DRAFT,
            ActionStatus.PENDING_APPROVAL,
        }:
            raise InvalidStateTransitionError(
                f"cannot create action {action.id} while {status.value}"
            )
        now = _now()
        values = _action_values(action)
        record = SocialActionRecord(
            id=str(action.id),
            created_at=_as_utc(action.created_at, name="created_at"),
            status=status.value,
            attempts=action.attempts,
            published_at=None,
            updated_at=now,
            **values,
        )
        try:
            async with self._sessions() as session:
                async with session.begin():
                    session.add(record)
                    session.add(
                        _event(
                            record.id,
                            "created",
                            to_status=action.status,
                            created_at=now,
                        )
                    )
        except IntegrityError as error:
            raise ActionAlreadyExistsError(f"action {action.id} already exists") from error
        return _to_action(record)

    async def get(self, action_id: UUID | str) -> SocialAction | None:
        async with self._sessions() as session:
            record = await session.get(SocialActionRecord, str(action_id))
            return _to_action(record) if record is not None else None

    async def list(
        self,
        *,
        statuses: tuple[ActionStatus, ...] | list[ActionStatus] | None = None,
        platform: Platform | None = None,
        account_name: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[SocialAction]:
        if limit <= 0:
            raise ValueError("limit must be positive")
        if offset < 0:
            raise ValueError("offset cannot be negative")
        statement = select(SocialActionRecord)
        if statuses:
            statement = statement.where(
                SocialActionRecord.status.in_([_status_value(item) for item in statuses])
            )
        if platform is not None:
            statement = statement.where(
                SocialActionRecord.platform == _platform_value(platform)
            )
        if account_name is not None:
            statement = statement.where(
                SocialActionRecord.account_name == account_name
            )
        statement = statement.order_by(
            SocialActionRecord.created_at.desc(), SocialActionRecord.id
        ).limit(limit).offset(offset)
        async with self._sessions() as session:
            records = (await session.scalars(statement)).all()
            return [_to_action(record) for record in records]

    async def update(self, action: SocialAction) -> SocialAction:
        """Update editable data without bypassing transition or claim guards."""
        action = _revalidate_action(action)
        status = ActionStatus(action.status)
        if status in {
            ActionStatus.SCHEDULED,
            ActionStatus.PROCESSING,
            ActionStatus.PUBLISHED,
            ActionStatus.CANCELLED,
        }:
            raise InvalidStateTransitionError(
                f"{status.value} actions are immutable through CRUD update"
            )
        now = _now()
        values = _action_values(action)
        statement = (
            update(SocialActionRecord)
            .where(
                SocialActionRecord.id == str(action.id),
                SocialActionRecord.status == status.value,
            )
            .values(**values, updated_at=now)
            .returning(SocialActionRecord)
        )
        async with self._sessions() as session:
            async with session.begin():
                record = (await session.scalars(statement)).one_or_none()
                if record is None:
                    await self._raise_missing_or_transition(
                        session, action.id, "update"
                    )
            return _to_action(record)

    async def delete(self, action_id: UUID | str) -> bool:
        """Delete a non-processing, non-published action."""
        identifier = str(action_id)
        statement = (
            delete(SocialActionRecord)
            .where(
                SocialActionRecord.id == identifier,
                SocialActionRecord.status.not_in(
                    [ActionStatus.PROCESSING.value, ActionStatus.PUBLISHED.value]
                ),
            )
            .returning(SocialActionRecord.id)
        )
        async with self._sessions() as session:
            async with session.begin():
                deleted_id = (await session.scalars(statement)).one_or_none()
                if deleted_id is not None:
                    return True
                status = await session.scalar(
                    select(SocialActionRecord.status).where(
                        SocialActionRecord.id == identifier
                    )
                )
                if status is None:
                    return False
                raise InvalidStateTransitionError(
                    f"cannot delete action {identifier} while {status}"
                )

    async def schedule(
        self,
        action_id: UUID | str,
        scheduled_at: datetime,
    ) -> SocialAction:
        when = _as_utc(scheduled_at, name="scheduled_at")
        now = _now()
        allowed = [ActionStatus.DRAFT.value, ActionStatus.FAILED.value]
        statement = (
            update(SocialActionRecord)
            .where(
                SocialActionRecord.id == str(action_id),
                SocialActionRecord.status.in_(allowed),
                SocialActionRecord.attempts < SocialActionRecord.max_attempts,
            )
            .values(
                status=ActionStatus.SCHEDULED.value,
                scheduled_at=when,
                retry_available_at=None,
                claim_owner=None,
                claim_expires_at=None,
                external_dispatch_started_at=None,
                updated_at=now,
            )
            .returning(SocialActionRecord)
        )
        return await self._apply_transition(
            statement,
            action_id,
            operation="schedule",
            event_type="scheduled",
            from_status=None,
            to_status=ActionStatus.SCHEDULED,
            event_at=now,
        )

    async def approve(
        self,
        action_id: UUID | str,
        *,
        scheduled_at: datetime | None = None,
        approved_at: datetime | None = None,
    ) -> SocialAction:
        now = _now(approved_at)
        when = (
            _as_utc(scheduled_at, name="scheduled_at")
            if scheduled_at is not None
            else None
        )
        schedule_value: Any = when if when is not None else func.coalesce(
            SocialActionRecord.scheduled_at, now
        )
        statement = (
            update(SocialActionRecord)
            .where(
                SocialActionRecord.id == str(action_id),
                SocialActionRecord.status == ActionStatus.PENDING_APPROVAL.value,
            )
            .values(
                status=ActionStatus.SCHEDULED.value,
                scheduled_at=schedule_value,
                updated_at=now,
            )
            .returning(SocialActionRecord)
        )
        return await self._apply_transition(
            statement,
            action_id,
            operation="approve",
            event_type="approved",
            from_status=ActionStatus.PENDING_APPROVAL,
            to_status=ActionStatus.SCHEDULED,
            event_at=now,
        )

    async def cancel(
        self,
        action_id: UUID | str,
        *,
        reason: str | None = None,
        cancelled_at: datetime | None = None,
    ) -> SocialAction:
        now = _now(cancelled_at)
        cancellable = [
            ActionStatus.DRAFT.value,
            ActionStatus.PENDING_APPROVAL.value,
            ActionStatus.SCHEDULED.value,
            ActionStatus.FAILED.value,
        ]
        statement = (
            update(SocialActionRecord)
            .where(
                SocialActionRecord.id == str(action_id),
                SocialActionRecord.status.in_(cancellable),
            )
            .values(
                status=ActionStatus.CANCELLED.value,
                retry_available_at=None,
                claim_owner=None,
                claim_expires_at=None,
                updated_at=now,
            )
            .returning(SocialActionRecord)
        )
        return await self._apply_transition(
            statement,
            action_id,
            operation="cancel",
            event_type="cancelled",
            from_status=None,
            to_status=ActionStatus.CANCELLED,
            event_at=now,
            message=reason,
        )

    async def list_due_actions(
        self,
        limit: int = 100,
        now: datetime | None = None,
    ) -> list[SocialAction]:
        """List currently eligible work without reserving or mutating it."""
        if limit <= 0:
            raise ValueError("limit must be positive")
        instant = _now(now)
        due_at = func.coalesce(
            SocialActionRecord.retry_available_at,
            SocialActionRecord.scheduled_at,
            SocialActionRecord.created_at,
        )
        statement = (
            select(SocialActionRecord)
            .where(self._due_predicate(instant))
            .order_by(due_at, SocialActionRecord.created_at, SocialActionRecord.id)
            .limit(limit)
        )
        async with self._sessions() as session:
            records = (await session.scalars(statement)).all()
            return [_to_action(record) for record in records]

    async def claim_due_actions(
        self,
        worker_id: str,
        *,
        limit: int = 1,
        lease_duration: timedelta = timedelta(minutes=5),
        now: datetime | None = None,
    ) -> list[SocialAction]:
        """Atomically claim due work with a guarded compare-and-update."""
        owner = worker_id.strip()
        if not owner:
            raise ValueError("worker_id cannot be empty")
        if limit <= 0:
            raise ValueError("limit must be positive")
        if lease_duration <= timedelta(0):
            raise ValueError("lease_duration must be positive")
        instant = _now(now)
        lease_expires = instant + lease_duration
        eligible = self._due_predicate(instant)
        due_at = func.coalesce(
            SocialActionRecord.retry_available_at,
            SocialActionRecord.scheduled_at,
            SocialActionRecord.created_at,
        )
        candidates = (
            select(SocialActionRecord.id)
            .where(eligible)
            .order_by(due_at, SocialActionRecord.created_at, SocialActionRecord.id)
            .limit(limit)
        )
        statement = (
            update(SocialActionRecord)
            .where(SocialActionRecord.id.in_(candidates), self._due_predicate(instant))
            .values(
                status=ActionStatus.PROCESSING.value,
                attempts=SocialActionRecord.attempts + 1,
                claim_owner=owner,
                claim_expires_at=lease_expires,
                external_dispatch_started_at=None,
                retry_available_at=None,
                updated_at=instant,
            )
            .returning(SocialActionRecord)
        )
        async with self._sessions() as session:
            async with session.begin():
                records = list((await session.scalars(statement)).all())
                for record in records:
                    session.add(
                        _event(
                            record.id,
                            "claimed",
                            to_status=ActionStatus.PROCESSING,
                            context={
                                "worker_id": owner,
                                "claim_expires_at": lease_expires.isoformat(),
                                "attempt": record.attempts,
                            },
                            created_at=instant,
                        )
                    )
            records.sort(
                key=lambda record: (
                    record.scheduled_at or record.created_at,
                    record.created_at,
                    record.id,
                )
            )
            return [_to_action(record) for record in records]

    async def claim_action(
        self,
        action_id: UUID | str,
        worker_id: str,
        *,
        lease_duration: timedelta = timedelta(minutes=5),
        now: datetime | None = None,
    ) -> SocialAction | None:
        """Claim one specific action if it is due; return None on contention."""
        owner = worker_id.strip()
        if not owner:
            raise ValueError("worker_id cannot be empty")
        if lease_duration <= timedelta(0):
            raise ValueError("lease_duration must be positive")
        instant = _now(now)
        statement = (
            update(SocialActionRecord)
            .where(
                SocialActionRecord.id == str(action_id),
                self._due_predicate(instant),
            )
            .values(
                status=ActionStatus.PROCESSING.value,
                attempts=SocialActionRecord.attempts + 1,
                claim_owner=owner,
                claim_expires_at=instant + lease_duration,
                external_dispatch_started_at=None,
                retry_available_at=None,
                updated_at=instant,
            )
            .returning(SocialActionRecord)
        )
        async with self._sessions() as session:
            async with session.begin():
                record = (await session.scalars(statement)).one_or_none()
                if record is None:
                    return None
                session.add(
                    _event(
                        record.id,
                        "claimed",
                        to_status=ActionStatus.PROCESSING,
                        context={"worker_id": owner, "attempt": record.attempts},
                        created_at=instant,
                    )
                )
            return _to_action(record)

    async def verify_claim(
        self,
        action_id: UUID | str,
        worker_id: str,
        now: datetime | None = None,
    ) -> bool:
        """Return whether a worker still owns an unexpired processing claim."""
        owner = worker_id.strip()
        if not owner:
            raise ValueError("worker_id cannot be empty")
        instant = _now(now)
        statement = select(SocialActionRecord.id).where(
            SocialActionRecord.id == str(action_id),
            SocialActionRecord.status == ActionStatus.PROCESSING.value,
            SocialActionRecord.claim_owner == owner,
            SocialActionRecord.claim_expires_at.is_not(None),
            SocialActionRecord.claim_expires_at > instant,
        )
        async with self._sessions() as session:
            return (await session.scalar(statement)) is not None

    async def begin_external_dispatch(
        self,
        action_id: UUID | str,
        worker_id: str,
        now: datetime | None = None,
    ) -> SocialAction:
        """Durably mark the boundary immediately before an external call."""
        owner = worker_id.strip()
        if not owner:
            raise ValueError("worker_id cannot be empty")
        instant = _now(now)
        statement = (
            update(SocialActionRecord)
            .where(
                SocialActionRecord.id == str(action_id),
                SocialActionRecord.status == ActionStatus.PROCESSING.value,
                SocialActionRecord.claim_owner == owner,
                SocialActionRecord.claim_expires_at.is_not(None),
                SocialActionRecord.claim_expires_at > instant,
                SocialActionRecord.external_dispatch_started_at.is_(None),
            )
            .values(
                external_dispatch_started_at=instant,
                updated_at=instant,
            )
            .returning(SocialActionRecord)
        )
        async with self._sessions() as session:
            async with session.begin():
                record = (await session.scalars(statement)).one_or_none()
                if record is None:
                    await self._raise_claim_conflict(
                        session, action_id, owner, "begin external dispatch for"
                    )
                session.add(
                    _event(
                        record.id,
                        "external_dispatch_started",
                        from_status=ActionStatus.PROCESSING,
                        to_status=ActionStatus.PROCESSING,
                        context={
                            "worker_id": owner,
                            "external_dispatch_started_at": instant.isoformat(),
                        },
                        created_at=instant,
                    )
                )
            return _to_action(record)

    async def defer_claim(
        self,
        action_id: UUID | str,
        worker_id: str,
        scheduled_at: datetime,
        reason: str,
    ) -> SocialAction:
        """Release pre-dispatch work without consuming an execution attempt."""
        owner = worker_id.strip()
        if not owner:
            raise ValueError("worker_id cannot be empty")
        diagnostic = reason.strip()
        if not diagnostic:
            raise ValueError("reason cannot be empty")
        retry_at = _as_utc(scheduled_at, name="scheduled_at")
        instant = _now()
        restored_attempts = case(
            (
                SocialActionRecord.attempts > 0,
                SocialActionRecord.attempts - 1,
            ),
            else_=0,
        )
        statement = (
            update(SocialActionRecord)
            .where(
                SocialActionRecord.id == str(action_id),
                SocialActionRecord.status == ActionStatus.PROCESSING.value,
                SocialActionRecord.claim_owner == owner,
                SocialActionRecord.external_dispatch_started_at.is_(None),
            )
            .values(
                status=ActionStatus.FAILED.value,
                attempts=restored_attempts,
                last_error=diagnostic,
                retry_available_at=retry_at,
                error_page_url=None,
                error_screenshot_path=None,
                external_dispatch_started_at=None,
                claim_owner=None,
                claim_expires_at=None,
                updated_at=instant,
            )
            .returning(SocialActionRecord)
        )
        async with self._sessions() as session:
            async with session.begin():
                record = (await session.scalars(statement)).one_or_none()
                if record is None:
                    await self._raise_claim_conflict(
                        session, action_id, owner, "defer"
                    )
                session.add(
                    _event(
                        record.id,
                        "deferred",
                        from_status=ActionStatus.PROCESSING,
                        to_status=ActionStatus.FAILED,
                        message=diagnostic,
                        context={
                            "worker_id": owner,
                            "retry_available_at": retry_at.isoformat(),
                            "restored_attempt": record.attempts,
                        },
                        created_at=instant,
                    )
                )
            return _to_action(record)

    async def renew_claim(
        self,
        action_id: UUID | str,
        worker_id: str,
        *,
        lease_duration: timedelta = timedelta(minutes=5),
        now: datetime | None = None,
    ) -> bool:
        if lease_duration <= timedelta(0):
            raise ValueError("lease_duration must be positive")
        owner = worker_id.strip()
        if not owner:
            raise ValueError("worker_id cannot be empty")
        instant = _now(now)
        statement = (
            update(SocialActionRecord)
            .where(
                SocialActionRecord.id == str(action_id),
                SocialActionRecord.status == ActionStatus.PROCESSING.value,
                SocialActionRecord.claim_owner == owner,
                SocialActionRecord.claim_expires_at > instant,
            )
            .values(
                claim_expires_at=instant + lease_duration,
                updated_at=instant,
            )
            .returning(SocialActionRecord.id)
        )
        async with self._sessions() as session:
            async with session.begin():
                return (await session.scalar(statement)) is not None

    async def recover_stale_claims(
        self,
        *,
        now: datetime | None = None,
    ) -> list[SocialAction]:
        """Recover only pre-dispatch leases; uncertain outcomes require review."""
        instant = _now(now)
        lease_error = "execution lease expired"
        uncertain_error = (
            "execution lease expired after external dispatch started; "
            "outcome uncertain; manual review required"
        )
        dispatch_started = (
            SocialActionRecord.external_dispatch_started_at.is_not(None)
        )
        retry_at = case(
            (
                and_(
                    SocialActionRecord.external_dispatch_started_at.is_(None),
                    SocialActionRecord.attempts < SocialActionRecord.max_attempts,
                ),
                instant,
            ),
            else_=None,
        )
        statement = (
            update(SocialActionRecord)
            .where(
                SocialActionRecord.status == ActionStatus.PROCESSING.value,
                SocialActionRecord.claim_expires_at.is_not(None),
                SocialActionRecord.claim_expires_at <= instant,
            )
            .values(
                status=ActionStatus.FAILED.value,
                last_error=case(
                    (dispatch_started, uncertain_error),
                    else_=lease_error,
                ),
                retry_available_at=retry_at,
                claim_owner=None,
                claim_expires_at=None,
                updated_at=instant,
            )
            .returning(SocialActionRecord)
        )
        async with self._sessions() as session:
            async with session.begin():
                records = list((await session.scalars(statement)).all())
                for record in records:
                    uncertain = record.external_dispatch_started_at is not None
                    if uncertain:
                        existing_indefinite_pause = and_(
                            AccountStateRecord.paused.is_(True),
                            AccountStateRecord.paused_until.is_(None),
                        )
                        pause_insert = sqlite_insert(AccountStateRecord).values(
                            platform=record.platform,
                            account_name=record.account_name,
                            paused=True,
                            paused_at=instant,
                            paused_until=None,
                            pause_reason=uncertain_error,
                            created_at=instant,
                            updated_at=instant,
                        )
                        pause_statement = pause_insert.on_conflict_do_update(
                            index_elements=[
                                AccountStateRecord.platform,
                                AccountStateRecord.account_name,
                            ],
                            set_={
                                "paused": True,
                                "paused_at": case(
                                    (
                                        existing_indefinite_pause,
                                        func.coalesce(
                                            AccountStateRecord.paused_at,
                                            instant,
                                        ),
                                    ),
                                    else_=instant,
                                ),
                                "paused_until": None,
                                "pause_reason": uncertain_error,
                                "updated_at": instant,
                            },
                        )
                        await session.execute(pause_statement)
                    session.add(
                        _event(
                            record.id,
                            (
                                "claim_expired_uncertain"
                                if uncertain
                                else "claim_expired"
                            ),
                            from_status=ActionStatus.PROCESSING,
                            to_status=ActionStatus.FAILED,
                            message=uncertain_error if uncertain else lease_error,
                            context=(
                                {
                                    "external_dispatch_started_at": (
                                        record.external_dispatch_started_at.isoformat()
                                    ),
                                    "manual_review_required": True,
                                    "account_paused_indefinitely": True,
                                    "platform": record.platform,
                                    "account_name": record.account_name,
                                }
                                if uncertain
                                else None
                            ),
                            created_at=instant,
                        )
                    )
            return [_to_action(record) for record in records]

    async def mark_published(
        self,
        action_id: UUID | str,
        worker_id: str,
        *,
        external_content_id: str | None = None,
        external_content_url: str | None = None,
        published_at: datetime | None = None,
    ) -> SocialAction:
        """Complete an owned processing claim exactly once."""
        owner = worker_id.strip()
        if not owner:
            raise ValueError("worker_id cannot be empty")
        instant = _now(published_at)
        statement = (
            update(SocialActionRecord)
            .where(
                SocialActionRecord.id == str(action_id),
                SocialActionRecord.status == ActionStatus.PROCESSING.value,
                SocialActionRecord.claim_owner == owner,
            )
            .values(
                status=ActionStatus.PUBLISHED.value,
                external_content_id=external_content_id,
                external_content_url=external_content_url,
                published_at=instant,
                last_error=None,
                retry_available_at=None,
                error_page_url=None,
                error_screenshot_path=None,
                claim_owner=None,
                claim_expires_at=None,
                external_dispatch_started_at=None,
                updated_at=instant,
            )
            .returning(SocialActionRecord)
        )
        async with self._sessions() as session:
            async with session.begin():
                record = (await session.scalars(statement)).one_or_none()
                if record is None:
                    await self._raise_claim_conflict(
                        session, action_id, owner, "publish"
                    )
                session.add(
                    _event(
                        record.id,
                        "published",
                        from_status=ActionStatus.PROCESSING,
                        to_status=ActionStatus.PUBLISHED,
                        page_url=external_content_url,
                        context={"external_content_id": external_content_id},
                        created_at=instant,
                    )
                )
            return _to_action(record)

    async def mark_failed(
        self,
        action_id: UUID | str,
        worker_id: str,
        error: str,
        *,
        retry_at: datetime | None = None,
        page_url: str | None = None,
        screenshot_path: str | None = None,
        context: dict[str, Any] | None = None,
        failed_at: datetime | None = None,
    ) -> SocialAction:
        owner = worker_id.strip()
        if not owner:
            raise ValueError("worker_id cannot be empty")
        if not error.strip():
            raise ValueError("error cannot be empty")
        instant = _now(failed_at)
        retry_instant = (
            _as_utc(retry_at, name="retry_at") if retry_at is not None else None
        )
        retry_value = case(
            (
                SocialActionRecord.attempts < SocialActionRecord.max_attempts,
                retry_instant,
            ),
            else_=None,
        )
        statement = (
            update(SocialActionRecord)
            .where(
                SocialActionRecord.id == str(action_id),
                SocialActionRecord.status == ActionStatus.PROCESSING.value,
                SocialActionRecord.claim_owner == owner,
            )
            .values(
                status=ActionStatus.FAILED.value,
                last_error=error,
                retry_available_at=retry_value,
                error_page_url=page_url,
                error_screenshot_path=screenshot_path,
                claim_owner=None,
                claim_expires_at=None,
                external_dispatch_started_at=None,
                updated_at=instant,
            )
            .returning(SocialActionRecord)
        )
        async with self._sessions() as session:
            async with session.begin():
                record = (await session.scalars(statement)).one_or_none()
                if record is None:
                    await self._raise_claim_conflict(
                        session, action_id, owner, "fail"
                    )
                session.add(
                    _event(
                        record.id,
                        "failed",
                        from_status=ActionStatus.PROCESSING,
                        to_status=ActionStatus.FAILED,
                        message=error,
                        page_url=page_url,
                        screenshot_path=screenshot_path,
                        context=context,
                        created_at=instant,
                    )
                )
            return _to_action(record)

    async def find_published_duplicate(
        self,
        action: SocialAction,
        *,
        window: timedelta,
        now: datetime | None = None,
    ) -> SocialAction | None:
        if window < timedelta(0):
            raise ValueError("window cannot be negative")
        action = _revalidate_action(action)
        instant = _now(now)
        statement = (
            select(SocialActionRecord)
            .where(
                SocialActionRecord.id != str(action.id),
                SocialActionRecord.status == ActionStatus.PUBLISHED.value,
                SocialActionRecord.platform == _platform_value(action.platform),
                SocialActionRecord.account_name == action.account_name,
                SocialActionRecord.fingerprint == action.fingerprint,
                SocialActionRecord.target_scope == target_scope_for(action),
                SocialActionRecord.published_at >= instant - window,
                SocialActionRecord.published_at <= instant,
            )
            .order_by(SocialActionRecord.published_at.desc())
            .limit(1)
        )
        async with self._sessions() as session:
            record = await session.scalar(statement)
            return _to_action(record) if record is not None else None

    async def has_published_duplicate(
        self,
        action: SocialAction,
        *,
        window: timedelta,
        now: datetime | None = None,
    ) -> bool:
        return (
            await self.find_published_duplicate(action, window=window, now=now)
        ) is not None

    async def count_published_since(
        self,
        platform: Platform | str,
        account_name: str,
        since: datetime,
        *,
        until: datetime | None = None,
    ) -> int:
        start = _as_utc(since, name="since")
        statement = select(func.count(SocialActionRecord.id)).where(
            SocialActionRecord.platform == _platform_value(platform),
            SocialActionRecord.account_name == account_name,
            SocialActionRecord.status == ActionStatus.PUBLISHED.value,
            SocialActionRecord.published_at >= start,
        )
        if until is not None:
            statement = statement.where(
                SocialActionRecord.published_at <= _as_utc(until, name="until")
            )
        async with self._sessions() as session:
            return int((await session.scalar(statement)) or 0)

    async def get_rate_usage(
        self,
        platform: Platform | str,
        account_name: str,
        *,
        now: datetime | None = None,
    ) -> RateUsage:
        """Return rolling one-hour/day counts and the latest publication."""
        instant = _now(now)
        hourly_start = instant - timedelta(hours=1)
        daily_start = instant - timedelta(days=1)
        statement = select(
            func.coalesce(
                func.sum(
                    case((SocialActionRecord.published_at >= hourly_start, 1), else_=0)
                ),
                0,
            ),
            func.coalesce(
                func.sum(
                    case((SocialActionRecord.published_at >= daily_start, 1), else_=0)
                ),
                0,
            ),
            func.max(SocialActionRecord.published_at),
        ).where(
            SocialActionRecord.platform == _platform_value(platform),
            SocialActionRecord.account_name == account_name,
            SocialActionRecord.status == ActionStatus.PUBLISHED.value,
            SocialActionRecord.published_at <= instant,
        )
        async with self._sessions() as session:
            row = (await session.execute(statement)).one()
            return RateUsage(
                hourly=int(row[0]),
                daily=int(row[1]),
                last_published_at=row[2],
            )

    async def get_hourly_usage(
        self,
        platform: Platform | str,
        account_name: str,
        *,
        now: datetime | None = None,
    ) -> int:
        return (await self.get_rate_usage(platform, account_name, now=now)).hourly

    async def get_daily_usage(
        self,
        platform: Platform | str,
        account_name: str,
        *,
        now: datetime | None = None,
    ) -> int:
        return (await self.get_rate_usage(platform, account_name, now=now)).daily

    async def min_delay_remaining(
        self,
        platform: Platform | str,
        account_name: str,
        minimum_delay: timedelta,
        *,
        now: datetime | None = None,
    ) -> timedelta:
        if minimum_delay < timedelta(0):
            raise ValueError("minimum_delay cannot be negative")
        instant = _now(now)
        usage = await self.get_rate_usage(platform, account_name, now=instant)
        if usage.last_published_at is None:
            return timedelta(0)
        remaining = minimum_delay - (instant - usage.last_published_at)
        return max(remaining, timedelta(0))

    async def list_events(
        self,
        action_id: UUID | str,
        *,
        limit: int = 100,
    ) -> list[ActionEvent]:
        if limit <= 0:
            raise ValueError("limit must be positive")
        statement = (
            select(ActionEventRecord)
            .where(ActionEventRecord.action_id == str(action_id))
            .order_by(ActionEventRecord.created_at, ActionEventRecord.id)
            .limit(limit)
        )
        async with self._sessions() as session:
            records = (await session.scalars(statement)).all()
            return [_to_event(record) for record in records]

    @staticmethod
    def _due_predicate(instant: datetime) -> Any:
        due_status = or_(
            and_(
                SocialActionRecord.status == ActionStatus.SCHEDULED.value,
                SocialActionRecord.scheduled_at.is_not(None),
                SocialActionRecord.scheduled_at <= instant,
            ),
            and_(
                SocialActionRecord.status == ActionStatus.FAILED.value,
                SocialActionRecord.retry_available_at.is_not(None),
                SocialActionRecord.retry_available_at <= instant,
            ),
        )
        claim_available = or_(
            SocialActionRecord.claim_owner.is_(None),
            SocialActionRecord.claim_expires_at.is_(None),
            SocialActionRecord.claim_expires_at <= instant,
        )
        active_account_pause = (
            select(AccountStateRecord.platform)
            .where(
                AccountStateRecord.platform == SocialActionRecord.platform,
                AccountStateRecord.account_name == SocialActionRecord.account_name,
                AccountStateRecord.paused.is_(True),
                or_(
                    AccountStateRecord.paused_until.is_(None),
                    AccountStateRecord.paused_until > instant,
                ),
            )
            .correlate(SocialActionRecord)
            .exists()
        )
        return and_(
            due_status,
            claim_available,
            SocialActionRecord.external_dispatch_started_at.is_(None),
            ~active_account_pause,
            SocialActionRecord.status != ActionStatus.PUBLISHED.value,
            SocialActionRecord.attempts < SocialActionRecord.max_attempts,
        )

    async def _apply_transition(
        self,
        statement: Any,
        action_id: UUID | str,
        *,
        operation: str,
        event_type: str,
        from_status: ActionStatus | None,
        to_status: ActionStatus,
        event_at: datetime,
        message: str | None = None,
    ) -> SocialAction:
        async with self._sessions() as session:
            async with session.begin():
                record = (await session.scalars(statement)).one_or_none()
                if record is None:
                    await self._raise_missing_or_transition(
                        session, action_id, operation
                    )
                session.add(
                    _event(
                        record.id,
                        event_type,
                        from_status=from_status,
                        to_status=to_status,
                        message=message,
                        created_at=event_at,
                    )
                )
            return _to_action(record)

    @staticmethod
    async def _raise_missing_or_transition(
        session: AsyncSession,
        action_id: UUID | str,
        operation: str,
    ) -> None:
        status = await session.scalar(
            select(SocialActionRecord.status).where(
                SocialActionRecord.id == str(action_id)
            )
        )
        if status is None:
            raise ActionNotFoundError(f"action {action_id} does not exist")
        raise InvalidStateTransitionError(
            f"cannot {operation} action {action_id} while {status}"
        )

    @staticmethod
    async def _raise_claim_conflict(
        session: AsyncSession,
        action_id: UUID | str,
        worker_id: str,
        operation: str,
    ) -> None:
        row = (
            await session.execute(
                select(
                    SocialActionRecord.status,
                    SocialActionRecord.claim_owner,
                ).where(SocialActionRecord.id == str(action_id))
            )
        ).one_or_none()
        if row is None:
            raise ActionNotFoundError(f"action {action_id} does not exist")
        raise ClaimConflictError(
            f"worker {worker_id!r} cannot {operation} action {action_id}; "
            f"status={row.status}, claim_owner={row.claim_owner!r}"
        )


class AccountStateRepository:
    """Persist account pauses and atomically maintain failure counters."""

    def __init__(self, session_factory: SessionFactory) -> None:
        self._sessions = session_factory

    async def get(
        self,
        platform: Platform | str,
        account_name: str,
    ) -> AccountState | None:
        key = (_platform_value(platform), account_name)
        async with self._sessions() as session:
            record = await session.get(AccountStateRecord, key)
            return _to_account_state(record) if record is not None else None

    async def get_or_create(
        self,
        platform: Platform | str,
        account_name: str,
        *,
        now: datetime | None = None,
    ) -> AccountState:
        platform_value = _platform_value(platform)
        instant = _now(now)
        statement = (
            sqlite_insert(AccountStateRecord)
            .values(
                platform=platform_value,
                account_name=account_name,
                created_at=instant,
                updated_at=instant,
            )
            .on_conflict_do_nothing(
                index_elements=[
                    AccountStateRecord.platform,
                    AccountStateRecord.account_name,
                ]
            )
        )
        async with self._sessions() as session:
            async with session.begin():
                await session.execute(statement)
                record = await session.get(
                    AccountStateRecord, (platform_value, account_name)
                )
                if record is None:
                    raise RepositoryError("account state upsert did not produce a row")
            return _to_account_state(record)

    async def acquire_execution_lease(
        self,
        platform: Platform | str,
        account_name: str,
        owner: str,
        lease_duration: timedelta,
        now: datetime | None = None,
    ) -> bool:
        """Atomically create or reserve an account for one execution owner."""
        reservation_owner = owner.strip()
        if not reservation_owner:
            raise ValueError("owner cannot be empty")
        if lease_duration <= timedelta(0):
            raise ValueError("lease_duration must be positive")
        platform_value = _platform_value(platform)
        instant = _now(now)
        lease_expires = instant + lease_duration
        insert_statement = sqlite_insert(AccountStateRecord).values(
            platform=platform_value,
            account_name=account_name,
            execution_owner=reservation_owner,
            execution_expires_at=lease_expires,
            created_at=instant,
            updated_at=instant,
        )
        lease_available = or_(
            AccountStateRecord.execution_owner.is_(None),
            AccountStateRecord.execution_expires_at <= instant,
            AccountStateRecord.execution_owner == reservation_owner,
        )
        statement = (
            insert_statement.on_conflict_do_update(
                index_elements=[
                    AccountStateRecord.platform,
                    AccountStateRecord.account_name,
                ],
                set_={
                    "execution_owner": reservation_owner,
                    "execution_expires_at": lease_expires,
                    "updated_at": instant,
                },
                where=lease_available,
            )
            .returning(AccountStateRecord.platform)
        )
        async with self._sessions() as session:
            async with session.begin():
                return (await session.scalar(statement)) is not None

    async def renew_execution_lease(
        self,
        platform: Platform | str,
        account_name: str,
        owner: str,
        lease_duration: timedelta,
        now: datetime | None = None,
    ) -> bool:
        """Extend an execution lease only while its exact owner still holds it."""
        reservation_owner = owner.strip()
        if not reservation_owner:
            raise ValueError("owner cannot be empty")
        if lease_duration <= timedelta(0):
            raise ValueError("lease_duration must be positive")
        instant = _now(now)
        statement = (
            update(AccountStateRecord)
            .where(
                AccountStateRecord.platform == _platform_value(platform),
                AccountStateRecord.account_name == account_name,
                AccountStateRecord.execution_owner == reservation_owner,
                AccountStateRecord.execution_expires_at.is_not(None),
                AccountStateRecord.execution_expires_at > instant,
            )
            .values(
                execution_expires_at=instant + lease_duration,
                updated_at=instant,
            )
            .returning(AccountStateRecord.platform)
        )
        async with self._sessions() as session:
            async with session.begin():
                return (await session.scalar(statement)) is not None

    async def release_execution_lease(
        self,
        platform: Platform | str,
        account_name: str,
        owner: str,
        now: datetime | None = None,
    ) -> bool:
        """Release an account reservation without disturbing a newer owner."""
        reservation_owner = owner.strip()
        if not reservation_owner:
            raise ValueError("owner cannot be empty")
        instant = _now(now)
        statement = (
            update(AccountStateRecord)
            .where(
                AccountStateRecord.platform == _platform_value(platform),
                AccountStateRecord.account_name == account_name,
                AccountStateRecord.execution_owner == reservation_owner,
            )
            .values(
                execution_owner=None,
                execution_expires_at=None,
                updated_at=instant,
            )
            .returning(AccountStateRecord.platform)
        )
        async with self._sessions() as session:
            async with session.begin():
                return (await session.scalar(statement)) is not None

    async def pause(
        self,
        platform: Platform | str,
        account_name: str,
        *,
        reason: str | None = None,
        until: datetime | None = None,
        now: datetime | None = None,
    ) -> AccountState:
        platform_value = _platform_value(platform)
        instant = _now(now)
        pause_until = _as_utc(until, name="until") if until is not None else None
        if pause_until is not None and pause_until <= instant:
            raise ValueError("until must be later than now")
        insert_statement = sqlite_insert(AccountStateRecord).values(
            platform=platform_value,
            account_name=account_name,
            paused=True,
            paused_at=instant,
            paused_until=pause_until,
            pause_reason=reason,
            created_at=instant,
            updated_at=instant,
        )
        statement = insert_statement.on_conflict_do_update(
            index_elements=[
                AccountStateRecord.platform,
                AccountStateRecord.account_name,
            ],
            set_={
                "paused": True,
                "paused_at": instant,
                "paused_until": pause_until,
                "pause_reason": reason,
                "updated_at": instant,
            },
        ).returning(AccountStateRecord)
        return await self._execute_account_returning(statement)

    async def unpause(
        self,
        platform: Platform | str,
        account_name: str,
        *,
        reset_failures: bool = True,
        now: datetime | None = None,
    ) -> AccountState:
        state = await self.get_or_create(platform, account_name, now=now)
        instant = _now(now)
        values: dict[str, Any] = {
            "paused": False,
            "paused_at": None,
            "paused_until": None,
            "pause_reason": None,
            "updated_at": instant,
        }
        if reset_failures:
            values["consecutive_failures"] = 0
            values["last_error"] = None
        statement = (
            update(AccountStateRecord)
            .where(
                AccountStateRecord.platform == state.platform.value,
                AccountStateRecord.account_name == state.account_name,
            )
            .values(**values)
            .returning(AccountStateRecord)
        )
        return await self._execute_account_returning(statement)

    async def is_paused(
        self,
        platform: Platform | str,
        account_name: str,
        *,
        now: datetime | None = None,
    ) -> bool:
        instant = _now(now)
        statement = select(AccountStateRecord.platform).where(
            AccountStateRecord.platform == _platform_value(platform),
            AccountStateRecord.account_name == account_name,
            AccountStateRecord.paused.is_(True),
            or_(
                AccountStateRecord.paused_until.is_(None),
                AccountStateRecord.paused_until > instant,
            ),
        )
        async with self._sessions() as session:
            return (await session.scalar(statement)) is not None

    async def clear_expired_pauses(
        self,
        *,
        now: datetime | None = None,
    ) -> int:
        instant = _now(now)
        statement = (
            update(AccountStateRecord)
            .where(
                AccountStateRecord.paused.is_(True),
                AccountStateRecord.paused_until.is_not(None),
                AccountStateRecord.paused_until <= instant,
            )
            .values(
                paused=False,
                paused_at=None,
                paused_until=None,
                pause_reason=None,
                updated_at=instant,
            )
            .returning(AccountStateRecord.platform)
        )
        async with self._sessions() as session:
            async with session.begin():
                return len((await session.scalars(statement)).all())

    async def record_failure(
        self,
        platform: Platform | str,
        account_name: str,
        error: str,
        *,
        failure_threshold: int,
        pause_for: timedelta | None = None,
        now: datetime | None = None,
    ) -> AccountState:
        if failure_threshold <= 0:
            raise ValueError("failure_threshold must be positive")
        if pause_for is not None and pause_for <= timedelta(0):
            raise ValueError("pause_for must be positive")
        if not error.strip():
            raise ValueError("error cannot be empty")
        platform_value = _platform_value(platform)
        instant = _now(now)
        pause_until = instant + pause_for if pause_for is not None else None
        initially_paused = failure_threshold == 1
        insert_statement = sqlite_insert(AccountStateRecord).values(
            platform=platform_value,
            account_name=account_name,
            paused=initially_paused,
            paused_at=instant if initially_paused else None,
            paused_until=pause_until if initially_paused else None,
            pause_reason=error if initially_paused else None,
            consecutive_failures=1,
            total_failures=1,
            failure_threshold=failure_threshold,
            last_failure_at=instant,
            last_error=error,
            created_at=instant,
            updated_at=instant,
        )
        next_failures = AccountStateRecord.consecutive_failures + 1
        threshold_reached = next_failures >= failure_threshold
        if pause_until is None:
            weaker_pause = or_(
                AccountStateRecord.paused.is_(False),
                AccountStateRecord.paused_until.is_not(None),
            )
        else:
            weaker_pause = or_(
                AccountStateRecord.paused.is_(False),
                and_(
                    AccountStateRecord.paused_until.is_not(None),
                    AccountStateRecord.paused_until < pause_until,
                ),
            )
        apply_pause = and_(threshold_reached, weaker_pause)
        statement = insert_statement.on_conflict_do_update(
            index_elements=[
                AccountStateRecord.platform,
                AccountStateRecord.account_name,
            ],
            set_={
                "paused": case(
                    (apply_pause, True), else_=AccountStateRecord.paused
                ),
                "paused_at": case(
                    (apply_pause, instant),
                    else_=AccountStateRecord.paused_at,
                ),
                "paused_until": case(
                    (apply_pause, pause_until),
                    else_=AccountStateRecord.paused_until,
                ),
                "pause_reason": case(
                    (apply_pause, error),
                    else_=AccountStateRecord.pause_reason,
                ),
                "consecutive_failures": next_failures,
                "total_failures": AccountStateRecord.total_failures + 1,
                "failure_threshold": failure_threshold,
                "last_failure_at": instant,
                "last_error": error,
                "updated_at": instant,
            },
        ).returning(AccountStateRecord)
        return await self._execute_account_returning(statement)

    async def record_success(
        self,
        platform: Platform | str,
        account_name: str,
        *,
        now: datetime | None = None,
    ) -> AccountState:
        platform_value = _platform_value(platform)
        instant = _now(now)
        insert_statement = sqlite_insert(AccountStateRecord).values(
            platform=platform_value,
            account_name=account_name,
            consecutive_failures=0,
            total_failures=0,
            last_success_at=instant,
            created_at=instant,
            updated_at=instant,
        )
        statement = insert_statement.on_conflict_do_update(
            index_elements=[
                AccountStateRecord.platform,
                AccountStateRecord.account_name,
            ],
            set_={
                "consecutive_failures": 0,
                "last_success_at": instant,
                "last_error": None,
                "updated_at": instant,
            },
        ).returning(AccountStateRecord)
        return await self._execute_account_returning(statement)

    async def list_active_pauses(
        self,
        *,
        now: datetime | None = None,
    ) -> list[AccountState]:
        instant = _now(now)
        statement = (
            select(AccountStateRecord)
            .where(
                AccountStateRecord.paused.is_(True),
                or_(
                    AccountStateRecord.paused_until.is_(None),
                    AccountStateRecord.paused_until > instant,
                ),
            )
            .order_by(AccountStateRecord.platform, AccountStateRecord.account_name)
        )
        async with self._sessions() as session:
            records = (await session.scalars(statement)).all()
            return [_to_account_state(record) for record in records]

    async def _execute_account_returning(self, statement: Any) -> AccountState:
        async with self._sessions() as session:
            async with session.begin():
                record = (await session.scalars(statement)).one_or_none()
                if record is None:
                    raise RepositoryError("account state operation did not return a row")
            return _to_account_state(record)
