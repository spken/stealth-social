"""One async persistence facade for content-generation aggregates."""

from __future__ import annotations

import hashlib
import json
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import and_, func, or_, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from bot.content.models import (
    CandidateApprovalStatus,
    CandidateDecision,
    CandidateDraft,
    DiscoveredTopic,
    GenerationType,
    GenerationStatus,
    PreparedGenerationData,
    RankingResult,
    SanitizedFailure,
    StoredCandidate,
    StoredGenerationRequest,
    ValidationResult,
)
from bot.examples.models import (
    CollectedExample,
    CollectionRunResult,
    CollectionRunStatus,
    ContentExample,
    ExampleCollectionRequest,
    ExampleCollectionRun,
    ExampleListFilters,
    ExampleSelectionFilters,
    ExampleType,
    ExampleUpsertReport,
    SelectedExample,
)
from bot.models import ActionStatus, Platform
from bot.storage.content_models import (
    ContentExampleRecord,
    DiscoveredTopicRecord,
    ExampleCollectionRunRecord,
    GeneratedCandidateRecord,
    GenerationRequestRecord,
)
from bot.storage.models import SocialActionRecord

SessionFactory = async_sessionmaker[AsyncSession]


class ContentRepositoryError(RuntimeError):
    """Base class for content persistence failures."""


class ContentNotFoundError(ContentRepositoryError):
    """Requested content record does not exist."""


def _utc(value: datetime | None = None) -> datetime:
    value = value or datetime.now(UTC)
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("content repository timestamps must be timezone-aware")
    return value.astimezone(UTC)


def _json(value: Any, *, default: Any = None) -> Any:
    candidate = default if value is None else value
    try:
        return json.loads(json.dumps(candidate, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError("content persistence values must be JSON serializable") from error


def _example_hash(example: CollectedExample) -> str:
    parts = (
        example.platform.value,
        example.content_type.value,
        " ".join((example.title or "").split()).casefold(),
        " ".join(example.body.split()).casefold(),
        " ".join((example.parent_text or "").split()).casefold(),
    )
    return hashlib.sha256(json.dumps(parts, ensure_ascii=False).encode()).hexdigest()


def _selected_json(item: SelectedExample) -> dict[str, Any]:
    return {
        "example_id": str(item.example_id),
        "score": item.score,
        "component_scores": _json(item.component_scores, default={}),
        "selection_reason": item.selection_reason,
    }


def _selected_from_json(value: Any, example: ContentExample) -> SelectedExample:
    if not isinstance(value, dict):
        raise ValueError("stored selected example is malformed")
    return SelectedExample(
        example_id=UUID(str(value["example_id"])),
        score=float(value["score"]),
        component_scores=dict(value.get("component_scores") or {}),
        selection_reason=str(value.get("selection_reason") or "stored selection"),
        example=example,
    )


async def _hydrate_selected_examples(
    session: AsyncSession,
    record: GenerationRequestRecord,
) -> tuple[SelectedExample, ...]:
    """Load selected example text from its source rows, never from request JSON."""

    stored = record.selected_examples_json or ()
    ids = [
        str(item["example_id"])
        for item in stored
        if isinstance(item, dict) and item.get("example_id")
    ]
    if not ids:
        return ()
    rows = (
        await session.scalars(
            select(ContentExampleRecord).where(ContentExampleRecord.id.in_(ids))
        )
    ).all()
    examples = {row.id: _to_example(row) for row in rows}
    missing = [example_id for example_id in ids if example_id not in examples]
    if missing:
        raise ContentNotFoundError(
            "selected example rows were not found: " + ", ".join(missing[:8])
        )
    selected: list[SelectedExample] = []
    for item in stored:
        if not isinstance(item, dict):
            raise ValueError("stored selected example is malformed")
        example = examples.get(str(item.get("example_id")))
        if example is None:
            raise ContentNotFoundError("selected example row was not found")
        selected.append(_selected_from_json(item, example))
    return tuple(selected)


def _to_run(record: ExampleCollectionRunRecord) -> ExampleCollectionRun:
    return ExampleCollectionRun(
        id=UUID(record.id),
        platform=Platform(record.platform),
        started_at=record.started_at,
        finished_at=record.finished_at,
        status=CollectionRunStatus(record.status),
        request_snapshot=_json(record.request_json, default={}),
        collected_count=record.collected_count,
        rejected_count=record.rejected_count,
        duplicate_count=record.duplicate_count,
        disabled_count=record.disabled_count,
        error_type=record.error_type,
        error_message=record.error_message,
        retry_after_seconds=record.retry_after_seconds,
    )


def _to_example(record: ContentExampleRecord) -> ContentExample:
    return ContentExample(
        id=UUID(record.id),
        collection_run_id=UUID(record.collection_run_id)
        if record.collection_run_id
        else None,
        platform=Platform(record.platform),
        content_type=ExampleType(record.content_type),
        external_id=record.external_id,
        source_url=record.source_url,
        author_identifier=record.author_identifier,
        title=record.title,
        body=record.body,
        parent_text=record.parent_text,
        subreddit=record.subreddit,
        published_at=record.published_at,
        collected_at=record.collected_at,
        expires_at=record.expires_at,
        content_hash=record.content_hash,
        engagement_score=record.engagement_score,
        metadata=_json(record.metadata_json, default={}),
        topic_tags=tuple(record.topic_tags_json or ()),
        is_own_content=record.is_own_content,
        generated=record.generated,
        is_active=record.is_active,
        is_quarantined=record.is_quarantined,
        injection_findings=tuple(record.injection_findings_json or ()),
    )


def _to_request(
    record: GenerationRequestRecord,
    *,
    selected_examples: Sequence[SelectedExample] = (),
) -> StoredGenerationRequest:
    failure = (
        SanitizedFailure.model_validate(record.failure_json)
        if record.failure_json is not None
        else None
    )
    return StoredGenerationRequest(
        id=UUID(record.id),
        generation_type=GenerationType(record.generation_type),
        platform=Platform(record.platform),
        account_name=record.account_name,
        campaign_id=record.campaign_id,
        status=GenerationStatus(record.status),
        request_snapshot=_json(record.request_json, default={}),
        resolved_profile=_json(record.resolved_profile_json, default={}),
        target_snapshot=_json(record.target_snapshot_json),
        selected_examples=tuple(selected_examples),
        prompt_version=record.prompt_version,
        schema_version=record.schema_version,
        desired_generation_time=record.desired_generation_time,
        claim_owner=record.claim_owner,
        claim_expires_at=record.claim_expires_at,
        attempt_count=record.attempt_count,
        next_retry_at=record.next_retry_at,
        completed_at=record.completed_at,
        failure=failure,
        metadata=_json(record.metadata_json, default={}),
        unattended_approval_requested=record.unattended_approval_requested,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )


def _to_candidate(record: GeneratedCandidateRecord) -> StoredCandidate:
    ranking = (
        RankingResult.model_validate(record.ranking_json)
        if record.ranking_json is not None
        else None
    )
    decision = (
        CandidateDecision.model_validate(record.decision_json)
        if record.decision_json is not None
        else None
    )
    return StoredCandidate(
        id=UUID(record.id),
        request_id=UUID(record.request_id),
        ordinal=record.ordinal,
        revision_of_candidate_id=(
            UUID(record.revision_of_candidate_id)
            if record.revision_of_candidate_id
            else None
        ),
        draft=CandidateDraft(
            title=record.title,
            body=record.body,
            strategy=record.strategy,
            used_example_ids=tuple(UUID(item) for item in (record.used_example_ids_json or ())),
        ),
        model_name=record.model_name,
        generation_parameters=_json(record.generation_parameters_json, default={}),
        generated_at=record.generated_at,
        validation=ValidationResult.model_validate(record.validation_json),
        ranking=ranking,
        approval_status=CandidateApprovalStatus(record.approval_status),
        decision=decision,
        social_action_id=UUID(record.social_action_id) if record.social_action_id else None,
        metadata=_json(record.metadata_json, default={}),
    )


def _to_topic(record: DiscoveredTopicRecord) -> DiscoveredTopic:
    return DiscoveredTopic(
        id=UUID(record.id),
        platform=Platform(record.platform),
        label=record.label,
        keywords=tuple(record.keywords_json or ()),
        supporting_example_ids=tuple(UUID(item) for item in (record.supporting_example_ids_json or ())),
        support_count=record.support_count,
        distinct_source_count=record.distinct_source_count,
        median_recency=record.median_recency,
        discovered_at=record.discovered_at,
        expires_at=record.expires_at,
        is_active=record.is_active,
        score=record.score,
    )


class ContentRepository:
    """Persist content aggregates and the scheduled-generation queue."""

    def __init__(self, session_factory: SessionFactory) -> None:
        self._sessions = session_factory

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[AsyncSession]:
        """Yield one caller-owned transaction for atomic generation writes."""

        async with self._sessions() as session:
            async with session.begin():
                yield session

    async def start_collection_run(
        self, request: ExampleCollectionRequest
    ) -> ExampleCollectionRun:
        run = ExampleCollectionRun(
            id=request.id,
            platform=request.platform,
            request_snapshot=request.model_dump(mode="json"),
        )
        record = ExampleCollectionRunRecord(
            id=str(run.id),
            platform=run.platform.value,
            started_at=run.started_at,
            status=run.status.value,
            request_json=_json(run.request_snapshot, default={}),
        )
        async with self._sessions() as session:
            async with session.begin():
                session.add(record)
                await session.flush()
        return run

    async def finish_collection_run(
        self, run_id: UUID, result: CollectionRunResult
    ) -> ExampleCollectionRun:
        async with self._sessions() as session:
            async with session.begin():
                record = await session.get(ExampleCollectionRunRecord, str(run_id))
                if record is None:
                    raise ContentNotFoundError(f"collection run {run_id} was not found")
                record.finished_at = _utc(result.finished_at)
                record.status = result.status.value
                record.collected_count = result.collected_count
                record.rejected_count = result.rejected_count
                record.duplicate_count = result.duplicate_count
                record.disabled_count = result.disabled_count
                record.error_type = result.error_type
                record.error_message = result.error_message
                record.retry_after_seconds = result.retry_after_seconds
            return _to_run(record)

    async def latest_collection_run(
        self,
        platform: Platform,
        *,
        account_name: str,
    ) -> ExampleCollectionRun | None:
        """Return the latest completed or partial run for one configured account."""

        statement = (
            select(ExampleCollectionRunRecord)
            .where(
                ExampleCollectionRunRecord.platform == platform.value,
                ExampleCollectionRunRecord.status.in_(
                    [
                        CollectionRunStatus.COMPLETED.value,
                        CollectionRunStatus.PARTIAL.value,
                    ]
                ),
            )
            .order_by(ExampleCollectionRunRecord.finished_at.desc())
        )
        async with self._sessions() as session:
            records = (await session.scalars(statement)).all()
            for record in records:
                request = _json(record.request_json, default={})
                if isinstance(request, dict) and request.get("account_name") == account_name:
                    return _to_run(record)
        return None

    async def upsert_examples(
        self, run_id: UUID, examples: Sequence[CollectedExample]
    ) -> ExampleUpsertReport:
        inserted = refreshed = duplicates = rejected = 0
        async with self._sessions() as session:
            async with session.begin():
                if await session.get(ExampleCollectionRunRecord, str(run_id)) is None:
                    raise ContentNotFoundError(f"collection run {run_id} was not found")
                for example in examples:
                    content_hash = example.content_hash or _example_hash(example)
                    conditions = [
                        ContentExampleRecord.platform == example.platform.value,
                        ContentExampleRecord.content_type == example.content_type.value,
                        ContentExampleRecord.content_hash == content_hash,
                    ]
                    record = (
                        await session.scalars(
                            select(ContentExampleRecord).where(and_(*conditions)).limit(1)
                        )
                    ).one_or_none()
                    if record is None and example.external_id:
                        record = (
                            await session.scalars(
                                select(ContentExampleRecord).where(
                                    ContentExampleRecord.platform == example.platform.value,
                                    ContentExampleRecord.external_id == example.external_id,
                                )
                            )
                        ).one_or_none()
                    if record is not None:
                        high_risk = any(
                            isinstance(finding, dict)
                            and finding.get("severity") == "high"
                            for finding in example.injection_findings
                        )
                        existing_metadata = _json(record.metadata_json, default={})
                        incoming_metadata = _json(example.metadata, default={})
                        if existing_metadata.get("disabled_at"):
                            incoming_metadata["disabled_at"] = existing_metadata["disabled_at"]
                        record.engagement_score = example.engagement_score
                        record.collected_at = _utc(example.collected_at)
                        record.expires_at = example.expires_at
                        record.published_at = example.published_at
                        record.metadata_json = incoming_metadata
                        record.topic_tags_json = list(example.topic_tags)
                        record.injection_findings_json = list(example.injection_findings)
                        if high_risk:
                            record.is_active = False
                            record.is_quarantined = True
                        elif not existing_metadata.get("disabled_at"):
                            record.is_active = True
                            record.is_quarantined = False
                        refreshed += 1
                        duplicates += 1
                        continue
                    high_risk = any(
                        isinstance(finding, dict)
                        and finding.get("severity") == "high"
                        for finding in example.injection_findings
                    )
                    try:
                        async with session.begin_nested():
                            session.add(
                                ContentExampleRecord(
                                    id=str(uuid4()),
                                    collection_run_id=str(run_id),
                                    platform=example.platform.value,
                                    content_type=example.content_type.value,
                                    external_id=example.external_id,
                                    source_url=example.source_url,
                                    author_identifier=example.author_identifier,
                                    title=example.title,
                                    body=example.body,
                                    parent_text=example.parent_text,
                                    subreddit=example.subreddit,
                                    published_at=example.published_at,
                                    collected_at=_utc(example.collected_at),
                                    expires_at=example.expires_at,
                                    content_hash=content_hash,
                                    engagement_score=example.engagement_score,
                                    metadata_json=_json(example.metadata, default={}),
                                    topic_tags_json=list(example.topic_tags),
                                    is_own_content=example.is_own_content,
                                    generated=example.generated,
                                    is_active=not high_risk,
                                    is_quarantined=high_risk,
                                    injection_findings_json=list(example.injection_findings),
                                )
                            )
                            await session.flush()
                    except IntegrityError:
                        rejected += 1
                        continue
                    inserted += 1
        return ExampleUpsertReport(
            inserted_count=inserted,
            refreshed_count=refreshed,
            duplicate_count=duplicates,
            rejected_count=rejected,
        )

    async def disable_expired_examples(
        self, platform: Platform, *, disabled_at: datetime
    ) -> int:
        """Disable expired examples after a successful replacement pass."""

        instant = _utc(disabled_at)
        async with self._sessions() as session:
            async with session.begin():
                records = (
                    await session.scalars(
                        select(ContentExampleRecord).where(
                            ContentExampleRecord.platform == platform.value,
                            ContentExampleRecord.is_active.is_(True),
                            ContentExampleRecord.expires_at.is_not(None),
                            ContentExampleRecord.expires_at <= instant,
                        )
                    )
                ).all()
                for record in records:
                    record.is_active = False
                    metadata = _json(record.metadata_json, default={})
                    metadata["disabled_at"] = instant.isoformat()
                    record.metadata_json = metadata
                return len(records)

    async def get_example(self, example_id: UUID) -> ContentExample | None:
        async with self._sessions() as session:
            record = await session.get(ContentExampleRecord, str(example_id))
            return _to_example(record) if record is not None else None

    async def list_examples(self, filters: ExampleListFilters) -> list[ContentExample]:
        statement = select(ContentExampleRecord)
        if filters.platform is not None:
            statement = statement.where(
                ContentExampleRecord.platform == filters.platform.value
            )
        if filters.content_type is not None:
            statement = statement.where(
                ContentExampleRecord.content_type == filters.content_type.value
            )
        if filters.subreddit is not None:
            statement = statement.where(
                func.lower(ContentExampleRecord.subreddit)
                == filters.subreddit.casefold()
            )
        if filters.active_only:
            now = _utc()
            statement = statement.where(
                ContentExampleRecord.is_active.is_(True),
                or_(
                    ContentExampleRecord.expires_at.is_(None),
                    ContentExampleRecord.expires_at > now,
                ),
            )
        statement = statement.order_by(
            ContentExampleRecord.collected_at.desc(), ContentExampleRecord.id
        ).limit(filters.limit).offset(filters.offset)
        async with self._sessions() as session:
            records = (await session.scalars(statement)).all()
            return [_to_example(record) for record in records]

    async def disable_example(
        self, example_id: UUID, *, disabled_at: datetime
    ) -> ContentExample:
        async with self._sessions() as session:
            async with session.begin():
                record = await session.get(ContentExampleRecord, str(example_id))
                if record is None:
                    raise ContentNotFoundError(f"example {example_id} was not found")
                record.is_active = False
                metadata = _json(record.metadata_json, default={})
                metadata["disabled_at"] = _utc(disabled_at).isoformat()
                record.metadata_json = metadata
            return _to_example(record)

    async def list_selection_pool(
        self, filters: ExampleSelectionFilters
    ) -> list[ContentExample]:
        now = _utc()
        statement = select(ContentExampleRecord).where(
            ContentExampleRecord.platform == filters.platform.value,
            ContentExampleRecord.is_active.is_(True),
            ContentExampleRecord.is_quarantined.is_(False),
            or_(
                ContentExampleRecord.expires_at.is_(None),
                ContentExampleRecord.expires_at > now,
            ),
        )
        if not filters.allow_generated:
            statement = statement.where(ContentExampleRecord.generated.is_(False))
        if filters.content_type is not None:
            statement = statement.where(
                ContentExampleRecord.content_type == filters.content_type.value
            )
        if filters.compatible_types:
            statement = statement.where(
                ContentExampleRecord.content_type.in_(
                    [item.value for item in filters.compatible_types]
                )
            )
        if filters.subreddit is not None:
            statement = statement.where(
                func.lower(ContentExampleRecord.subreddit)
                == filters.subreddit.casefold()
            )
        if filters.allowed_subreddits:
            statement = statement.where(
                func.lower(ContentExampleRecord.subreddit).in_(
                    [item.casefold() for item in filters.allowed_subreddits]
                )
            )
        statement = statement.order_by(
            ContentExampleRecord.collected_at.desc(), ContentExampleRecord.id
        )
        async with self._sessions() as session:
            records = (await session.scalars(statement)).all()
            return [_to_example(record) for record in records]

    async def create_generation_request(
        self, request: StoredGenerationRequest
    ) -> StoredGenerationRequest:
        if request.status is GenerationStatus.PROCESSING:
            if (
                not request.claim_owner
                or request.claim_expires_at is None
                or request.claim_expires_at <= request.updated_at
            ):
                raise ContentRepositoryError(
                    "processing generation requests require an unexpired owner lease"
                )
        record = GenerationRequestRecord(
            id=str(request.id),
            generation_type=request.generation_type.value,
            platform=request.platform.value,
            account_name=request.account_name,
            campaign_id=request.campaign_id,
            status=request.status.value,
            request_json=_json(request.request_snapshot, default={}),
            resolved_profile_json=_json(request.resolved_profile, default={}),
            target_snapshot_json=_json(request.target_snapshot),
            selected_examples_json=[
                _selected_json(item) for item in request.selected_examples
            ],
            prompt_version=request.prompt_version,
            schema_version=request.schema_version,
            desired_generation_time=request.desired_generation_time,
            claim_owner=request.claim_owner,
            claim_expires_at=request.claim_expires_at,
            attempt_count=request.attempt_count,
            next_retry_at=request.next_retry_at,
            completed_at=request.completed_at,
            failure_json=(request.failure.model_dump(mode="json") if request.failure else None),
            metadata_json=_json(request.metadata, default={}),
            unattended_approval_requested=request.unattended_approval_requested,
            created_at=request.created_at,
            updated_at=request.updated_at,
        )
        try:
            async with self._sessions() as session:
                async with session.begin():
                    session.add(record)
                    await session.flush()
        except IntegrityError as error:
            raise ContentRepositoryError(
                f"generation request {request.id} already exists"
            ) from error
        return request

    async def list_generation_requests_for_campaign(
        self,
        campaign_id: str,
        *,
        limit: int = 1000,
    ) -> list[StoredGenerationRequest]:
        """Return recent requests for one exact autopost campaign."""

        if not campaign_id.strip():
            raise ValueError("campaign_id cannot be empty")
        if limit <= 0:
            raise ValueError("limit must be positive")
        statement = (
            select(GenerationRequestRecord)
            .where(GenerationRequestRecord.campaign_id == campaign_id)
            .order_by(
                GenerationRequestRecord.created_at.desc(),
                GenerationRequestRecord.id.desc(),
            )
            .limit(limit)
        )
        async with self._sessions() as session:
            records = (await session.scalars(statement)).all()
            result: list[StoredGenerationRequest] = []
            for record in records:
                selected = await _hydrate_selected_examples(session, record)
                result.append(_to_request(record, selected_examples=selected))
            return result

    async def get_approved_candidate_for_request(
        self,
        request_id: UUID,
    ) -> StoredCandidate | None:
        """Return the authoritative approved candidate for one request."""

        statement = (
            select(GeneratedCandidateRecord)
            .where(
                GeneratedCandidateRecord.request_id == str(request_id),
                GeneratedCandidateRecord.approval_status
                == CandidateApprovalStatus.APPROVED.value,
            )
            .limit(1)
        )
        async with self._sessions() as session:
            record = (await session.scalars(statement)).one_or_none()
            return _to_candidate(record) if record is not None else None

    async def reclaim_expired_generation_request(
        self,
        request_id: UUID,
        *,
        owner: str,
        lease_duration: timedelta,
        now: datetime,
    ) -> StoredGenerationRequest | None:
        """Adopt one expired processing claim with a guarded update."""

        claim_owner = owner.strip()
        if not claim_owner:
            raise ValueError("owner cannot be empty")
        if lease_duration <= timedelta(0):
            raise ValueError("lease_duration must be positive")
        instant = _utc(now)
        expires_at = instant + lease_duration
        statement = (
            update(GenerationRequestRecord)
            .where(
                GenerationRequestRecord.id == str(request_id),
                GenerationRequestRecord.status == GenerationStatus.PROCESSING.value,
                GenerationRequestRecord.claim_expires_at.is_not(None),
                GenerationRequestRecord.claim_expires_at <= instant,
            )
            .values(
                claim_owner=claim_owner,
                claim_expires_at=expires_at,
                attempt_count=GenerationRequestRecord.attempt_count + 1,
                next_retry_at=None,
                updated_at=instant,
            )
            .returning(GenerationRequestRecord)
        )
        async with self._sessions() as session:
            async with session.begin():
                record = (await session.scalars(statement)).one_or_none()
                if record is None:
                    return None
                selected = await _hydrate_selected_examples(session, record)
            return _to_request(record, selected_examples=selected)

    async def get_generation_request(
        self, request_id: UUID
    ) -> StoredGenerationRequest | None:
        async with self._sessions() as session:
            return await self.get_generation_request_in_session(session, request_id)

    async def get_generation_request_in_session(
        self, session: AsyncSession, request_id: UUID
    ) -> StoredGenerationRequest | None:
        record = await session.get(GenerationRequestRecord, str(request_id))
        if record is None:
            return None
        selected = await _hydrate_selected_examples(session, record)
        return _to_request(record, selected_examples=selected)

    async def prepare_generation_request(
        self,
        request_id: UUID,
        prepared: PreparedGenerationData,
        *,
        owner: str | None = None,
        now: datetime | None = None,
    ) -> StoredGenerationRequest:
        async with self._sessions() as session:
            async with session.begin():
                record = await session.get(GenerationRequestRecord, str(request_id))
                if record is None:
                    raise ContentNotFoundError(f"generation request {request_id} was not found")
                if owner is not None:
                    await self.ensure_active_generation_claim_in_session(
                        session,
                        request_id,
                        owner,
                        now=now or _utc(),
                    )
                record.target_snapshot_json = (
                    prepared.target_context.model_dump(mode="json")
                    if prepared.target_context is not None
                    else None
                )
                record.selected_examples_json = [
                    _selected_json(item) for item in prepared.selected_examples
                ]
                record.request_json = _json(prepared.request_snapshot, default={})
                record.resolved_profile_json = _json(prepared.resolved_profile, default={})
                record.prompt_version = prepared.prompt_version
                record.schema_version = prepared.schema_version
                metadata = _json(record.metadata_json, default={})
                metadata["prepared"] = True
                metadata["selection_outcome"] = (
                    "examples_selected"
                    if prepared.selected_examples
                    else "no_examples_selected"
                )
                record.metadata_json = metadata
                await session.flush()
                selected = await _hydrate_selected_examples(session, record)
            return _to_request(record, selected_examples=selected)

    async def schedule_generation_request(
        self, request_id: UUID, when: datetime
    ) -> StoredGenerationRequest:
        instant = _utc(when)
        async with self._sessions() as session:
            async with session.begin():
                record = await session.get(GenerationRequestRecord, str(request_id))
                if record is None:
                    raise ContentNotFoundError(f"generation request {request_id} was not found")
                record.status = GenerationStatus.SCHEDULED.value
                record.desired_generation_time = instant
                record.next_retry_at = None
                record.failure_json = None
                record.claim_owner = None
                record.claim_expires_at = None
                selected = await _hydrate_selected_examples(session, record)
            return _to_request(record, selected_examples=selected)

    async def restart_generation_request(
        self,
        request_id: UUID,
        *,
        started_at: datetime,
        owner: str | None = None,
        lease_duration: timedelta = timedelta(minutes=5),
    ) -> StoredGenerationRequest:
        instant = _utc(started_at)
        claim_owner = owner.strip() if owner is not None else f"foreground-{uuid4().hex}"
        if not claim_owner:
            raise ValueError("generation request owner cannot be empty")
        if lease_duration <= timedelta(0):
            raise ValueError("lease_duration must be positive")
        async with self._sessions() as session:
            async with session.begin():
                record = await session.get(GenerationRequestRecord, str(request_id))
                if record is None:
                    raise ContentNotFoundError(f"generation request {request_id} was not found")
                if record.status not in {
                    GenerationStatus.COMPLETED.value,
                    GenerationStatus.FAILED.value,
                }:
                    raise ContentRepositoryError(
                        f"generation request {request_id} cannot be regenerated from {record.status}"
                    )
                metadata = _json(record.metadata_json, default={})
                history: list[dict[str, Any]] = [
                    item
                    for item in (metadata.get("attempt_history") or ())
                    if isinstance(item, dict)
                ]
                history.append({"attempt": record.attempt_count, "metadata": metadata})
                metadata["attempt_history"] = history[-10:]
                metadata["regeneration"] = True
                record.metadata_json = metadata
                record.status = GenerationStatus.PROCESSING.value
                record.attempt_count += 1
                record.failure_json = None
                record.next_retry_at = None
                record.completed_at = None
                record.claim_owner = claim_owner
                record.claim_expires_at = instant + lease_duration
                record.updated_at = instant
                selected = await _hydrate_selected_examples(session, record)
            return _to_request(record, selected_examples=selected)

    async def claim_foreground_generation_request(
        self,
        request_id: UUID,
        *,
        owner: str,
        lease_duration: timedelta,
        now: datetime,
    ) -> StoredGenerationRequest:
        """Adopt an unclaimed processing row using the recoverable lease protocol."""

        if not owner.strip() or lease_duration <= timedelta(0):
            raise ValueError("owner and lease_duration must be valid")
        instant = _utc(now)
        async with self._sessions() as session:
            async with session.begin():
                statement = (
                    update(GenerationRequestRecord)
                    .where(
                        GenerationRequestRecord.id == str(request_id),
                        GenerationRequestRecord.status
                        == GenerationStatus.PROCESSING.value,
                        GenerationRequestRecord.claim_owner.is_(None),
                        GenerationRequestRecord.claim_expires_at.is_(None),
                    )
                    .values(
                        claim_owner=owner.strip(),
                        claim_expires_at=instant + lease_duration,
                        updated_at=instant,
                    )
                    .returning(GenerationRequestRecord.id)
                )
                claimed_id = (await session.scalars(statement)).one_or_none()
                if claimed_id is None:
                    record = await session.get(GenerationRequestRecord, str(request_id))
                    if record is None:
                        raise ContentNotFoundError(
                            f"generation request {request_id} was not found"
                        )
                    raise ContentRepositoryError(
                        f"generation request {request_id} is already owned or not processing"
                    )
                record = await session.get(GenerationRequestRecord, claimed_id)
                if record is None:
                    raise ContentRepositoryError("foreground generation claim disappeared")
                selected = await _hydrate_selected_examples(session, record)
            return _to_request(record, selected_examples=selected)

    async def claim_due_generation_requests(
        self,
        owner: str,
        *,
        limit: int,
        lease_duration: timedelta,
        now: datetime,
    ) -> list[StoredGenerationRequest]:
        if not owner.strip() or limit <= 0 or lease_duration <= timedelta(0):
            raise ValueError("owner, limit, and lease_duration must be valid")
        instant = _utc(now)
        expiry = instant + lease_duration
        async with self._sessions() as session:
            async with session.begin():
                scheduled = and_(
                    GenerationRequestRecord.status == GenerationStatus.SCHEDULED.value,
                    GenerationRequestRecord.claim_owner.is_(None),
                    GenerationRequestRecord.claim_expires_at.is_(None),
                )
                retryable_failure = and_(
                    GenerationRequestRecord.status == GenerationStatus.FAILED.value,
                    GenerationRequestRecord.next_retry_at.is_not(None),
                    GenerationRequestRecord.next_retry_at <= instant,
                    GenerationRequestRecord.claim_owner.is_(None),
                    GenerationRequestRecord.claim_expires_at.is_(None),
                )
                expired_processing = and_(
                    GenerationRequestRecord.status == GenerationStatus.PROCESSING.value,
                    GenerationRequestRecord.claim_owner.is_not(None),
                    GenerationRequestRecord.claim_expires_at.is_not(None),
                    GenerationRequestRecord.claim_expires_at <= instant,
                )
                desired = or_(
                    GenerationRequestRecord.desired_generation_time.is_(None),
                    GenerationRequestRecord.desired_generation_time <= instant,
                )
                eligible_ids = (
                    select(GenerationRequestRecord.id)
                    .where(or_(scheduled, retryable_failure, expired_processing), desired)
                    .order_by(
                        GenerationRequestRecord.desired_generation_time,
                        GenerationRequestRecord.created_at,
                        GenerationRequestRecord.id,
                    )
                    .limit(limit)
                )
                claim_statement = (
                    update(GenerationRequestRecord)
                    .where(
                        GenerationRequestRecord.id.in_(eligible_ids),
                        or_(scheduled, retryable_failure, expired_processing),
                        desired,
                    )
                    .values(
                        status=GenerationStatus.PROCESSING.value,
                        claim_owner=owner.strip(),
                        claim_expires_at=expiry,
                        next_retry_at=None,
                        attempt_count=GenerationRequestRecord.attempt_count + 1,
                        updated_at=instant,
                    )
                    .returning(GenerationRequestRecord.id)
                )
                claimed_ids = list((await session.scalars(claim_statement)).all())
                if not claimed_ids:
                    return []
                records = list(
                    (
                        await session.scalars(
                            select(GenerationRequestRecord).where(
                                GenerationRequestRecord.id.in_(claimed_ids)
                            )
                        )
                    ).all()
                )
                records_by_id = {record.id: record for record in records}
                records = [records_by_id[record_id] for record_id in claimed_ids]
                hydrated = [
                    await _hydrate_selected_examples(session, record)
                    for record in records
                ]
            return [
                _to_request(record, selected_examples=selected)
                for record, selected in zip(records, hydrated, strict=True)
            ]

    async def ensure_active_generation_claim(
        self,
        request_id: UUID,
        owner: str,
        *,
        now: datetime,
    ) -> StoredGenerationRequest:
        """Verify a live generation lease before invoking external work."""

        async with self._sessions() as session:
            async with session.begin():
                await self.ensure_active_generation_claim_in_session(
                    session,
                    request_id,
                    owner,
                    now=now,
                )
                record = await session.get(GenerationRequestRecord, str(request_id))
                if record is None:
                    raise ContentNotFoundError(
                        f"generation request {request_id} was not found"
                    )
                selected = await _hydrate_selected_examples(session, record)
            return _to_request(record, selected_examples=selected)

    async def complete_generation_request(
        self,
        request_id: UUID,
        *,
        metadata: dict[str, Any],
        completed_at: datetime,
        owner: str | None = None,
    ) -> StoredGenerationRequest:
        async with self._sessions() as session:
            async with session.begin():
                return await self.complete_generation_request_in_session(
                    session,
                    request_id,
                    metadata=metadata,
                    completed_at=completed_at,
                    owner=owner,
                )

    async def complete_generation_request_in_session(
        self,
        session: AsyncSession,
        request_id: UUID,
        *,
        metadata: dict[str, Any],
        completed_at: datetime,
        owner: str | None = None,
    ) -> StoredGenerationRequest:
        """Finalize only the still-owned, unexpired generation attempt."""

        instant = _utc(completed_at)
        conditions = [GenerationRequestRecord.id == str(request_id)]
        if owner is not None:
            conditions.extend(
                [
                    GenerationRequestRecord.status
                    == GenerationStatus.PROCESSING.value,
                    GenerationRequestRecord.claim_owner == owner,
                    GenerationRequestRecord.claim_expires_at > instant,
                ]
            )
        statement = (
            update(GenerationRequestRecord)
            .where(*conditions)
            .values(
                status=GenerationStatus.COMPLETED.value,
                metadata_json=_json(metadata, default={}),
                completed_at=instant,
                next_retry_at=None,
                failure_json=None,
                claim_owner=None,
                claim_expires_at=None,
                updated_at=instant,
            )
            .returning(GenerationRequestRecord.id)
        )
        record_id = (await session.scalars(statement)).one_or_none()
        if record_id is None:
            record = await session.get(GenerationRequestRecord, str(request_id))
            if record is None:
                raise ContentNotFoundError(
                    f"generation request {request_id} was not found"
                )
            if owner is not None:
                raise ContentRepositoryError(
                    f"generation request {request_id} is no longer actively claimed by {owner}"
                )
            raise ContentRepositoryError(
                f"generation request {request_id} cannot be completed from {record.status}"
            )
        record = await session.get(GenerationRequestRecord, record_id)
        if record is None:
            raise ContentRepositoryError("completed generation request disappeared")
        selected = await _hydrate_selected_examples(session, record)
        return _to_request(record, selected_examples=selected)

    async def fail_generation_request(
        self,
        request_id: UUID,
        *,
        failure: SanitizedFailure,
        retry_at: datetime | None,
        failed_at: datetime,
        owner: str | None = None,
    ) -> StoredGenerationRequest:
        instant = _utc(failed_at)
        retry = _utc(retry_at) if retry_at is not None else None
        async with self._sessions() as session:
            async with session.begin():
                record = await session.get(GenerationRequestRecord, str(request_id))
                if record is None:
                    raise ContentNotFoundError(f"generation request {request_id} was not found")
                if owner is not None:
                    await self.ensure_active_generation_claim_in_session(
                        session,
                        request_id,
                        owner,
                        now=instant,
                    )
                record.status = GenerationStatus.FAILED.value
                record.failure_json = failure.model_dump(mode="json")
                record.next_retry_at = retry
                record.completed_at = None
                record.claim_owner = None
                record.claim_expires_at = None
                record.updated_at = instant
                selected = await _hydrate_selected_examples(session, record)
            return _to_request(record, selected_examples=selected)

    async def release_generation_claim(
        self,
        request_id: UUID,
        owner: str,
        *,
        now: datetime,
    ) -> bool:
        """Release one owned claim without declaring the request successful."""

        instant = _utc(now)
        async with self._sessions() as session:
            async with session.begin():
                result = await session.execute(
                    update(GenerationRequestRecord)
                    .where(
                        GenerationRequestRecord.id == str(request_id),
                        GenerationRequestRecord.status == GenerationStatus.PROCESSING.value,
                        GenerationRequestRecord.claim_owner == owner,
                    )
                    .values(
                        status=GenerationStatus.SCHEDULED.value,
                        claim_owner=None,
                        claim_expires_at=None,
                        updated_at=instant,
                    )
                )
                return getattr(result, "rowcount", 0) == 1

    async def reschedule_failed_generation_request(
        self,
        request_id: UUID,
        *,
        attempt_count: int,
        failure: SanitizedFailure,
        retry_at: datetime,
        failed_at: datetime,
    ) -> bool:
        """Attach a bounded retry only to the failed attempt that produced it."""

        instant = _utc(failed_at)
        retry = _utc(retry_at)
        async with self._sessions() as session:
            async with session.begin():
                result = await session.execute(
                    update(GenerationRequestRecord)
                    .where(
                        GenerationRequestRecord.id == str(request_id),
                        GenerationRequestRecord.status == GenerationStatus.FAILED.value,
                        GenerationRequestRecord.claim_owner.is_(None),
                        GenerationRequestRecord.attempt_count == attempt_count,
                    )
                    .values(
                        failure_json=failure.model_dump(mode="json"),
                        next_retry_at=retry,
                        updated_at=instant,
                    )
                )
                return getattr(result, "rowcount", 0) == 1

    async def renew_generation_claim(
        self,
        request_id: UUID,
        owner: str,
        *,
        lease_duration: timedelta,
        now: datetime,
    ) -> bool:
        instant = _utc(now)
        if lease_duration <= timedelta(0):
            raise ValueError("lease_duration must be positive")
        async with self._sessions() as session:
            async with session.begin():
                result = await session.execute(
                    update(GenerationRequestRecord)
                    .where(
                        GenerationRequestRecord.id == str(request_id),
                        GenerationRequestRecord.status == GenerationStatus.PROCESSING.value,
                        GenerationRequestRecord.claim_owner == owner,
                        GenerationRequestRecord.claim_expires_at > instant,
                    )
                    .values(claim_expires_at=instant + lease_duration)
                )
                return getattr(result, "rowcount", 0) == 1

    async def ensure_active_generation_claim_in_session(
        self,
        session: AsyncSession,
        request_id: UUID,
        owner: str,
        *,
        now: datetime,
    ) -> GenerationRequestRecord:
        """Lock one active lease before a generation-owned write."""

        if not owner.strip():
            raise ValueError("generation request owner cannot be empty")
        instant = _utc(now)
        statement = (
            update(GenerationRequestRecord)
            .where(
                GenerationRequestRecord.id == str(request_id),
                GenerationRequestRecord.status == GenerationStatus.PROCESSING.value,
                GenerationRequestRecord.claim_owner == owner,
                GenerationRequestRecord.claim_expires_at > instant,
            )
            .values(updated_at=instant)
            .returning(GenerationRequestRecord.id)
        )
        record_id = (await session.scalars(statement)).one_or_none()
        if record_id is None:
            record = await session.get(GenerationRequestRecord, str(request_id))
            if record is None:
                raise ContentNotFoundError(
                    f"generation request {request_id} was not found"
                )
            raise ContentRepositoryError(
                f"generation request {request_id} is no longer actively claimed by {owner}"
            )
        record = await session.get(GenerationRequestRecord, record_id)
        if record is None:
            raise ContentRepositoryError("active generation claim disappeared")
        return record

    async def add_candidates(
        self,
        request_id: UUID,
        candidates: Sequence[StoredCandidate],
        *,
        owner: str | None = None,
        now: datetime | None = None,
    ) -> list[StoredCandidate]:
        async with self._sessions() as session:
            async with session.begin():
                if await session.get(GenerationRequestRecord, str(request_id)) is None:
                    raise ContentNotFoundError(f"generation request {request_id} was not found")
                records: list[GeneratedCandidateRecord] = []
                for candidate in candidates:
                    record = await self.add_candidate_in_session(
                        session,
                        candidate,
                        owner=owner,
                        now=now,
                    )
                    records.append(record)
                await session.flush()
            return [_to_candidate(record) for record in records]

    async def add_candidate_in_session(
        self,
        session: AsyncSession,
        candidate: StoredCandidate,
        *,
        owner: str | None = None,
        now: datetime | None = None,
    ) -> GeneratedCandidateRecord:
        """Insert one candidate without opening or committing another session."""

        if owner is not None:
            await self.ensure_active_generation_claim_in_session(
                session,
                candidate.request_id,
                owner,
                now=now or _utc(),
            )
        if await session.get(GenerationRequestRecord, str(candidate.request_id)) is None:
            raise ContentNotFoundError(
                f"generation request {candidate.request_id} was not found"
            )
        ranking = candidate.ranking.model_dump(mode="json") if candidate.ranking else None
        decision = candidate.decision.model_dump(mode="json") if candidate.decision else None
        record = GeneratedCandidateRecord(
            id=str(candidate.id),
            request_id=str(candidate.request_id),
            ordinal=candidate.ordinal,
            revision_of_candidate_id=(
                str(candidate.revision_of_candidate_id)
                if candidate.revision_of_candidate_id
                else None
            ),
            title=candidate.draft.title,
            body=candidate.draft.body,
            strategy=candidate.draft.strategy,
            model_name=candidate.model_name,
            generation_parameters_json=_json(candidate.generation_parameters, default={}),
            used_example_ids_json=[str(item) for item in candidate.draft.used_example_ids],
            generated_at=candidate.generated_at,
            validation_json=candidate.validation.model_dump(mode="json"),
            ranking_score=candidate.ranking.score if candidate.ranking else None,
            ranking_explanation=(candidate.ranking.explanation if candidate.ranking else None),
            ranking_json=ranking,
            approval_status=candidate.approval_status.value,
            decision_json=decision,
            social_action_id=(str(candidate.social_action_id) if candidate.social_action_id else None),
            metadata_json=_json(candidate.metadata, default={}),
        )
        session.add(record)
        await session.flush()
        return record

    async def get_candidate(self, candidate_id: UUID) -> StoredCandidate | None:
        async with self._sessions() as session:
            return await self.get_candidate_in_session(session, candidate_id)

    async def get_candidate_in_session(
        self, session: AsyncSession, candidate_id: UUID
    ) -> StoredCandidate | None:
        record = await session.get(GeneratedCandidateRecord, str(candidate_id))
        return _to_candidate(record) if record is not None else None

    async def list_candidates(self, request_id: UUID) -> list[StoredCandidate]:
        async with self._sessions() as session:
            return await self.list_candidates_in_session(session, request_id)

    async def list_candidates_in_session(
        self, session: AsyncSession, request_id: UUID
    ) -> list[StoredCandidate]:
        statement = (
            select(GeneratedCandidateRecord)
            .where(GeneratedCandidateRecord.request_id == str(request_id))
            .order_by(GeneratedCandidateRecord.ordinal, GeneratedCandidateRecord.id)
        )
        records = (await session.scalars(statement)).all()
        return [_to_candidate(record) for record in records]

    async def reject_candidate(
        self, candidate_id: UUID, decision: CandidateDecision
    ) -> StoredCandidate:
        return await self._decide_candidate(
            candidate_id, decision, CandidateApprovalStatus.REJECTED
        )

    async def supersede_candidate(
        self, candidate_id: UUID, replacement_id: UUID, decision: CandidateDecision
    ) -> StoredCandidate:
        async with self._sessions() as session:
            async with session.begin():
                replacement = await session.get(GeneratedCandidateRecord, str(replacement_id))
                if replacement is None:
                    raise ContentNotFoundError("candidate revision lineage was not found")
                record = await self.supersede_candidate_in_session(
                    session, candidate_id, replacement_id, decision
                )
            return _to_candidate(record)

    async def supersede_candidate_in_session(
        self,
        session: AsyncSession,
        candidate_id: UUID,
        replacement_id: UUID,
        decision: CandidateDecision,
    ) -> GeneratedCandidateRecord:
        record = await session.get(GeneratedCandidateRecord, str(candidate_id))
        replacement = await session.get(GeneratedCandidateRecord, str(replacement_id))
        if record is None or replacement is None:
            raise ContentNotFoundError("candidate revision lineage was not found")
        if record.request_id != replacement.request_id:
            raise ContentRepositoryError(
                "candidate revisions must belong to the same generation request"
            )
        if record.approval_status != CandidateApprovalStatus.PENDING.value:
            raise ContentRepositoryError("only pending candidates can be superseded")
        record.approval_status = CandidateApprovalStatus.SUPERSEDED.value
        record.decision_json = decision.model_dump(mode="json")
        await session.flush()
        return record

    async def approve_candidate_in_session(
        self,
        session: AsyncSession,
        candidate_id: UUID,
        action_id: UUID,
        decision: CandidateDecision,
    ) -> StoredCandidate:
        record = await session.get(GeneratedCandidateRecord, str(candidate_id))
        if record is None:
            raise ContentNotFoundError(f"candidate {candidate_id} was not found")
        if record.approval_status != CandidateApprovalStatus.PENDING.value:
            raise ContentRepositoryError("candidate is no longer pending")
        validation = ValidationResult.model_validate(record.validation_json)
        if validation.has_errors:
            raise ContentRepositoryError("candidate has blocking validation errors")
        sibling = await session.scalar(
            select(GeneratedCandidateRecord.id).where(
                GeneratedCandidateRecord.request_id == record.request_id,
                GeneratedCandidateRecord.approval_status == CandidateApprovalStatus.APPROVED.value,
                GeneratedCandidateRecord.id != record.id,
            ).limit(1)
        )
        if sibling is not None:
            raise ContentRepositoryError("another candidate for this request is already approved")
        action = await session.get(SocialActionRecord, str(action_id))
        if action is None or action.status != ActionStatus.DRAFT.value:
            raise ContentRepositoryError("approved candidate requires a newly created draft action")
        record.approval_status = CandidateApprovalStatus.APPROVED.value
        record.decision_json = decision.model_dump(mode="json")
        record.social_action_id = str(action_id)
        await session.flush()
        return _to_candidate(record)

    async def has_approved_candidate_for_action(self, action_id: UUID) -> bool:
        statement = select(GeneratedCandidateRecord.id).where(
            GeneratedCandidateRecord.social_action_id == str(action_id),
            GeneratedCandidateRecord.approval_status == CandidateApprovalStatus.APPROVED.value,
        ).limit(1)
        async with self._sessions() as session:
            return (await session.scalar(statement)) is not None

    async def replace_topics(
        self,
        platform: Platform,
        topics: Sequence[DiscoveredTopic],
        *,
        discovered_at: datetime,
    ) -> list[DiscoveredTopic]:
        instant = _utc(discovered_at)
        async with self._sessions() as session:
            async with session.begin():
                await session.execute(
                    update(DiscoveredTopicRecord)
                    .where(
                        DiscoveredTopicRecord.platform == platform.value,
                        DiscoveredTopicRecord.is_active.is_(True),
                    )
                    .values(is_active=False)
                )
                records: list[DiscoveredTopicRecord] = []
                for topic in topics:
                    if topic.platform is not platform:
                        raise ValueError("topic platform does not match replacement platform")
                    record = DiscoveredTopicRecord(
                        id=str(topic.id),
                        platform=topic.platform.value,
                        label=topic.label,
                        keywords_json=list(topic.keywords),
                        supporting_example_ids_json=[str(item) for item in topic.supporting_example_ids],
                        support_count=topic.support_count,
                        distinct_source_count=topic.distinct_source_count,
                        median_recency=topic.median_recency,
                        discovered_at=instant,
                        expires_at=topic.expires_at,
                        is_active=True,
                        score=topic.score,
                    )
                    session.add(record)
                    records.append(record)
                await session.flush()
            return [_to_topic(record) for record in records]

    async def list_topics(
        self, platform: Platform | None = None, *, active_only: bool = True
    ) -> list[DiscoveredTopic]:
        statement = select(DiscoveredTopicRecord)
        if platform is not None:
            statement = statement.where(DiscoveredTopicRecord.platform == platform.value)
        if active_only:
            statement = statement.where(
                DiscoveredTopicRecord.is_active.is_(True),
                or_(
                    DiscoveredTopicRecord.expires_at.is_(None),
                    DiscoveredTopicRecord.expires_at > _utc(),
                ),
            )
        statement = statement.order_by(
            DiscoveredTopicRecord.discovered_at.desc(), DiscoveredTopicRecord.label
        )
        async with self._sessions() as session:
            records = (await session.scalars(statement)).all()
            return [_to_topic(record) for record in records]

    async def get_topic(self, topic_id: UUID) -> DiscoveredTopic | None:
        async with self._sessions() as session:
            record = await session.get(DiscoveredTopicRecord, str(topic_id))
            return _to_topic(record) if record is not None else None

    async def _decide_candidate(
        self,
        candidate_id: UUID,
        decision: CandidateDecision,
        status: CandidateApprovalStatus,
    ) -> StoredCandidate:
        async with self._sessions() as session:
            async with session.begin():
                record = await session.get(GeneratedCandidateRecord, str(candidate_id))
                if record is None:
                    raise ContentNotFoundError(f"candidate {candidate_id} was not found")
                if record.approval_status != CandidateApprovalStatus.PENDING.value:
                    raise ContentRepositoryError("only pending candidates can be decided")
                record.approval_status = status.value
                record.decision_json = decision.model_dump(mode="json")
            return _to_candidate(record)


__all__ = [
    "ContentNotFoundError",
    "ContentRepository",
    "ContentRepositoryError",
]
