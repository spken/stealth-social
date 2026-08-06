"""Collection, storage, refresh, and target-resolution coordination."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast
import asyncio
from datetime import UTC, datetime, timedelta
from uuid import UUID

import structlog

from bot.config import Settings
from bot.content.models import ContentRequest
from bot.content.prompt_safety import inspect_untrusted_text
from bot.examples.collectors import ExampleCollector, TargetContextResolver
from bot.examples.models import (
    CollectedExample,
    CollectionRunResult,
    CollectionRunStatus,
    ContentExample,
    ExampleCollectionError,
    ExampleCollectionRequest,
    ExampleType,
    ExampleListFilters,
    ExampleSelectionFilters,
    ExampleTargetUnavailableError,
    SelectedExample,
    TargetContext,
    TargetContextRequest,
)
from bot.examples.normalization import normalize_example
from bot.examples.selection import COMPATIBLE_TYPES, ExampleSelector, NoRelevantExamplesFound
from bot.models import ActionStatus, ActionType, Platform
from bot.storage.content_repository import ContentRepository
from bot.storage.repositories import ActionRepository

logger = structlog.get_logger(__name__)


class ExampleService:
    """Keep collection safety, normalization, and selection in one boundary."""

    def __init__(
        self,
        settings: Settings,
        content_repository: ContentRepository,
        *,
        action_repository: ActionRepository | None = None,
        collectors: Mapping[Platform, ExampleCollector] | None = None,
        target_resolvers: Mapping[Platform, TargetContextResolver] | None = None,
        selector: ExampleSelector | None = None,
    ) -> None:
        self._settings = settings
        self._repository = content_repository
        self._actions = action_repository
        self._collectors = dict(collectors or {})
        self._target_resolvers: dict[Platform, TargetContextResolver] = dict(
            target_resolvers
            or {
                platform: cast(TargetContextResolver, collector)
                for platform, collector in self._collectors.items()
            }
        )
        self._selector = selector or ExampleSelector()

    async def collect(
        self,
        request: ExampleCollectionRequest,
        *,
        replace_expired: bool = False,
    ):
        run = await self._repository.start_collection_run(request)
        collector = self._collectors.get(request.platform)
        if collector is None:
            error = ExampleCollectionError(
                f"no {request.platform.value} public collector is configured"
            )
            _annotate_collection_error(
                error, run.id, 0, status=CollectionRunStatus.FAILED
            )
            await self._repository.finish_collection_run(
                run.id,
                CollectionRunResult(
                    status=CollectionRunStatus.FAILED,
                    error_type=type(error).__name__,
                    error_message=str(error)[:300],
                ),
            )
            raise error
        inserted = refreshed = duplicates = rejected = 0
        try:
            collected = await collector.collect(request)
            normalized: list[CollectedExample] = []
            for item in collected:
                try:
                    normalized.append(self._normalize_collected(item, run.id))
                except ValueError:
                    rejected += 1
            for offset in range(0, len(normalized), 50):
                report = await self._repository.upsert_examples(
                    run.id, normalized[offset : offset + 50]
                )
                inserted += report.inserted_count
                refreshed += report.refreshed_count
                duplicates += report.duplicate_count
                rejected += report.rejected_count
            disabled_count = (
                await self._repository.disable_expired_examples(
                    request.platform, disabled_at=datetime.now(UTC)
                )
                if replace_expired
                else 0
            )
            finished = await self._repository.finish_collection_run(
                run.id,
                CollectionRunResult(
                    status=CollectionRunStatus.COMPLETED,
                    collected_count=inserted + refreshed,
                    rejected_count=rejected,
                    duplicate_count=duplicates,
                    disabled_count=disabled_count,
                ),
            )
            return run.model_copy(
                update={
                    "status": CollectionRunStatus.COMPLETED,
                    "finished_at": finished.finished_at,
                    "collected_count": inserted + refreshed,
                    "rejected_count": rejected,
                    "duplicate_count": duplicates,
                    "disabled_count": disabled_count,
                }
            )
        except asyncio.CancelledError:
            error = ExampleCollectionError("collection was cancelled")
            _annotate_collection_error(
                error,
                run.id,
                inserted + refreshed,
                status=CollectionRunStatus.PARTIAL,
            )
            await self._repository.finish_collection_run(
                run.id,
                CollectionRunResult(
                    status=CollectionRunStatus.PARTIAL,
                    collected_count=inserted + refreshed,
                    rejected_count=rejected,
                    error_type="CancelledError",
                    error_message="collection was cancelled",
                ),
            )
            raise
        except ExampleCollectionError as error:
            _annotate_collection_error(
                error,
                run.id,
                inserted + refreshed,
                status=CollectionRunStatus.PARTIAL,
            )
            await self._repository.finish_collection_run(
                run.id,
                CollectionRunResult(
                    status=CollectionRunStatus.PARTIAL,
                    collected_count=inserted + refreshed,
                    rejected_count=rejected,
                    duplicate_count=duplicates,
                    error_type=type(error).__name__,
                    error_message=str(error)[:300],
                    retry_after_seconds=error.retry_after_seconds,
                ),
            )
            raise
        except (TypeError, ValueError) as error:
            _annotate_collection_error(
                error,
                run.id,
                inserted + refreshed,
                status=CollectionRunStatus.FAILED,
            )
            await self._repository.finish_collection_run(
                run.id,
                CollectionRunResult(
                    status=CollectionRunStatus.FAILED,
                    collected_count=inserted + refreshed,
                    rejected_count=rejected,
                    error_type=type(error).__name__,
                    error_message=" ".join(str(error).split())[:300],
                ),
            )
            raise
        except Exception as error:
            _annotate_collection_error(
                error,
                run.id,
                inserted + refreshed,
                status=CollectionRunStatus.FAILED,
            )
            await self._repository.finish_collection_run(
                run.id,
                CollectionRunResult(
                    status=CollectionRunStatus.FAILED,
                    collected_count=inserted + refreshed,
                    rejected_count=rejected,
                    duplicate_count=duplicates,
                    error_type=type(error).__name__,
                    error_message=" ".join(str(error).split())[:300],
                ),
            )
            raise

    async def refresh(
        self,
        platform: Platform,
        *,
        account_name: str,
    ):
        config = self._settings.example_collection
        if platform is Platform.X:
            sources = config.x
            request = ExampleCollectionRequest(
                platform=platform,
                account_name=account_name,
                accounts=sources.accounts,
                queries=sources.queries,
                post_urls=sources.post_urls,
                maximum_items_per_source=config.maximum_items_per_source,
                maximum_comments_per_post=config.maximum_comments_per_post,
                minimum_score=config.minimum_score,
                include_own_content=config.include_own_content,
            )
        else:
            sources = config.reddit
            request = ExampleCollectionRequest(
                platform=platform,
                account_name=account_name,
                subreddits=sources.subreddits,
                queries=sources.queries,
                post_urls=sources.post_urls,
                sort=sources.sort,
                time_filter=sources.time_filter,
                maximum_items_per_source=config.maximum_items_per_source,
                maximum_comments_per_post=config.maximum_comments_per_post,
                minimum_score=config.minimum_score,
                include_own_content=config.include_own_content,
            )
        latest = await self._repository.latest_collection_run(
            platform,
            account_name=account_name,
        )
        if (
            latest is not None
            and latest.finished_at is not None
            and datetime.now(UTC) - latest.finished_at
            < timedelta(hours=config.refresh_interval_hours)
        ):
            return latest
        return await self.collect(request, replace_expired=True)

    async def resolve_target(self, request: TargetContextRequest) -> TargetContext:
        resolver = self._target_resolvers.get(request.platform)
        if resolver is None:
            raise ExampleTargetUnavailableError(
                f"no {request.platform.value} target resolver is configured"
            )
        return await resolver.resolve_target(request)

    async def list(self, filters: ExampleListFilters) -> list[ContentExample]:
        return await self._repository.list_examples(filters)

    async def show(self, example_id: UUID) -> ContentExample:
        example = await self._repository.get_example(example_id)
        if example is None:
            raise ExampleTargetUnavailableError(f"example {example_id} was not found")
        return example

    async def disable(self, example_id: UUID) -> ContentExample:
        return await self._repository.disable_example(example_id, disabled_at=datetime.now(UTC))

    async def select_for_request(self, request: ContentRequest) -> tuple[SelectedExample, ...]:
        base_filters = request.example_selection_filters or ExampleSelectionFilters(
            platform=request.platform,
            compatible_types=tuple(COMPATIBLE_TYPES[request.generation_type]),
            topic=request.topic,
            keywords=request.keywords,
            target_text=request.source_post_text or request.source_comment_text,
            audience=request.target_audience,
            tone=request.tone,
            maximum_context_examples=int(
                request.resolved_parameters.get(
                    "maximum_context_examples",
                    self._settings.content_generation.maximum_context_examples,
                )
            ),
            maximum_example_characters=int(
                request.resolved_parameters.get(
                    "maximum_example_characters",
                    self._settings.content_generation.maximum_example_characters,
                )
            ),
            allow_generated=self._settings.content_generation.allow_generated_style_examples,
            useful_window_days=self._settings.example_collection.useful_window_days,
        )
        account = (
            self._settings.accounts.x.get(request.account_name)
            if request.platform is Platform.X
            else self._settings.accounts.reddit.get(request.account_name)
        )
        if account is None or not account.enabled:
            raise ValueError(
                f"no enabled {request.platform.value} account named {request.account_name!r}"
            )
        resolved_subreddit = request.subreddit or (
            request.target_context.subreddit
            if request.target_context is not None
            else None
        )
        reddit_account = (
            self._settings.accounts.reddit.get(request.account_name)
            if request.platform is Platform.REDDIT
            else None
        )
        filters = base_filters.model_copy(
            update={
                "platform": request.platform,
                "compatible_types": tuple(COMPATIBLE_TYPES[request.generation_type]),
                "subreddit": resolved_subreddit,
                "allowed_subreddits": tuple(
                    reddit_account.allowed_subreddits
                    if reddit_account is not None
                    else ()
                ),
                "maximum_context_examples": int(
                    request.resolved_parameters.get(
                        "maximum_context_examples",
                        self._settings.content_generation.maximum_context_examples,
                    )
                ),
                "maximum_example_characters": int(
                    request.resolved_parameters.get(
                        "maximum_example_characters",
                        self._settings.content_generation.maximum_example_characters,
                    )
                ),
                "allow_generated": (
                    base_filters.allow_generated
                    and self._settings.content_generation.allow_generated_style_examples
                ),
                "useful_window_days": self._settings.example_collection.useful_window_days,
            }
        )
        pool = await self._repository.list_selection_pool(filters)
        pool = list(pool) + await self._own_examples(request)
        try:
            selected = self._selector.select(
                request.model_copy(update={"example_selection_filters": filters}),
                pool,
            )
        except NoRelevantExamplesFound:
            logger.info(
                "no_relevant_examples_found",
                request_id=str(request.id),
                selected_count=0,
            )
            return ()
        return selected

    def _normalize_collected(
        self, item: CollectedExample, run_id: UUID
    ) -> CollectedExample:
        findings = []
        for text in (item.title, item.body, item.parent_text):
            if not text:
                continue
            findings.extend(
                finding.model_dump(mode="json")
                for finding in inspect_untrusted_text(text).findings
            )
        content = normalize_example(
            item.model_copy(update={"injection_findings": tuple(findings)}),
            collection_run_id=run_id,
            expires_at=datetime.now(UTC)
            + timedelta(hours=self._settings.example_collection.expiry_interval_hours),
        )
        return CollectedExample(
            platform=content.platform,
            content_type=content.content_type,
            external_id=content.external_id,
            source_url=content.source_url,
            author_identifier=content.author_identifier,
            title=content.title,
            body=content.body,
            parent_text=content.parent_text,
            subreddit=content.subreddit,
            published_at=content.published_at,
            collected_at=content.collected_at,
            expires_at=content.expires_at,
            content_hash=content.content_hash,
            engagement_score=content.engagement_score,
            metadata=content.metadata,
            topic_tags=content.topic_tags,
            is_own_content=content.is_own_content,
            generated=content.generated,
            injection_findings=content.injection_findings,
        )

    async def _own_examples(self, request: ContentRequest) -> list[ContentExample]:
        if not request.account_name or not self._settings.example_collection.include_own_content:
            return []
        if self._actions is None:
            return []
        actions = await self._actions.list(
            statuses=[ActionStatus.PUBLISHED],
            platform=request.platform,
            account_name=request.account_name,
            limit=100,
        )
        results: list[ContentExample] = []
        for action in actions:
            if not action.external_content_url:
                continue
            content_type = {
                ActionType.X_POST: ExampleType.X_POST,
                ActionType.REDDIT_POST: ExampleType.REDDIT_POST,
                ActionType.REDDIT_COMMENT: ExampleType.REDDIT_COMMENT,
                ActionType.REDDIT_REPLY: ExampleType.REDDIT_COMMENT,
            }[action.action_type]
            raw = CollectedExample(
                platform=request.platform,
                content_type=content_type,
                external_id=action.external_content_id,
                source_url=action.external_content_url,
                title=action.title,
                body=action.content,
                subreddit=action.subreddit,
                is_own_content=True,
                generated=bool(action.metadata.get("generated_candidate_id")),
                metadata={"own_published_action_id": str(action.id)},
            )
            try:
                results.append(normalize_example(raw, collection_run_id=None))
            except ValueError:
                continue
        return results


def _annotate_collection_error(
    error: BaseException,
    run_id: UUID,
    saved_count: int,
    *,
    status: CollectionRunStatus,
) -> None:
    """Expose bounded run context to the CLI without storing browser details."""

    setattr(error, "run_id", run_id)
    setattr(error, "saved_count", max(0, saved_count))
    setattr(error, "run_status", status.value)


__all__ = ["ExampleService"]
