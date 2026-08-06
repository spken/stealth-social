"""Composition root for content generation and browser-backed examples."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncIterator

from bot.browser.manager import BrowserManager
from bot.config import Settings
from bot.content.approvals import CandidateService
from bot.content.ollama_generator import OllamaContentGenerator
from bot.content.prompt_builder import PromptBuilder
from bot.content.ranking import HeuristicCandidateRanker, OllamaCandidateRanker
from bot.content.service import GenerationService
from bot.content.topics import TopicDiscoveryService
from bot.content.validation import CandidateValidator
from bot.examples.collectors import RedditExampleCollector, XExampleCollector
from bot.examples.service import ExampleService
from bot.generation_worker import GenerationWorker
from bot.models import Platform
from bot.ollama.client import OllamaClient
from bot.storage.database import Database
from bot.storage.repositories import AccountStateRepository, ActionRepository
from bot.storage.content_repository import ContentRepository


@dataclass(frozen=True, slots=True)
class ContentRuntime:
    database: Database
    browser_manager: BrowserManager
    ollama_client: OllamaClient
    action_repository: ActionRepository
    account_state_repository: AccountStateRepository
    content_repository: ContentRepository
    example_service: ExampleService
    generation_service: GenerationService
    candidate_service: CandidateService
    topic_service: TopicDiscoveryService
    generation_worker: GenerationWorker


@asynccontextmanager
async def content_runtime(settings: Settings) -> AsyncIterator[ContentRuntime]:
    """Own content resources and close them in dependency order."""

    database = Database(settings.database_url)
    browser_manager: BrowserManager | None = None
    ollama_client: OllamaClient | None = None
    generation_worker: GenerationWorker | None = None
    try:
        await database.initialize()
        browser_manager = BrowserManager(settings)
        ollama_client = OllamaClient(settings.content_generation)
        action_repository = ActionRepository(database.session_factory)
        account_state_repository = AccountStateRepository(database.session_factory)
        content_repository = ContentRepository(database.session_factory)
        collectors = {
            Platform.X: XExampleCollector(browser_manager, settings),
            Platform.REDDIT: RedditExampleCollector(browser_manager, settings),
        }
        example_service = ExampleService(
            settings,
            content_repository,
            action_repository=action_repository,
            collectors=collectors,
        )
        prompt_builder = PromptBuilder()
        generator = OllamaContentGenerator(
            ollama_client,
            prompt_builder,
            settings.content_generation,
        )
        ranker = (
            OllamaCandidateRanker(ollama_client, prompt_builder)
            if settings.content_generation.ranking_mode.value == "ollama"
            else HeuristicCandidateRanker()
        )
        generation_service = GenerationService(
            settings,
            content_repository,
            example_service,
            generator,
            CandidateValidator(),
            ranker,
            action_repository=action_repository,
        )
        candidate_service = CandidateService(
            settings,
            content_repository,
            action_repository,
            generation_service,
            database.session_factory,
        )
        generation_service.attach_unattended_approval_service(candidate_service)
        topic_service = TopicDiscoveryService(
            settings,
            content_repository,
            generation_service,
        )
        generation_worker = GenerationWorker(
            settings,
            content_repository,
            generation_service,
        )
        yield ContentRuntime(
            database=database,
            browser_manager=browser_manager,
            ollama_client=ollama_client,
            action_repository=action_repository,
            account_state_repository=account_state_repository,
            content_repository=content_repository,
            example_service=example_service,
            generation_service=generation_service,
            candidate_service=candidate_service,
            topic_service=topic_service,
            generation_worker=generation_worker,
        )
    finally:
        close_errors: list[BaseException] = []
        cancellation_error: asyncio.CancelledError | None = None
        if generation_worker is not None:
            try:
                await generation_worker.close()
            except asyncio.CancelledError as error:
                cancellation_error = error
            except BaseException as error:
                close_errors.append(error)
        if ollama_client is not None:
            try:
                await ollama_client.close()
            except asyncio.CancelledError as error:
                cancellation_error = error
            except BaseException as error:
                close_errors.append(error)
        if browser_manager is not None:
            try:
                await browser_manager.shutdown()
            except asyncio.CancelledError as error:
                cancellation_error = error
            except BaseException as error:
                close_errors.append(error)
        try:
            await database.close()
        except asyncio.CancelledError as error:
            cancellation_error = error
        except BaseException as error:
            close_errors.append(error)
        if cancellation_error is not None:
            raise cancellation_error
        if close_errors:
            raise RuntimeError("content runtime resource cleanup failed") from close_errors[0]


__all__ = ["ContentRuntime", "content_runtime"]
