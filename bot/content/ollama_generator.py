"""Structured Qwen generation with one bounded repair attempt."""

from __future__ import annotations

import hashlib
import re
import time
from datetime import UTC, datetime
from typing import TypedDict, cast

import structlog

from bot.config import ContentGenerationSettings
from bot.content.generator import ContentGenerator
from bot.content.models import ContentRequest, GenerationResult, StructuredGenerationResponse
from bot.content.prompt_builder import PromptBuilder, PromptConstructionError, PromptPackage
from bot.ollama.client import OllamaClient, OllamaMessage


class StructuredOutputParsingError(RuntimeError):
    """The original and one constrained repair response were not usable."""

    def __init__(
        self,
        message: str,
        *,
        response_sha256: str,
        response_length: int,
        response_excerpt: str,
    ) -> None:
        self.response_sha256 = response_sha256
        self.response_length = response_length
        self.response_excerpt = response_excerpt
        super().__init__(message)


logger = structlog.get_logger(__name__)


class _ResolvedParameters(TypedDict):
    model: str
    temperature: float
    top_p: float
    candidate_count: int
    thinking: bool
    maximum_context_examples: int
    prompt_version: str
    schema_version: str


class _FailureMetadata(TypedDict):
    response_sha256: str
    response_length: int
    response_excerpt: str


class OllamaContentGenerator(ContentGenerator):
    """Implement the batch generator contract on top of :class:`OllamaClient`."""

    def __init__(
        self,
        client: OllamaClient,
        prompt_builder: PromptBuilder,
        settings: ContentGenerationSettings,
    ) -> None:
        self._client = client
        self._prompts = prompt_builder
        self._settings = settings

    async def generate(self, request: ContentRequest) -> GenerationResult:
        package = self._prompts.build(request)
        if self._settings.debug_prompt_logging:
            logger.info(
                "prompt_debug_metadata",
                request_id=str(request.id),
                prompt_version=package.prompt_version,
                schema_version=package.schema_version,
                prompt_fingerprint=package.prompt_fingerprint,
                prompt_character_count=len(package.system_message)
                + len(package.user_message),
                selected_example_count=len(request.selected_examples),
                target_context_present=request.target_context is not None,
            )
        parameters = self._resolved_parameters(request, package)
        await self._client.require_model(parameters["model"])
        started = time.monotonic()
        response = await self._client.chat(
            model=parameters["model"],
            messages=(
                OllamaMessage(role="system", content=package.system_message),
                OllamaMessage(role="user", content=package.user_message),
            ),
            format_schema=package.output_schema,
            options={
                "temperature": parameters["temperature"],
                "top_p": parameters["top_p"],
            },
            think=parameters["thinking"],
        )
        try:
            drafts = self._parse(response.content, request, package.strategies)
            repair_count = 0
        except (ValueError, TypeError, PromptConstructionError):
            drafts, repair_count = await self._repair(
                response.content,
                request,
                package,
                model=parameters["model"],
            )
        latency = time.monotonic() - started
        return GenerationResult(
            candidates=drafts,
            model_name=response.model,
            resolved_parameters=cast(dict[str, object], parameters),
            created_at=datetime.now(UTC),
            latency_seconds=latency,
            prompt_tokens=response.prompt_eval_count,
            completion_tokens=response.eval_count,
            metadata={
                "done_reason": response.done_reason,
                "total_duration": response.total_duration,
                "load_duration": response.load_duration,
                "prompt_eval_duration": response.prompt_eval_duration,
                "eval_duration": response.eval_duration,
                "prompt_version": package.prompt_version,
                "schema_version": package.schema_version,
                "prompt_fingerprint": package.prompt_fingerprint,
                "repair_count": repair_count,
                "thinking_was_returned": response.thinking_was_returned,
            },
        )

    def _parse(self, content: str, request: ContentRequest, strategies: tuple[str, ...]):
        parsed = StructuredGenerationResponse.model_validate_json(content)
        return self._prompts.validate_response(parsed, request, strategies)

    async def _repair(
        self,
        malformed_content: str,
        request: ContentRequest,
        package: PromptPackage,
        *,
        model: str,
    ):
        metadata = _failure_metadata(malformed_content, request)
        safe_content = _redact_content(malformed_content[:12_000], request)
        repair_package = self._prompts.build_repair(safe_content, package.output_schema)
        response = await self._client.chat(
            model=model,
            messages=(
                OllamaMessage(role="system", content=repair_package.system_message),
                OllamaMessage(role="user", content=repair_package.user_message),
            ),
            format_schema=repair_package.output_schema,
            options={"temperature": 0.0, "top_p": 1.0},
            think=False,
        )
        try:
            return self._parse(response.content, request, package.strategies), 1
        except (ValueError, TypeError, PromptConstructionError) as error:
            raise StructuredOutputParsingError(
                "Ollama returned structured content that could not be repaired",
                **metadata,
            ) from error

    def _resolved_parameters(
        self, request: ContentRequest, package
    ) -> _ResolvedParameters:
        resolved = dict(request.resolved_parameters)
        resolved.setdefault("model", self._settings.model)
        resolved.setdefault("temperature", self._settings.temperature)
        resolved.setdefault("top_p", self._settings.top_p)
        resolved.setdefault("candidate_count", request.candidate_count)
        resolved.setdefault("thinking", self._settings.thinking)
        resolved.setdefault("maximum_context_examples", len(request.selected_examples))
        resolved.setdefault("prompt_version", package.prompt_version)
        resolved.setdefault("schema_version", package.schema_version)
        return cast(_ResolvedParameters, resolved)


def _failure_metadata(content: str, request: ContentRequest) -> _FailureMetadata:
    excerpt = _redact_content(content[:240], request)
    excerpt = re.sub(r"[\x00-\x1f\x7f]", " ", excerpt)
    return {
        "response_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "response_length": len(content),
        "response_excerpt": " ".join(excerpt.split())[:240],
    }


def _redact_content(content: str, request: ContentRequest) -> str:
    values = _prompt_secret_values(request)
    redacted = content
    for secret in values:
        if secret:
            redacted = redacted.replace(secret, "[REDACTED]")
    return redacted


def _prompt_secret_values(request: ContentRequest) -> tuple[str, ...]:
    return tuple(
        value
        for value in (
            request.account_context.identity,
            *request.account_context.products,
            *request.account_context.verified_facts,
            *request.account_context.forbidden_claims,
            *request.account_context.required_disclosures,
            *request.forbidden_claims,
            *request.forbidden_phrases,
            *(item.statement for item in request.required_facts),
            *(term for item in request.required_facts for term in item.required_terms),
            request.product_context,
            request.project_context,
        )
        if value and value.strip()
    )


__all__ = ["OllamaContentGenerator", "StructuredOutputParsingError"]
