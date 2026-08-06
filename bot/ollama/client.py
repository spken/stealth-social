"""Async Ollama HTTP transport with bounded retries and cancellation safety."""

from __future__ import annotations

import asyncio
import random
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, StringConstraints
from typing_extensions import Annotated

from bot.config import ContentGenerationSettings
from bot.ollama.errors import (
    InvalidOllamaResponseError,
    OllamaModelMissingError,
    OllamaTimeoutError,
    OllamaUnavailableError,
)

NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class _OllamaModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class OllamaServerStatus(_OllamaModel):
    base_url: NonEmptyString
    reachable: bool
    version: str | None = None


class OllamaModel(_OllamaModel):
    name: NonEmptyString
    model: NonEmptyString | None = None
    size: int | None = Field(default=None, ge=0)
    parameter_size: str | None = None
    quantization_level: str | None = None
    modified_at: datetime | str | None = None


class OllamaMessage(_OllamaModel):
    role: Literal["system", "user", "assistant"]
    content: str


class OllamaChatRequest(_OllamaModel):
    model: NonEmptyString
    messages: tuple[OllamaMessage, ...]
    format_schema: dict[str, Any] = Field(alias="format")
    options: dict[str, Any] = Field(default_factory=dict)
    stream: Literal[False] = False
    think: bool = False

    model_config = ConfigDict(
        alias_generator=None,
        populate_by_name=True,
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
    )


class OllamaChatResponse(_OllamaModel):
    content: str
    model: NonEmptyString
    created_at: str | None = None
    done_reason: str | None = None
    total_duration: int | None = Field(default=None, ge=0)
    load_duration: int | None = Field(default=None, ge=0)
    prompt_eval_count: int | None = Field(default=None, ge=0)
    prompt_eval_duration: int | None = Field(default=None, ge=0)
    eval_count: int | None = Field(default=None, ge=0)
    eval_duration: int | None = Field(default=None, ge=0)
    thinking_was_returned: bool = False


class OllamaClient:
    """Own or reuse one HTTP client for the lifetime of a runtime."""

    def __init__(
        self,
        settings: ContentGenerationSettings,
        *,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        base_url = str(settings.base_url).rstrip("/")
        if not base_url:
            raise ValueError("Ollama base URL must not be empty")
        self._settings = settings
        self._base_url = base_url
        self._client = http_client or httpx.AsyncClient(
            timeout=httpx.Timeout(settings.request_timeout_seconds)
        )
        self._owns_client = http_client is None
        self._closed = False

    @property
    def base_url(self) -> str:
        return self._base_url

    async def status(self) -> OllamaServerStatus:
        payload = await self._request("GET", "/api/version")
        version = payload.get("version")
        if version is not None and not isinstance(version, str):
            raise InvalidOllamaResponseError("Ollama version response was malformed")
        return OllamaServerStatus(
            base_url=self._base_url,
            reachable=True,
            version=version,
        )

    async def list_models(self) -> tuple[OllamaModel, ...]:
        payload = await self._request("GET", "/api/tags")
        raw_models = payload.get("models")
        if not isinstance(raw_models, list):
            raise InvalidOllamaResponseError("Ollama model response was malformed")
        models: list[OllamaModel] = []
        for raw_model in raw_models:
            if not isinstance(raw_model, Mapping):
                raise InvalidOllamaResponseError("Ollama model response was malformed")
            model_name = raw_model.get("name") or raw_model.get("model")
            if not isinstance(model_name, str):
                raise InvalidOllamaResponseError("Ollama model response was malformed")
            try:
                models.append(
                    OllamaModel(
                        name=model_name,
                        model=raw_model.get("model"),
                        size=raw_model.get("size"),
                        parameter_size=(raw_model.get("details") or {}).get(
                            "parameter_size"
                        ),
                        quantization_level=(raw_model.get("details") or {}).get(
                            "quantization_level"
                        ),
                        modified_at=raw_model.get("modified_at"),
                    )
                )
            except (TypeError, ValueError) as error:
                raise InvalidOllamaResponseError(
                    "Ollama model response was malformed"
                ) from error
        return tuple(models)

    async def require_model(self, model: str | None = None) -> OllamaModel:
        requested = (model or self._settings.model).strip()
        if not requested:
            raise ValueError("Ollama model must not be blank")
        for installed in await self.list_models():
            if requested in {installed.name.strip(), (installed.model or "").strip()}:
                return installed
        raise OllamaModelMissingError(requested)

    async def show_model(self, model: str | None = None) -> dict[str, Any]:
        requested = (model or self._settings.model).strip()
        payload = await self._request("POST", "/api/show", json={"name": requested})
        allowed = {"details", "capabilities"}
        return {key: payload[key] for key in allowed if key in payload}

    async def chat(
        self,
        *,
        model: str,
        messages: Sequence[OllamaMessage],
        format_schema: dict[str, Any],
        options: Mapping[str, Any],
        think: bool,
    ) -> OllamaChatResponse:
        request = OllamaChatRequest(
            model=model,
            messages=tuple(messages),
            format=format_schema,
            options=dict(options),
            stream=False,
            think=think,
        )
        payload = await self._request(
            "POST",
            "/api/chat",
            json=request.model_dump(by_alias=True),
        )
        message = payload.get("message")
        if not isinstance(message, Mapping) or not isinstance(message.get("content"), str):
            raise InvalidOllamaResponseError("Ollama chat response was malformed")
        try:
            return OllamaChatResponse(
                content=message["content"],
                model=payload.get("model") or model,
                created_at=payload.get("created_at"),
                done_reason=payload.get("done_reason"),
                total_duration=payload.get("total_duration"),
                load_duration=payload.get("load_duration"),
                prompt_eval_count=payload.get("prompt_eval_count"),
                prompt_eval_duration=payload.get("prompt_eval_duration"),
                eval_count=payload.get("eval_count"),
                eval_duration=payload.get("eval_duration"),
                thinking_was_returned=isinstance(message.get("thinking"), str),
            )
        except (TypeError, ValueError) as error:
            raise InvalidOllamaResponseError("Ollama chat response was malformed") from error

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> OllamaClient:
        if self._closed:
            raise OllamaUnavailableError("Ollama client is closed")
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self._closed:
            raise OllamaUnavailableError("Ollama client is closed")
        attempts = self._settings.maximum_retries + 1
        for attempt in range(attempts):
            try:
                response = await self._client.request(
                    method,
                    f"{self._base_url}{path}",
                    json=json,
                )
            except asyncio.CancelledError:
                raise
            except httpx.TimeoutException as error:
                # Treat request timeouts like other transient transport failures;
                # cancellation remains the only immediate exit from this loop.
                if attempt + 1 < attempts:
                    await self._retry_delay(attempt)
                    continue
                raise OllamaTimeoutError(
                    f"Ollama request timed out after {self._settings.request_timeout_seconds:g} seconds"
                ) from error
            except httpx.RequestError as error:
                if attempt + 1 >= attempts:
                    raise OllamaUnavailableError(
                        f"Ollama server at {self._base_url} is unavailable"
                    ) from error
                await self._retry_delay(attempt)
                continue

            if response.status_code in {502, 503, 504} and attempt + 1 < attempts:
                await self._retry_delay(attempt)
                continue
            if response.status_code < 200 or response.status_code >= 300:
                if response.status_code in {502, 503, 504}:
                    raise OllamaUnavailableError(
                        f"Ollama server at {self._base_url} is unavailable"
                    )
                raise InvalidOllamaResponseError(
                    f"Ollama returned HTTP {response.status_code} for {path}"
                )
            try:
                payload = response.json()
            except (TypeError, ValueError) as error:
                raise InvalidOllamaResponseError(
                    "Ollama returned invalid JSON"
                ) from error
            if not isinstance(payload, dict):
                raise InvalidOllamaResponseError("Ollama returned an invalid JSON object")
            return payload
        raise OllamaUnavailableError(f"Ollama server at {self._base_url} is unavailable")

    async def _retry_delay(self, attempt: int) -> None:
        delay = min(4.0, 0.25 * (2**attempt)) + random.uniform(0.0, 0.1)
        await asyncio.sleep(delay)


__all__ = [
    "OllamaChatRequest",
    "OllamaChatResponse",
    "OllamaClient",
    "OllamaMessage",
    "OllamaModel",
    "OllamaServerStatus",
]
