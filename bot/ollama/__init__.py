"""Small typed HTTP integration for a local Ollama server."""

from bot.ollama.client import (
    OllamaChatRequest,
    OllamaChatResponse,
    OllamaClient,
    OllamaMessage,
    OllamaModel,
    OllamaServerStatus,
)
from bot.ollama.errors import (
    InvalidOllamaResponseError,
    OllamaError,
    OllamaModelMissingError,
    OllamaTimeoutError,
    OllamaUnavailableError,
)

__all__ = [
    "InvalidOllamaResponseError",
    "OllamaChatRequest",
    "OllamaChatResponse",
    "OllamaClient",
    "OllamaError",
    "OllamaMessage",
    "OllamaModel",
    "OllamaModelMissingError",
    "OllamaServerStatus",
    "OllamaTimeoutError",
    "OllamaUnavailableError",
]
