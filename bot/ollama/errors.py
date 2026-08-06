"""Typed, user-safe Ollama failures."""


class OllamaError(RuntimeError):
    """Base error for local Ollama operations."""


class OllamaUnavailableError(OllamaError):
    """The configured Ollama server could not be reached."""


class OllamaModelMissingError(OllamaError):
    """The server is reachable but the configured model is absent."""

    def __init__(self, model: str) -> None:
        self.model = model
        super().__init__(
            f"Ollama is running, but {model} is not installed.\n\n"
            f"Install it with:\n\nollama pull {model}"
        )


class OllamaTimeoutError(OllamaError):
    """The configured request timeout elapsed."""


class InvalidOllamaResponseError(OllamaError):
    """Ollama returned malformed or unsupported JSON."""
