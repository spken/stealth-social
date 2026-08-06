"""Safe Ollama health and model-inspection commands."""

from __future__ import annotations

from typing import Annotated

import typer

from bot.commands.common import (
    _emit_json,
    _format_datetime,
    _run_async,
    _safe_command,
    _settings,
)
from bot.ollama.client import OllamaClient, OllamaModel
from bot.ollama.errors import OllamaError

ollama_app = typer.Typer(
    help="Inspect the configured local Ollama service without running generation.",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
    rich_markup_mode=None,
)


async def _with_client(operation):
    settings = _settings()
    async with OllamaClient(settings.content_generation) as client:
        return await operation(client, settings)


def _model_summary(model: OllamaModel) -> dict[str, object]:
    modified = model.modified_at
    if hasattr(modified, "isoformat"):
        modified = _format_datetime(modified)  # type: ignore[arg-type]
    return {
        "name": model.name,
        "size": model.size,
        "parameter_size": model.parameter_size,
        "quantization_level": model.quantization_level,
        "modified_at": modified,
    }


def _ollama_error_payload(settings, error: OllamaError) -> dict[str, object]:
    return {
        "base_url": str(settings.content_generation.base_url),
        "ready": False,
        "error_type": type(error).__name__,
        "error": " ".join(str(error).split())[:500]
        or "Ollama operation failed safely.",
    }


@ollama_app.command("status")
@_safe_command
def status_command(
    as_json: Annotated[
        bool,
        typer.Option("--json", help="Emit a safe JSON object."),
    ] = False,
) -> None:
    """Report configured Ollama reachability and server version."""

    settings = _settings()

    async def operation():
        async with OllamaClient(settings.content_generation) as client:
            return await client.status()

    try:
        status = _run_async(operation())
    except OllamaError as error:
        result = {
            "base_url": str(settings.content_generation.base_url),
            "reachable": False,
            "version": None,
            "error_type": type(error).__name__,
            "error": " ".join(str(error).split())[:500]
            or "Ollama is unavailable at the configured base URL.",
        }
        if as_json:
            _emit_json(result)
        else:
            typer.echo(f"base_url={result['base_url']}")
            typer.echo("reachable=false")
            typer.echo("version=unavailable")
            typer.echo(f"error={result['error']}", err=True)
        raise typer.Exit(code=1) from error

    result = {
        "base_url": status.base_url,
        "reachable": status.reachable,
        "version": status.version,
    }
    if as_json:
        _emit_json(result)
    else:
        typer.echo(f"base_url={status.base_url}")
        typer.echo(f"reachable={'true' if status.reachable else 'false'}")
        typer.echo(f"version={status.version or 'unknown'}")


@ollama_app.command("models")
@_safe_command
def models_command(
    as_json: Annotated[
        bool,
        typer.Option("--json", help="Emit safe JSON objects."),
    ] = False,
) -> None:
    """List installed model metadata without templates or licenses."""

    try:
        models = _run_async(
            _with_client(lambda client, _settings: client.list_models())
        )
    except OllamaError as error:
        result = _ollama_error_payload(_settings(), error)
        result["models"] = []
        if as_json:
            _emit_json(result)
        else:
            typer.echo(f"error={result['error']}", err=True)
        raise typer.Exit(code=1) from error
    rows = [_model_summary(model) for model in models]
    if as_json:
        _emit_json(rows)
        return
    if not rows:
        typer.echo("No installed models found.")
        return
    for row in rows:
        typer.echo(
            " ".join(
                f"{key}={row[key]}"
                for key in (
                    "name",
                    "size",
                    "parameter_size",
                    "quantization_level",
                    "modified_at",
                )
            )
        )


@ollama_app.command("check")
@_safe_command
def check_command(
    as_json: Annotated[
        bool,
        typer.Option("--json", help="Emit a safe JSON object."),
    ] = False,
) -> None:
    """Check reachability, configured model presence, and bounded capabilities."""

    settings = _settings()

    async def operation():
        async with OllamaClient(settings.content_generation) as client:
            status = await client.status()
            model = await client.require_model(settings.content_generation.model)
            capabilities = await client.show_model(settings.content_generation.model)
            raw_capabilities = capabilities.get("capabilities")
            bounded_capabilities = (
                [
                    item[:64]
                    for item in raw_capabilities
                    if isinstance(item, str)
                ][:20]
                if isinstance(raw_capabilities, list)
                else []
            )
            return status, model, bounded_capabilities

    try:
        status, model, capabilities = _run_async(operation())
    except OllamaError as error:
        result = _ollama_error_payload(settings, error)
        result["model"] = settings.content_generation.model
        result["capabilities"] = []
        if as_json:
            _emit_json(result)
        else:
            typer.echo(f"ready=false")
            typer.echo(f"error={result['error']}", err=True)
        raise typer.Exit(code=1) from error
    result = {
        "base_url": status.base_url,
        "reachable": status.reachable,
        "model": _model_summary(model),
        "capabilities": capabilities,
        "ready": True,
    }
    if as_json:
        _emit_json(result)
    else:
        typer.echo("ready=true")
        typer.echo(f"base_url={status.base_url}")
        typer.echo(f"model={model.name}")
        if capabilities:
            typer.echo("capabilities=" + ",".join(capabilities))


def register_ollama_commands(app: typer.Typer) -> None:
    app.add_typer(ollama_app, name="ollama")


__all__ = ["ollama_app", "register_ollama_commands"]
