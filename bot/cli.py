"""Installable Typer application assembled from focused command modules."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from bot.commands.actions import register_action_commands
from bot.commands.autopost import register_autopost_command
from bot.commands.auth import register_auth_commands
from bot.commands.candidates import register_candidate_commands
from bot.commands.common import CliState, LogLevel, _configure_logging
from bot.commands.examples import register_example_commands
from bot.commands.generate import register_generation_commands
from bot.commands.ollama import register_ollama_commands
from bot.commands.topics import register_topic_commands
from bot.commands.worker import register_worker_command

app = typer.Typer(
    name="social-bot",
    help="Create, review, schedule, and safely execute social actions.",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
    rich_markup_mode=None,
    context_settings={"help_option_names": ["-h", "--help"]},
)


@app.callback()
def root_callback(
    ctx: typer.Context,
    config: Annotated[
        Path | None,
        typer.Option(
            "--config",
            envvar="STEALTH_BOT_CONFIG",
            help="JSON or YAML configuration file.",
            dir_okay=False,
            readable=True,
            resolve_path=True,
        ),
    ] = None,
    log_level: Annotated[
        LogLevel,
        typer.Option("--log-level", help="Structured application log threshold."),
    ] = LogLevel.INFO,
    json_logs: Annotated[
        bool,
        typer.Option("--json-logs/--console-logs", help="Emit structured logs as JSON."),
    ] = False,
) -> None:
    """Initialize command context without opening runtime resources."""

    _configure_logging(log_level, json_logs=json_logs)
    ctx.obj = CliState(config_path=config, log_level=log_level, json_logs=json_logs)


register_auth_commands(app)
register_action_commands(app)
register_autopost_command(app)
register_ollama_commands(app)
register_example_commands(app)
register_generation_commands(app)
register_candidate_commands(app)
register_topic_commands(app)
register_worker_command(app)


def main() -> None:
    """Run the installed command-line application."""

    app(prog_name="social-bot")


__all__ = ["app", "main", "root_callback"]
