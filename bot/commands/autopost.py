"""One-shot JSON-only autopost command."""

from __future__ import annotations

import asyncio
import logging
import sys
from typing import Annotated

import typer

from bot.autopost.models import AutopostOutcome, AutopostResult
from bot.autopost.runtime import run_autopost
from bot.commands.common import _emit_json, _run_async, _settings
from bot.config import ConfigurationError

logger = logging.getLogger(__name__)

def _log_internal_error(error: Exception) -> None:
    if not logger.handlers:
        handler = logging.StreamHandler(sys.__stderr__)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.propagate = False
    logger.error("autopost_command_failed error_type=%s", type(error).__name__)


def autopost_command(
    campaign_id: Annotated[
        str,
        typer.Argument(help="Configured autopost campaign ID."),
    ],
) -> None:
    """Run one configured campaign occurrence."""

    try:
        settings = _settings()
        pending = run_autopost(settings, campaign_id)
        try:
            result = _run_async(pending)
        except BaseException:
            close = getattr(pending, "close", None)
            if callable(close):
                close()
            raise
    except ConfigurationError:
        result = AutopostResult(
            campaign_id=campaign_id,
            outcome=AutopostOutcome.CONFIGURATION_ERROR,
            attention_reason="invalid_configuration",
        )
    except (KeyboardInterrupt, asyncio.CancelledError):
        result = AutopostResult(
            campaign_id=campaign_id,
            outcome=AutopostOutcome.ATTENTION_REQUIRED,
            attention_reason="interrupted",
        )
        _emit_json(result)
        raise typer.Exit(code=130) from None
    except Exception as error:
        _log_internal_error(error)
        result = AutopostResult(
            campaign_id=campaign_id,
            outcome=AutopostOutcome.ATTENTION_REQUIRED,
            attention_reason="internal_error",
        )

    _emit_json(result)
    if result.exit_code:
        raise typer.Exit(code=result.exit_code)


def register_autopost_command(app: typer.Typer) -> None:
    app.command("autopost")(autopost_command)


__all__ = ["autopost_command", "register_autopost_command"]
