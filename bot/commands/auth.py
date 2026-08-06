"""Interactive browser authentication command."""

from __future__ import annotations

import asyncio
import signal
import threading
from typing import Annotated

import typer

from bot.browser.manager import BrowserManager
from bot.browser.sessions import SessionStatus, classify_auth_state
from bot.config import Settings
from bot.models import Platform
from bot.storage.database import Database

from bot.commands.common import (
    CliInputError,
    _emit_json,
    _interactive_terminal,
    _run_async,
    _safe_command,
    _settings,
    _validated_account,
)

_LOGIN_URLS = {
    Platform.X: "https://x.com/i/flow/login",
    Platform.REDDIT: "https://www.reddit.com/login/",
}


async def _wait_for_login_confirmation() -> None:
    loop = asyncio.get_running_loop()
    confirmation: asyncio.Future[BaseException | None] = loop.create_future()

    def deliver(outcome: BaseException | None) -> None:
        if not confirmation.done():
            confirmation.set_result(outcome)

    def read_confirmation() -> None:
        try:
            input()
        except BaseException as error:
            outcome: BaseException | None = error
        else:
            outcome = None
        try:
            loop.call_soon_threadsafe(deliver, outcome)
        except RuntimeError:
            return

    login_task = asyncio.current_task()
    sigterm_installed = False
    if login_task is not None:
        try:
            loop.add_signal_handler(signal.SIGTERM, login_task.cancel)
        except (NotImplementedError, RuntimeError, ValueError):
            pass
        else:
            sigterm_installed = True
    try:
        threading.Thread(
            target=read_confirmation,
            name="social-bot-login-input",
            daemon=True,
        ).start()
        outcome = await confirmation
    finally:
        if sigterm_installed:
            loop.remove_signal_handler(signal.SIGTERM)
    if outcome is None:
        return
    if isinstance(outcome, EOFError):
        raise CliInputError("Login confirmation requires an interactive terminal") from outcome
    raise outcome


async def _interactive_login(
    settings: Settings,
    platform: Platform,
    account_name: str,
) -> SessionStatus:
    browser_manager = BrowserManager(settings)
    database = Database(settings.database_url)
    try:
        await database.initialize()
        async with browser_manager.interactive_login(
            platform, account_name, _LOGIN_URLS[platform]
        ) as browser_session:
            typer.echo(
                "A headed browser is open. Complete login or any required human challenge, then press Enter here.",
                err=True,
            )
            await _wait_for_login_confirmation()
            return await classify_auth_state(browser_session.page, platform)
    finally:
        try:
            await browser_manager.shutdown()
        finally:
            await database.close()


@_safe_command
def login_command(
    platform: Annotated[Platform, typer.Argument(help="Platform: x or reddit.")],
    account: Annotated[str, typer.Option("--account", help="Configured account name.")],
) -> None:
    """Open a headed browser for manual, credential-free login."""

    if not _interactive_terminal():
        raise CliInputError("login requires an interactive terminal")
    settings = _settings()
    account_name = _validated_account(settings, platform, account)
    status = _run_async(_interactive_login(settings, platform, account_name))
    next_steps = {
        SessionStatus.AUTHENTICATED: "Session is authenticated.",
        SessionStatus.AUTH_REQUIRED: "Authentication was not confirmed; rerun login to continue.",
        SessionStatus.CHALLENGE_REQUIRED: "A human challenge remains; resolve it and rerun login.",
        SessionStatus.CLOSED: "The browser closed before authentication could be confirmed.",
        SessionStatus.UNKNOWN: "Authentication state could not be confirmed; rerun login.",
    }
    _emit_json(
        {
            "platform": platform.value,
            "account_name": account_name,
            "status": status.value,
            "authenticated": status is SessionStatus.AUTHENTICATED,
            "next_step": next_steps[status],
        }
    )
    if status is not SessionStatus.AUTHENTICATED:
        raise typer.Exit(code=1)


def register_auth_commands(app: typer.Typer) -> None:
    app.command("login")(login_command)


__all__ = ["login_command", "register_auth_commands"]
