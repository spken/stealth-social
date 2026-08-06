"""Shared helpers for building configured account contexts."""

from __future__ import annotations

from bot.config import Settings
from bot.content.models import AccountContext
from bot.models import Platform


def configured_account_context(
    settings: Settings,
    platform: Platform,
    account_name: str,
) -> AccountContext:
    """Build the generation context for one enabled configured account."""

    platform = Platform(platform)
    accounts = settings.accounts.x if platform is Platform.X else settings.accounts.reddit
    account = accounts.get(account_name)
    if account is None or not account.enabled:
        raise ValueError(
            f"no enabled {platform.value} account named {account_name!r}"
        )
    return AccountContext(
        account_name=account_name,
        platform=platform,
        identity=account.identity,
        products=tuple(account.products),
        verified_facts=tuple(account.verified_facts),
        forbidden_claims=tuple(account.forbidden_claims),
        required_disclosures=tuple(account.required_disclosures),
    )


__all__ = ["configured_account_context"]
