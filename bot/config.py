"""Typed application configuration with file and environment loading."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Annotated, Any, Self

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveInt,
    StringConstraints,
    ValidationError,
    field_validator,
    model_validator,
)
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    SettingsError,
)
from sqlalchemy.engine import make_url
from sqlalchemy.exc import ArgumentError

CONFIG_PATH_ENVIRONMENT_VARIABLE = "STEALTH_BOT_CONFIG"

NonEmptyString = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]


class ConfigurationError(ValueError):
    """Raised when configuration cannot be loaded safely."""


class _ConfigModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
        validate_default=True,
    )


class RandomizedDelaySettings(_ConfigModel):
    """Jitter applied when scheduling an otherwise eligible action."""

    minimum_seconds: NonNegativeFloat = 5.0
    maximum_seconds: NonNegativeFloat = 30.0

    @model_validator(mode="after")
    def validate_range(self) -> Self:
        if self.maximum_seconds < self.minimum_seconds:
            raise ValueError(
                "maximum_seconds must be greater than or equal to minimum_seconds"
            )
        return self


class SafetyLimitOverrides(_ConfigModel):
    """Optional safety-limit overrides for one platform or account."""

    minimum_seconds_between_actions: NonNegativeFloat | None = None
    maximum_actions_per_hour: PositiveInt | None = None
    maximum_actions_per_day: PositiveInt | None = None
    duplicate_window_hours: NonNegativeInt | None = None
    failure_threshold: PositiveInt | None = None


class PlatformLimitOverrides(_ConfigModel):
    """Platform-specific overrides layered over the global limits."""

    x: SafetyLimitOverrides | None = None
    reddit: SafetyLimitOverrides | None = None


class SafetyLimits(_ConfigModel):
    """Global action limits and optional platform-level overrides."""

    minimum_seconds_between_actions: NonNegativeFloat = 120.0
    maximum_actions_per_hour: PositiveInt = 5
    maximum_actions_per_day: PositiveInt = 20
    duplicate_window_hours: NonNegativeInt = 168
    failure_threshold: PositiveInt = 3
    platforms: PlatformLimitOverrides = Field(default_factory=PlatformLimitOverrides)


class BrowserSettings(_ConfigModel):
    """Browser runtime paths and visibility settings."""

    headless: bool = False
    screenshots_directory: Path = Path("data/screenshots")
    sessions_directory: Path = Path("data/sessions")


class AccountSettings(_ConfigModel):
    """Configuration shared by every browser-backed account."""

    session_profile: NonEmptyString
    enabled: bool = True
    limits: SafetyLimitOverrides | None = None


class XAccountSettings(AccountSettings):
    """Configuration for one X account."""


class RedditAccountSettings(AccountSettings):
    """Configuration for one Reddit account."""

    allowed_subreddits: list[NonEmptyString] = Field(default_factory=list)


class AccountsSettings(_ConfigModel):
    """Named account maps for supported platforms."""

    x: dict[NonEmptyString, XAccountSettings] = Field(default_factory=dict)
    reddit: dict[NonEmptyString, RedditAccountSettings] = Field(default_factory=dict)


class Settings(BaseSettings):
    """Validated settings assembled from defaults, a file, and environment."""

    database_url: NonEmptyString = "sqlite:///data/stealth.db"
    dry_run: bool = True
    manual_approval: bool = True
    global_pause: bool = False
    randomized_delay: RandomizedDelaySettings = Field(
        default_factory=RandomizedDelaySettings
    )
    browser: BrowserSettings = Field(default_factory=BrowserSettings)
    limits: SafetyLimits = Field(default_factory=SafetyLimits)
    accounts: AccountsSettings = Field(default_factory=AccountsSettings)

    model_config = SettingsConfigDict(
        env_prefix="STEALTH_BOT_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
        frozen=True,
        nested_model_default_partial_update=True,
        str_strip_whitespace=True,
        validate_default=True,
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Give environment values precedence over configuration-file values."""
        return (
            env_settings,
            dotenv_settings,
            init_settings,
            file_secret_settings,
        )

    @field_validator("database_url")
    @classmethod
    def validate_database_url(cls, value: str) -> str:
        try:
            url = make_url(value)
        except ArgumentError as error:
            raise ValueError(
                "database_url must be a valid SQLAlchemy database URL"
            ) from error

        if url.drivername not in {"sqlite", "sqlite+aiosqlite"}:
            raise ValueError(
                "database_url must use sqlite or sqlite+aiosqlite"
            )
        if any(
            component is not None
            for component in (url.username, url.password, url.host, url.port)
        ):
            raise ValueError(
                "database_url must not include username, password, host, or port "
                "for SQLite"
            )
        return value

    @model_validator(mode="after")
    def validate_unique_session_profiles(self) -> Self:
        owners: dict[str, str] = {}
        account_maps = (
            ("x", self.accounts.x),
            ("reddit", self.accounts.reddit),
        )
        for platform, accounts in account_maps:
            for account_name, account in accounts.items():
                owner = f"accounts.{platform}.{account_name}"
                profile_key = account.session_profile.casefold()
                if previous_owner := owners.get(profile_key):
                    raise ValueError(
                        f"{owner}.session_profile duplicates "
                        f"{previous_owner}.session_profile; session profiles must be unique"
                    )
                owners[profile_key] = owner
        return self


def load_settings(config_path: str | Path | None = None) -> Settings:
    """Load JSON or YAML settings, then apply dotenv and environment overrides.

    An explicit ``config_path`` takes precedence over ``STEALTH_BOT_CONFIG``.
    Relative configuration paths are interpreted from the current working
    directory. Relative browser data paths are interpreted from the selected
    configuration file's directory, or from the current working directory when
    no file is selected.
    """
    selected_path = _select_config_path(config_path)
    if selected_path is None:
        base_directory = Path.cwd().resolve(strict=False)
        file_values: dict[str, Any] = {}
        source_description = "environment and defaults"
    else:
        resolved_config_path = _resolve_path(selected_path, Path.cwd())
        file_values = _read_config_file(resolved_config_path)
        _reject_unknown_top_level_fields(file_values, resolved_config_path)
        base_directory = resolved_config_path.parent
        source_description = str(resolved_config_path)

    try:
        settings = Settings(
            _env_file=base_directory / ".env",
            **file_values,
        )
    except ValidationError as error:
        raise ConfigurationError(
            _format_validation_error(error, source_description)
        ) from error
    except SettingsError as error:
        raise ConfigurationError(
            f"Unable to load environment overrides for {source_description}: {error}"
        ) from error

    browser = settings.browser.model_copy(
        update={
            "screenshots_directory": _resolve_path(
                settings.browser.screenshots_directory, base_directory
            ),
            "sessions_directory": _resolve_path(
                settings.browser.sessions_directory, base_directory
            ),
        }
    )
    return settings.model_copy(update={"browser": browser})


def _select_config_path(config_path: str | Path | None) -> Path | None:
    if config_path is not None:
        if isinstance(config_path, str) and not config_path.strip():
            raise ConfigurationError("config_path cannot be empty")
        return Path(config_path).expanduser()

    environment_value = os.getenv(CONFIG_PATH_ENVIRONMENT_VARIABLE)
    if environment_value is None:
        return None
    if not environment_value.strip():
        raise ConfigurationError(
            f"{CONFIG_PATH_ENVIRONMENT_VARIABLE} cannot be empty; unset it to use defaults"
        )
    return Path(environment_value).expanduser()


def _read_config_file(config_path: Path) -> dict[str, Any]:
    suffix = config_path.suffix.casefold()
    if suffix not in {".json", ".yaml", ".yml"}:
        raise ConfigurationError(
            f"Unsupported configuration file extension {suffix or '<none>'!r} for "
            f"{config_path}; expected .json, .yaml, or .yml"
        )

    try:
        contents = config_path.read_text(encoding="utf-8")
    except FileNotFoundError as error:
        raise ConfigurationError(f"Configuration file not found: {config_path}") from error
    except OSError as error:
        raise ConfigurationError(
            f"Unable to read configuration file {config_path}: {error.strerror or error}"
        ) from error

    try:
        parsed = json.loads(contents) if suffix == ".json" else yaml.safe_load(contents)
    except json.JSONDecodeError as error:
        raise ConfigurationError(
            f"Invalid JSON in {config_path} at line {error.lineno}, "
            f"column {error.colno}: {error.msg}"
        ) from error
    except yaml.YAMLError as error:
        location = ""
        if error.problem_mark is not None:
            location = (
                f" at line {error.problem_mark.line + 1}, "
                f"column {error.problem_mark.column + 1}"
            )
        problem = getattr(error, "problem", None) or "could not parse YAML"
        raise ConfigurationError(
            f"Invalid YAML in {config_path}{location}: {problem}"
        ) from error

    if parsed is None:
        return {}
    if not isinstance(parsed, dict):
        raise ConfigurationError(
            f"Configuration file {config_path} must contain an object at its top level"
        )
    if any(not isinstance(key, str) for key in parsed):
        raise ConfigurationError(
            f"Configuration file {config_path} must use string keys at its top level"
        )
    return parsed


def _reject_unknown_top_level_fields(
    values: dict[str, Any], config_path: Path
) -> None:
    unknown_fields = sorted(set(values).difference(Settings.model_fields))
    if unknown_fields:
        joined_fields = ", ".join(unknown_fields)
        raise ConfigurationError(
            f"Unknown top-level configuration field(s) in {config_path}: {joined_fields}"
        )


def _resolve_path(path: str | Path, base_directory: Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = base_directory / candidate
    return candidate.resolve(strict=False)


def _format_validation_error(
    error: ValidationError, source_description: str
) -> str:
    details: list[str] = []
    for issue in error.errors():
        location = ".".join(str(part) for part in issue["loc"]) or "configuration"
        details.append(f"  - {location}: {issue['msg']}")
    return f"Invalid configuration from {source_description}:\n" + "\n".join(details)


__all__ = [
    "AccountSettings",
    "AccountsSettings",
    "BrowserSettings",
    "CONFIG_PATH_ENVIRONMENT_VARIABLE",
    "ConfigurationError",
    "PlatformLimitOverrides",
    "RandomizedDelaySettings",
    "RedditAccountSettings",
    "SafetyLimitOverrides",
    "SafetyLimits",
    "Settings",
    "XAccountSettings",
    "load_settings",
]
