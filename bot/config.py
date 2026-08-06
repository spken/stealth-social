"""Typed application configuration with file and environment loading."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Annotated, Any, Literal, Self

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    AnyHttpUrl,
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
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
from yaml.error import MarkedYAMLError

from bot.content.models import ContentPurpose, GenerationType, RankingMode
from bot.models import Platform

CONFIG_PATH_ENVIRONMENT_VARIABLE = "STEALTH_BOT_CONFIG"

NonEmptyString = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]

_AUTOPOST_CAMPAIGN_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


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


class ContentGenerationSettings(_ConfigModel):
    """Resolved defaults for the local structured generator."""

    provider: Literal["ollama"] = "ollama"
    enabled: bool = True
    base_url: AnyHttpUrl = AnyHttpUrl("http://localhost:11434")
    model: NonEmptyString = "qwen3:8b"
    thinking: bool = False
    request_timeout_seconds: PositiveFloat = 180.0
    maximum_retries: NonNegativeInt = 2
    temperature: float = Field(default=0.75, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, gt=0.0, le=1.0)
    candidate_count: int = Field(default=3, ge=1, le=10)
    maximum_context_examples: int = Field(default=8, ge=0, le=50)
    maximum_example_characters: int = Field(default=12000, ge=0, le=100000)
    ranking_mode: RankingMode = RankingMode.HEURISTIC
    debug_prompt_logging: bool = False
    allow_generated_style_examples: bool = False

    @model_validator(mode="after")
    def validate_base_url(self) -> Self:
        if self.base_url.username or self.base_url.password:
            raise ValueError("content_generation.base_url must not contain credentials")
        if self.base_url.query or self.base_url.fragment:
            raise ValueError(
                "content_generation.base_url must not contain a query or fragment"
            )
        return self


class GenerationProfileSettings(_ConfigModel):
    """Optional per-generation-type overrides layered over global settings."""

    content_purpose: ContentPurpose | None = None
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    candidate_count: int | None = Field(default=None, ge=1, le=10)
    maximum_context_examples: int | None = Field(default=None, ge=0, le=50)
    maximum_example_characters: int | None = Field(default=None, ge=0, le=100000)


class XCollectionSettings(_ConfigModel):
    """Explicit public X source configuration."""

    account: NonEmptyString | None = None
    accounts: tuple[NonEmptyString, ...] = ()
    queries: tuple[NonEmptyString, ...] = ()
    post_urls: tuple[NonEmptyString, ...] = ()


class RedditCollectionSettings(_ConfigModel):
    """Explicit public Reddit source configuration."""

    account: NonEmptyString | None = None
    subreddits: tuple[NonEmptyString, ...] = ()
    queries: tuple[NonEmptyString, ...] = ()
    post_urls: tuple[NonEmptyString, ...] = ()
    sort: Literal["hot", "new", "top"] = "top"
    time_filter: Literal["hour", "day", "week", "month", "year", "all"] = "month"


class ExampleCollectionSettings(_ConfigModel):
    """Bounded browser-only collection settings."""

    enabled: bool = True
    maximum_items_per_source: int = Field(default=25, ge=1, le=500)
    maximum_comments_per_post: int = Field(default=20, ge=0, le=200)
    refresh_interval_hours: PositiveFloat = 24.0
    expiry_interval_hours: PositiveFloat = 168.0
    minimum_score: int = 5
    include_own_content: bool = True
    useful_window_days: PositiveInt = 90
    x: XCollectionSettings = Field(default_factory=XCollectionSettings)
    reddit: RedditCollectionSettings = Field(default_factory=RedditCollectionSettings)


class AutomationSettings(_ConfigModel):
    """Explicit capabilities for scheduled and unattended workflows."""

    allow_scheduled_generation: bool = True
    allow_unattended_approval: bool = False
    allow_unattended_publishing: bool = False


class SubredditContentRulesSettings(_ConfigModel):
    """Destination-specific Reddit promotion and formatting policy."""

    allow_promotional_content: bool = False
    required_disclosures: tuple[NonEmptyString, ...] = ()
    forbidden_phrases: tuple[NonEmptyString, ...] = ()
    maximum_title_characters: PositiveInt | None = None
    maximum_body_characters: PositiveInt | None = None


class AccountSettings(_ConfigModel):
    """Configuration shared by every browser-backed account."""

    session_profile: NonEmptyString
    enabled: bool = True
    limits: SafetyLimitOverrides | None = None
    identity: str | None = None
    products: tuple[NonEmptyString, ...] = ()
    verified_facts: tuple[NonEmptyString, ...] = ()
    forbidden_claims: tuple[NonEmptyString, ...] = ()
    required_disclosures: tuple[NonEmptyString, ...] = ()


class XAccountSettings(AccountSettings):
    """Configuration for one X account."""


class RedditAccountSettings(AccountSettings):
    """Configuration for one Reddit account."""

    allowed_subreddits: list[NonEmptyString] = Field(default_factory=list)
    community_rules: dict[str, SubredditContentRulesSettings] = Field(
        default_factory=dict
    )


class AccountsSettings(_ConfigModel):
    """Named account maps for supported platforms."""

    x: dict[NonEmptyString, XAccountSettings] = Field(default_factory=dict)
    reddit: dict[NonEmptyString, RedditAccountSettings] = Field(default_factory=dict)


class AutopostCampaignSettings(_ConfigModel):
    """One externally scheduled original-post campaign."""

    enabled: bool = True
    platform: Platform
    account: NonEmptyString
    topics: tuple[NonEmptyString, ...]
    minimum_interval_hours: PositiveFloat
    subreddit: NonEmptyString | None = None
    purpose: ContentPurpose = ContentPurpose.EDUCATIONAL
    goal: NonEmptyString | None = None
    product_context: NonEmptyString | None = None
    project_context: NonEmptyString | None = None
    target_audience: NonEmptyString | None = None
    tone: NonEmptyString | None = None
    desired_length: NonEmptyString | None = None
    call_to_action: NonEmptyString | None = None
    required_facts: tuple[NonEmptyString, ...] = ()
    forbidden_claims: tuple[NonEmptyString, ...] = ()
    forbidden_phrases: tuple[NonEmptyString, ...] = ()
    keywords: tuple[NonEmptyString, ...] = ()
    additional_instructions: NonEmptyString | None = None
    candidate_count: int | None = Field(default=None, ge=1, le=10)
    profile_name: NonEmptyString | None = None

    @field_validator("topics")
    @classmethod
    def validate_topics(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("topics must contain at least one topic")
        normalized = [item.casefold() for item in value]
        if len(normalized) != len(set(normalized)):
            raise ValueError("topics must be unique ignoring case")
        return value

    @field_validator("purpose", mode="before")
    @classmethod
    def normalize_purpose(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip().casefold().replace("-", "_")
        return value

    @model_validator(mode="after")
    def validate_destination(self) -> Self:
        if self.platform is Platform.X and self.subreddit is not None:
            raise ValueError("an X campaign cannot configure subreddit")
        if self.platform is Platform.REDDIT and self.subreddit is None:
            raise ValueError("a Reddit campaign requires subreddit")
        return self




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
    content_generation: ContentGenerationSettings = Field(
        default_factory=ContentGenerationSettings
    )
    generation_profiles: dict[GenerationType, GenerationProfileSettings] = Field(
        default_factory=dict
    )
    example_collection: ExampleCollectionSettings = Field(
        default_factory=ExampleCollectionSettings
    )
    automation: AutomationSettings = Field(default_factory=AutomationSettings)
    autopost_campaigns: dict[NonEmptyString, AutopostCampaignSettings] = Field(
        default_factory=dict
    )

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

    @model_validator(mode="after")
    def validate_generation_and_collection_accounts(self) -> Self:
        if not self.manual_approval and not self.automation.allow_unattended_approval:
            raise ValueError(
                "manual_approval=false requires automation.allow_unattended_approval=true"
            )

        reddit_accounts = self.accounts.reddit
        for account_name, account in reddit_accounts.items():
            allowed = {item.casefold() for item in account.allowed_subreddits}
            invalid_rules = sorted(
                subreddit
                for subreddit in account.community_rules
                if subreddit.casefold() not in allowed
            )
            if invalid_rules:
                raise ValueError(
                    f"accounts.reddit.{account_name}.community_rules keys must be "
                    "included in allowed_subreddits: "
                    + ", ".join(invalid_rules)
                )

        collection_x_account = self.example_collection.x.account
        if collection_x_account is not None:
            account = self.accounts.x.get(collection_x_account)
            if account is None or not account.enabled:
                raise ValueError(
                    "example_collection.x.account must name an enabled X account"
                )

        collection_reddit_account = self.example_collection.reddit.account
        if collection_reddit_account is not None:
            account = self.accounts.reddit.get(collection_reddit_account)
            if account is None or not account.enabled:
                raise ValueError(
                    "example_collection.reddit.account must name an enabled Reddit account"
                )
            allowed = {item.casefold() for item in account.allowed_subreddits}
            missing = [
                subreddit
                for subreddit in self.example_collection.reddit.subreddits
                if subreddit.casefold() not in allowed
            ]
            if missing:
                raise ValueError(
                    "configured Reddit collection subreddits are not allowlisted: "
                    + ", ".join(missing)
                )
        return self

    @model_validator(mode="after")
    def validate_autopost_campaigns(self) -> Self:
        for campaign_id, campaign in self.autopost_campaigns.items():
            if not _AUTOPOST_CAMPAIGN_ID.fullmatch(campaign_id):
                raise ValueError(
                    f"autopost campaign identifier {campaign_id!r} is not systemd-safe"
                )

            accounts = (
                self.accounts.x
                if campaign.platform is Platform.X
                else self.accounts.reddit
            )
            account = accounts.get(campaign.account)
            if account is None or not account.enabled:
                platform_name = "X" if campaign.platform is Platform.X else "Reddit"
                raise ValueError(
                    f"autopost campaign {campaign_id!r} requires an enabled "
                    f"{platform_name} account named {campaign.account!r}"
                )

            if campaign.platform is Platform.REDDIT:
                allowed = {
                    subreddit.casefold() for subreddit in account.allowed_subreddits
                }
                if campaign.subreddit is None:
                    raise ValueError(
                        f"autopost campaign {campaign_id!r} requires a Reddit subreddit"
                    )
                if campaign.subreddit.casefold() not in allowed:
                    raise ValueError(
                        f"autopost campaign {campaign_id!r} subreddit "
                        f"{campaign.subreddit!r} is not allowlisted"
                    )
                rule = next(
                    (
                        value
                        for name, value in account.community_rules.items()
                        if campaign.subreddit is not None
                        and name.casefold() == campaign.subreddit.casefold()
                    ),
                    None,
                )
                if campaign.purpose is ContentPurpose.PROMOTIONAL and (
                    rule is None or not rule.allow_promotional_content
                ):
                    raise ValueError(
                        f"autopost campaign {campaign_id!r} requires explicit "
                        "promotional Reddit permission"
                    )
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
        settings_values: dict[str, Any] = {
            "_env_file": base_directory / ".env",
            **file_values,
        }
        settings = Settings(**settings_values)
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
    except MarkedYAMLError as error:
        location = ""
        if error.problem_mark is not None:
            location = (
                f" at line {error.problem_mark.line + 1}, "
                f"column {error.problem_mark.column + 1}"
            )
        problem = error.problem or "could not parse YAML"
        raise ConfigurationError(
            f"Invalid YAML in {config_path}{location}: {problem}"
        ) from error
    except yaml.YAMLError as error:
        raise ConfigurationError(
            f"Invalid YAML in {config_path}: could not parse YAML"
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
    "AutopostCampaignSettings",
    "AccountsSettings",
    "AutomationSettings",
    "BrowserSettings",
    "CONFIG_PATH_ENVIRONMENT_VARIABLE",
    "ConfigurationError",
    "ContentGenerationSettings",
    "ExampleCollectionSettings",
    "GenerationProfileSettings",
    "PlatformLimitOverrides",
    "RandomizedDelaySettings",
    "RedditAccountSettings",
    "RedditCollectionSettings",
    "SafetyLimitOverrides",
    "SafetyLimits",
    "Settings",
    "SubredditContentRulesSettings",
    "XCollectionSettings",
    "XAccountSettings",
    "load_settings",
]
