"""Configuration tests for autopost campaigns."""

from __future__ import annotations

import unittest
from pathlib import Path

from pydantic import ValidationError

from bot.config import Settings, load_settings
from bot.content.models import ContentPurpose
from bot.models import Platform
from tests.support import settings_values


class AutopostConfigurationTests(unittest.TestCase):
    def test_valid_campaigns_are_typed(self) -> None:
        settings = Settings.model_validate(settings_values())
        self.assertEqual(settings.autopost_campaigns["daily-x"].platform, Platform.X)
        self.assertEqual(
            settings.autopost_campaigns["weekly-reddit"].purpose,
            ContentPurpose.BUILDER_UPDATE,
        )

    def test_checked_in_example_has_safe_autopost_campaigns(self) -> None:
        settings = load_settings(Path(__file__).parents[1] / "config.example.json")
        self.assertEqual(set(settings.autopost_campaigns), {"daily-x", "weekly-reddit"})
        self.assertTrue(settings.dry_run)
        self.assertFalse(settings.automation.allow_unattended_approval)
        self.assertFalse(settings.automation.allow_unattended_publishing)

    def test_duplicate_topics_are_case_insensitive(self) -> None:
        values = settings_values()
        values["autopost_campaigns"]["daily-x"]["topics"] = ["Same", " same "]
        with self.assertRaisesRegex(ValidationError, "topics must be unique"):
            Settings.model_validate(values)

    def test_campaign_id_must_be_systemd_safe(self) -> None:
        values = settings_values()
        values["autopost_campaigns"]["Daily X"] = values["autopost_campaigns"].pop(
            "daily-x"
        )
        with self.assertRaisesRegex(ValidationError, "campaign identifier"):
            Settings.model_validate(values)

    def test_campaign_account_must_exist_on_its_platform(self) -> None:
        values = settings_values()
        values["autopost_campaigns"]["daily-x"]["account"] = "missing"
        with self.assertRaisesRegex(ValidationError, "enabled X account"):
            Settings.model_validate(values)

    def test_reddit_subreddit_must_be_allowlisted(self) -> None:
        values = settings_values()
        values["autopost_campaigns"]["weekly-reddit"]["subreddit"] = "Python"
        with self.assertRaisesRegex(ValidationError, "allowlisted"):
            Settings.model_validate(values)

    def test_x_campaign_rejects_subreddit(self) -> None:
        values = settings_values()
        values["autopost_campaigns"]["daily-x"]["subreddit"] = "SideProject"
        with self.assertRaisesRegex(ValidationError, "X campaign"):
            Settings.model_validate(values)

    def test_promotional_reddit_requires_explicit_permission(self) -> None:
        values = settings_values()
        values["autopost_campaigns"]["weekly-reddit"]["purpose"] = "promotional"
        with self.assertRaisesRegex(ValidationError, "promotional"):
            Settings.model_validate(values)
