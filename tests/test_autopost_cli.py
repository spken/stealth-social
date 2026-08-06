"""JSON-only autopost CLI contract tests."""

from __future__ import annotations

import json
import unittest
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from bot.autopost.models import AutopostOutcome, AutopostResult
from bot.cli import app
from bot.config import ConfigurationError


class AutopostCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = CliRunner()

    def invoke_with_result(self, result: AutopostResult):
        with (
            patch("bot.commands.autopost._settings", return_value=MagicMock()),
            patch("bot.commands.autopost.run_autopost", new=MagicMock(return_value=MagicMock())),
            patch("bot.commands.autopost._run_async", return_value=result),
        ):
            return self.runner.invoke(app, ["autopost", "daily-x"])

    def test_help_requires_campaign_id_without_safety_overrides(self) -> None:
        result = self.runner.invoke(app, ["autopost", "--help"])
        self.assertEqual(result.exit_code, 0)
        self.assertIn("CAMPAIGN_ID", result.stdout)
        self.assertNotIn("--topic", result.stdout)
        self.assertNotIn("--account", result.stdout)
        self.assertNotIn("--bypass", result.stdout)

    def test_published_result_is_one_json_object_with_success_exit(self) -> None:
        result = self.invoke_with_result(
            AutopostResult(
                campaign_id="daily-x",
                outcome=AutopostOutcome.PUBLISHED,
            )
        )
        self.assertEqual(result.exit_code, 0)
        self.assertEqual(json.loads(result.stdout)["outcome"], "published")

    def test_configuration_result_exits_two(self) -> None:
        result = self.invoke_with_result(
            AutopostResult(
                campaign_id="daily-x",
                outcome=AutopostOutcome.CONFIGURATION_ERROR,
                attention_reason="dry_run_enabled",
            )
        )
        self.assertEqual(result.exit_code, 2)

    def test_attention_result_exits_three(self) -> None:
        result = self.invoke_with_result(
            AutopostResult(
                campaign_id="daily-x",
                outcome=AutopostOutcome.ATTENTION_REQUIRED,
                attention_reason="no_safe_candidate",
            )
        )
        self.assertEqual(result.exit_code, 3)

    def test_temporary_result_exits_seventy_five(self) -> None:
        result = self.invoke_with_result(
            AutopostResult(
                campaign_id="daily-x",
                outcome=AutopostOutcome.TEMPORARY_FAILURE,
                attention_reason="autopost_lock_busy",
            )
        )
        self.assertEqual(result.exit_code, 75)

    def test_configuration_error_before_runtime_is_json(self) -> None:
        with (
            patch(
                "bot.commands.autopost._settings",
                side_effect=ConfigurationError("secret configuration detail"),
            ),
            patch("bot.commands.autopost.run_autopost") as run_autopost,
        ):
            result = self.runner.invoke(app, ["autopost", "daily-x"])
        self.assertEqual(result.exit_code, 2)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["outcome"], "configuration_error")
        run_autopost.assert_not_called()

    def test_unexpected_error_is_bounded_json(self) -> None:
        with (
            patch("bot.commands.autopost._settings", return_value=MagicMock()),
            patch(
                "bot.commands.autopost._run_async",
                side_effect=RuntimeError("secret prompt content"),
            ),
        ):
            result = self.runner.invoke(app, ["autopost", "daily-x"])
        self.assertEqual(result.exit_code, 3)
        payload = json.loads(result.stdout)
        self.assertEqual(payload["outcome"], "attention_required")
        self.assertEqual(payload["attention_reason"], "internal_error")
        self.assertNotIn("secret", result.stdout)
