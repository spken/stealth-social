"""Stable result and cancellation-safe autopost lock tests."""

from __future__ import annotations

import asyncio
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from bot.autopost.locking import (
    AutopostLockBusyError,
    autopost_lock_path,
    hold_autopost_lock,
)
from bot.autopost.models import AutopostOutcome, AutopostResult


class AutopostResultTests(unittest.TestCase):
    def test_exit_codes_are_stable(self) -> None:
        expected = {
            AutopostOutcome.PUBLISHED: 0,
            AutopostOutcome.SKIPPED_RECENT_SUCCESS: 0,
            AutopostOutcome.CONFIGURATION_ERROR: 2,
            AutopostOutcome.ATTENTION_REQUIRED: 3,
            AutopostOutcome.TEMPORARY_FAILURE: 75,
        }
        for outcome, code in expected.items():
            with self.subTest(outcome=outcome):
                self.assertEqual(
                    AutopostResult(campaign_id="daily-x", outcome=outcome).exit_code,
                    code,
                )


class AutopostLockTests(unittest.IsolatedAsyncioTestCase):
    async def test_file_database_derives_adjacent_lock_path(self) -> None:
        self.assertEqual(
            autopost_lock_path("sqlite:///data/stealth.db"),
            (Path.cwd() / "data" / "stealth.db.autopost.lock").resolve(),
        )

    async def test_in_memory_database_requires_explicit_path(self) -> None:
        with self.assertRaisesRegex(ValueError, "file-backed"):
            autopost_lock_path("sqlite+aiosqlite:///:memory:")

    async def test_explicit_path_does_not_allow_non_sqlite_database(self) -> None:
        with TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "SQLite"):
                autopost_lock_path(
                    "postgresql://localhost/social",
                    explicit_path=Path(temp_dir) / "autopost.lock",
                )

    async def test_second_acquisition_times_out(self) -> None:
        with TemporaryDirectory() as temp_dir:
            lock_path = Path(temp_dir) / "autopost.lock"
            async with hold_autopost_lock(
                "sqlite+aiosqlite:///:memory:",
                explicit_path=lock_path,
            ):
                with self.assertRaises(AutopostLockBusyError):
                    async with hold_autopost_lock(
                        "sqlite+aiosqlite:///:memory:",
                        explicit_path=lock_path,
                        timeout_seconds=0.05,
                        poll_seconds=0.01,
                    ):
                        self.fail("the contended lock was acquired")

    async def test_exception_releases_lock_for_reacquisition(self) -> None:
        with TemporaryDirectory() as temp_dir:
            lock_path = Path(temp_dir) / "autopost.lock"
            with self.assertRaisesRegex(RuntimeError, "boom"):
                async with hold_autopost_lock(
                    "sqlite+aiosqlite:///:memory:",
                    explicit_path=lock_path,
                ):
                    raise RuntimeError("boom")
            async with hold_autopost_lock(
                "sqlite+aiosqlite:///:memory:",
                explicit_path=lock_path,
                timeout_seconds=0.05,
            ):
                pass

    async def test_cancelled_wait_does_not_strand_lock(self) -> None:
        with TemporaryDirectory() as temp_dir:
            lock_path = Path(temp_dir) / "autopost.lock"
            async with hold_autopost_lock(
                "sqlite+aiosqlite:///:memory:",
                explicit_path=lock_path,
            ):
                waiting = asyncio.create_task(
                    hold_autopost_lock(
                        "sqlite+aiosqlite:///:memory:",
                        explicit_path=lock_path,
                        timeout_seconds=5,
                        poll_seconds=0.01,
                    ).__aenter__()
                )
                await asyncio.sleep(0.03)
                waiting.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await waiting
