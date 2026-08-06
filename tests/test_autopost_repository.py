"""Async repository tests for autopost history and claims."""

from __future__ import annotations

import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

from bot.content.models import CandidateDecision, GenerationStatus
from bot.storage.content_repository import ContentRepository
from bot.storage.database import Database
from bot.storage.repositories import ActionRepository
from tests.support import make_action, make_candidate, make_generation_request


class AutopostRepositoryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.temp_dir = TemporaryDirectory()
        db_path = Path(self.temp_dir.name) / "autopost.sqlite"
        self.database = Database(f"sqlite+aiosqlite:///{db_path.as_posix()}")
        await self.database.initialize()
        self.actions = ActionRepository(self.database.session_factory)
        self.content = ContentRepository(self.database.session_factory)

    async def asyncTearDown(self) -> None:
        await self.database.close()
        self.temp_dir.cleanup()

    async def test_campaign_requests_are_exact_and_newest_first(self) -> None:
        older = make_generation_request(
            campaign_id="daily-x",
            created_at=datetime(2030, 1, 1, tzinfo=UTC),
        )
        newer = make_generation_request(
            campaign_id="daily-x",
            created_at=datetime(2030, 1, 2, tzinfo=UTC),
        )
        other = make_generation_request(
            campaign_id="daily-x-extra",
            created_at=datetime(2030, 1, 3, tzinfo=UTC),
        )
        for request in (older, newer, other):
            await self.content.create_generation_request(request)

        records = await self.content.list_generation_requests_for_campaign("daily-x")

        self.assertEqual([item.id for item in records], [newer.id, older.id])
        self.assertEqual(
            await self.content.list_generation_requests_for_campaign(" daily-x "),
            [],
        )

    async def test_campaign_request_limit_must_be_positive(self) -> None:
        with self.assertRaisesRegex(ValueError, "limit must be positive"):
            await self.content.list_generation_requests_for_campaign(
                "daily-x", limit=0
            )

    async def test_approved_candidate_lookup_requires_approved_status(self) -> None:
        request = make_generation_request()
        await self.content.create_generation_request(request)
        action = make_action()
        await self.actions.create(action)
        candidate = make_candidate(request.id)
        await self.content.add_candidates(request.id, [candidate])

        async with self.content.transaction() as session:
            await self.content.approve_candidate_in_session(
                session,
                candidate.id,
                action.id,
                CandidateDecision(method="test"),
            )

        approved = await self.content.get_approved_candidate_for_request(request.id)
        self.assertIsNotNone(approved)
        self.assertEqual(approved.id, candidate.id)
        self.assertEqual(approved.approval_status.value, "approved")

    async def test_expired_processing_claim_is_atomically_adopted(self) -> None:
        created = datetime(2030, 1, 1, tzinfo=UTC)
        expired = created + timedelta(minutes=10)
        now = created + timedelta(hours=1)
        request = make_generation_request(
            status=GenerationStatus.PROCESSING,
            created_at=created,
            claim_owner="old-owner",
            claim_expires_at=expired,
            attempt_count=1,
        )
        await self.content.create_generation_request(request)

        claimed = await self.content.reclaim_expired_generation_request(
            request.id,
            owner="new-owner",
            lease_duration=timedelta(minutes=5),
            now=now,
        )

        self.assertIsNotNone(claimed)
        self.assertEqual(claimed.claim_owner, "new-owner")
        self.assertEqual(claimed.claim_expires_at, now + timedelta(minutes=5))
        self.assertEqual(claimed.attempt_count, 2)
        self.assertIsNone(claimed.next_retry_at)

    async def test_active_or_wrong_status_claim_is_not_adopted(self) -> None:
        active = make_generation_request(
            status=GenerationStatus.PROCESSING,
            claim_owner="active",
            claim_expires_at=datetime(2030, 1, 2, tzinfo=UTC),
        )
        failed = make_generation_request(
            status=GenerationStatus.FAILED,
            claim_owner="failed",
            claim_expires_at=datetime(2030, 1, 1, tzinfo=UTC),
        )
        await self.content.create_generation_request(active)
        await self.content.create_generation_request(failed)
        now = datetime(2030, 1, 1, 12, tzinfo=UTC)

        self.assertIsNone(
            await self.content.reclaim_expired_generation_request(
                active.id,
                owner="new",
                lease_duration=timedelta(minutes=5),
                now=now,
            )
        )
        self.assertIsNone(
            await self.content.reclaim_expired_generation_request(
                failed.id,
                owner="new",
                lease_duration=timedelta(minutes=5),
                now=now,
            )
        )
