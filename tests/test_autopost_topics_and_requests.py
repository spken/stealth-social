"""Pure autopost topic and request-construction tests."""

from __future__ import annotations

import unittest
from datetime import UTC, datetime

from bot.autopost.requests import build_autopost_request
from bot.autopost.topics import select_campaign_topic
from bot.content.models import GenerationType
from tests.support import make_generation_request, make_settings


class CampaignTopicSelectionTests(unittest.TestCase):
    def test_never_used_topic_wins_in_configuration_order(self) -> None:
        used = make_generation_request(topic="First topic")
        self.assertEqual(
            select_campaign_topic(("First topic", "Second topic"), (used,)),
            "Second topic",
        )

    def test_oldest_attempted_topic_wins_after_full_cycle(self) -> None:
        first = make_generation_request(
            topic="First topic",
            created_at=datetime(2030, 1, 1, tzinfo=UTC),
        )
        second = make_generation_request(
            topic="Second topic",
            created_at=datetime(2030, 1, 2, tzinfo=UTC),
        )
        self.assertEqual(
            select_campaign_topic(("First topic", "Second topic"), (second, first)),
            "First topic",
        )

    def test_removed_historical_topic_is_ignored(self) -> None:
        removed = make_generation_request(topic="Removed topic")
        self.assertEqual(
            select_campaign_topic(("Current topic",), (removed,)),
            "Current topic",
        )


class AutopostRequestTests(unittest.TestCase):
    def test_x_campaign_builds_immediate_unattended_request(self) -> None:
        settings = make_settings()
        campaign = settings.autopost_campaigns["daily-x"]
        request = build_autopost_request(
            settings, "daily-x", campaign, "First topic"
        )
        self.assertEqual(request.generation_type, GenerationType.X_POST)
        self.assertEqual(request.campaign_id, "daily-x")
        self.assertTrue(request.unattended_approval_requested)
        self.assertIsNone(request.desired_generation_time)
        self.assertEqual(request.account_context.account_name, "main")

    def test_reddit_campaign_builds_reddit_post_request(self) -> None:
        settings = make_settings()
        campaign = settings.autopost_campaigns["weekly-reddit"]
        request = build_autopost_request(
            settings, "weekly-reddit", campaign, "Reddit topic one"
        )
        self.assertEqual(request.generation_type, GenerationType.REDDIT_POST)
        self.assertEqual(request.subreddit, "SideProject")
