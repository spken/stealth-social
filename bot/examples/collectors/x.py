"""X browser collector for configured public posts and visible replies."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from urllib.parse import quote, urlsplit

from playwright.async_api import Locator, Page

from bot.browser.manager import BrowserManager
from bot.config import Settings
from bot.examples.collectors.browser_common import (
    bounded_scroll,
    navigate_public_page,
    raise_for_collection_page,
    safe_attribute,
    safe_text,
    validate_public_url,
)
from bot.examples.models import (
    CollectedExample,
    ExampleCollectionRequest,
    ExampleTargetUnavailableError,
    ExampleType,
    TargetContext,
    TargetContextRequest,
)
from bot.examples.normalization import author_identifier, normalize_authored_text
from bot.models import Platform

_HANDLE = re.compile(r"^[A-Za-z0-9_]{1,15}$")
_STATUS_PATH = re.compile(r"^/(?P<user>[A-Za-z0-9_]{1,15})/status/(?P<id>[0-9]+)$")
_NUMBER = re.compile(r"(?P<number>[0-9]+(?:[.,][0-9]+)?)(?P<suffix>[kmb])?", re.IGNORECASE)


class XExampleCollector:
    """Read public X HTML through one configured browser profile."""

    def __init__(
        self,
        browser_manager: BrowserManager,
        settings: Settings,
        account_name: str | None = None,
    ) -> None:
        self._browser = browser_manager
        self._settings = settings
        self._account_name = account_name
        if account_name is not None:
            self._require_account(account_name)

    async def collect(self, request: ExampleCollectionRequest) -> list[CollectedExample]:
        if request.platform is not Platform.X:
            raise ValueError("X collector requires an X request")
        account_name = request.account_name
        if self._account_name is not None and account_name != self._account_name:
            raise ValueError("collection request account does not match collector account")
        self._require_account(account_name)
        routes = self.build_routes(request)
        if not routes:
            raise ValueError("X collection requires an explicit account, query, or post URL")
        examples: list[CollectedExample] = []
        seen: set[str] = set()
        explicit_routes = {
            validate_public_url(url, Platform.X) for url in request.post_urls
        }

        def add_items(items: list[CollectedExample]) -> None:
            for item in items:
                if item.external_id and item.external_id not in seen:
                    seen.add(item.external_id)
                    examples.append(item)

        async with self._browser.get_page(Platform.X, account_name) as page:
            for route in routes:
                route_items: list[CollectedExample] = []
                route_seen: set[str] = set()

                def add_route_items(items: list[CollectedExample]) -> None:
                    for item in items:
                        if item.external_id and item.external_id not in route_seen:
                            route_seen.add(item.external_id)
                            route_items.append(item)

                await navigate_public_page(page, route, Platform.X)
                add_route_items(
                    await self._extract_tweets(
                        page,
                        request,
                        collect_replies=route in explicit_routes,
                    )
                )
                if len(route_items) < request.maximum_items_per_source:
                    await bounded_scroll(
                        page,
                        item_ids=lambda: self._tweet_ids(page),
                        maximum_items=request.maximum_items_per_source,
                        maximum_scrolls=8,
                        stop_predicate=lambda: self._page_stopped(page),
                    )
                    await raise_for_collection_page(page, platform=Platform.X)
                    add_route_items(
                        await self._extract_tweets(
                            page,
                            request,
                            collect_replies=route in explicit_routes,
                        )
                    )
                add_items(route_items[: request.maximum_items_per_source])
        return examples

    async def resolve_target(self, request: TargetContextRequest) -> TargetContext:
        if request.platform is not Platform.X or request.target_url is None:
            raise ExampleTargetUnavailableError("X target URL is required")
        canonical = validate_public_url(request.target_url, Platform.X)
        status_id = _status_id(canonical)
        if status_id is None:
            raise ExampleTargetUnavailableError("X target status ID was not visible")
        self._require_account(request.account_name)
        async with self._browser.get_page(Platform.X, request.account_name) as page:
            await navigate_public_page(page, canonical, Platform.X)
            article = await self._find_status_article(page, status_id)
            if article is None:
                raise ExampleTargetUnavailableError("X target post was not visible")
            body = normalize_authored_text(
                await safe_text(article.locator('[data-testid="tweetText"]').first)
            )
            if not body:
                raise ExampleTargetUnavailableError("X target post had no visible text")
            metadata = await self._engagement_metadata(article)
            await raise_for_collection_page(page, platform=Platform.X)
            return TargetContext(
                platform=Platform.X,
                canonical_url=canonical,
                external_id=status_id,
                body=body,
                metadata=metadata,
            )

    def build_routes(self, request: ExampleCollectionRequest) -> tuple[str, ...]:
        routes: list[str] = []
        for handle in request.accounts:
            normalized = _normalize_handle(handle)
            routes.append(f"https://x.com/{normalized}")
        for query in request.queries:
            routes.append(f"https://x.com/search?q={quote(query)}&src=typed_query&f=live")
        for post_url in request.post_urls:
            routes.append(validate_public_url(post_url, Platform.X))
        return tuple(dict.fromkeys(routes))

    async def _extract_tweets(
        self,
        page: Page,
        request: ExampleCollectionRequest,
        *,
        collect_replies: bool,
    ) -> list[CollectedExample]:
        locator = page.locator('article[data-testid="tweet"]')
        results: list[CollectedExample] = []
        limit = request.maximum_items_per_source
        for index in range(min(await locator.count(), limit)):
            article = locator.nth(index)
            if await self._is_promoted(article):
                continue
            item = await self._tweet_example(article, request, content_type=ExampleType.X_POST)
            if item is not None:
                results.append(item)
            if len(results) >= limit:
                break
        if collect_replies:
            results.extend(await self._reply_examples(page, request))
        return results

    async def _tweet_example(
        self,
        article: Locator,
        request: ExampleCollectionRequest,
        *,
        content_type: ExampleType,
        parent_text: str | None = None,
    ) -> CollectedExample | None:
        body = normalize_authored_text(
            await safe_text(article.locator('[data-testid="tweetText"]').first)
        )
        status_href = await self._status_href(article)
        if not body or status_href is None:
            return None
        try:
            canonical = validate_public_url(status_href, Platform.X)
        except ValueError:
            return None
        external_id = _status_id(canonical)
        if external_id is None or _non_whitespace_length(body) < 20:
            return None
        handle = _handle_from_status_url(canonical)
        if handle is None:
            return None
        metadata = await self._engagement_metadata(article)
        if metadata["aggregate_engagement"] < request.minimum_score:
            return None
        return CollectedExample(
            platform=Platform.X,
            content_type=content_type,
            external_id=external_id,
            source_url=canonical,
            author_identifier=author_identifier(Platform.X, handle),
            body=body,
            parent_text=parent_text,
            published_at=_parse_datetime(
                await safe_attribute(article.locator("time[datetime]").first, "datetime")
            ),
            engagement_score=metadata["aggregate_engagement"],
            metadata=metadata,
        )

    async def _reply_examples(
        self,
        page: Page,
        request: ExampleCollectionRequest,
    ) -> list[CollectedExample]:
        conversation = page.locator('div[aria-label*="Conversation"] article[data-testid="tweet"]')
        if await conversation.count() == 0:
            return []
        root = conversation.first
        root_item = await self._tweet_example(root, request, content_type=ExampleType.X_POST)
        root_id = _status_id(root_item.source_url) if root_item else None
        root_text = root_item.body if root_item else ""
        results: list[CollectedExample] = []
        for index in range(min(await conversation.count(), request.maximum_items_per_source)):
            article = conversation.nth(index)
            status_href = await self._status_href(article)
            status_id = _status_id(status_href) if status_href else None
            if status_id is None or status_id == root_id:
                continue
            item = await self._tweet_example(
                article,
                request,
                content_type=ExampleType.X_REPLY,
                parent_text=root_text,
            )
            if item is not None:
                results.append(item)
        return results[: request.maximum_items_per_source]

    async def _status_href(self, article: Locator) -> str | None:
        links = article.locator('a[href*="/status/"]')
        for index in range(min(await links.count(), 20)):
            href = await safe_attribute(links.nth(index), "href")
            if href is not None and _status_id(href) is not None:
                return href
        return None

    async def _find_status_article(self, page: Page, status_id: str) -> Locator | None:
        locator = page.locator('article[data-testid="tweet"]')
        for index in range(min(await locator.count(), 100)):
            article = locator.nth(index)
            href = await self._status_href(article)
            if href is not None and _status_id(href) == status_id:
                return article
        return None

    async def _engagement_metadata(self, article: Locator) -> dict[str, int]:
        values: dict[str, int] = {}
        for action, selector in (
            ("replies", '[data-testid="reply"]'),
            ("reposts", '[data-testid="retweet"], [data-testid="unretweet"]'),
            ("likes", '[data-testid="like"], [data-testid="unlike"]'),
        ):
            label = await safe_attribute(article.locator(selector).first, "aria-label")
            values[action] = _parse_engagement(label)
        values["aggregate_engagement"] = min(
            10_000_000,
            values["replies"] + values["reposts"] + values["likes"],
        )
        return values

    async def _is_promoted(self, article: Locator) -> bool:
        for selector in (
            '[data-testid="placementTracking"]',
            '[data-testid="promotedIndicator"]',
            '[data-testid="socialContext"]',
        ):
            text = (await safe_text(article.locator(selector).first, maximum_length=200)).casefold()
            if "promoted" in text or "ad" == text.strip():
                return True
        return False

    async def _tweet_ids(self, page: Page) -> set[str]:
        locator = page.locator('article[data-testid="tweet"]')
        values: set[str] = set()
        for index in range(min(await locator.count(), 500)):
            href = await self._status_href(locator.nth(index))
            if href is not None and (status_id := _status_id(href)) is not None:
                values.add(status_id)
        return values

    async def _page_stopped(self, page: Page) -> bool:
        await raise_for_collection_page(page, platform=Platform.X)
        return False

    def _require_account(self, account_name: str):
        account = self._settings.accounts.x.get(account_name)
        if account is None:
            raise ValueError(f"no configured X account named {account_name!r}")
        if not account.enabled:
            raise ValueError(f"configured X account {account_name!r} is disabled")
        return account


def _normalize_handle(value: str) -> str:
    normalized = value.strip().lstrip("@")
    if not _HANDLE.fullmatch(normalized):
        raise ValueError("X public account handle is invalid")
    return normalized


def _status_id(value: str | None) -> str | None:
    if not value:
        return None
    try:
        parsed = urlsplit(value)
    except ValueError:
        return None
    match = _STATUS_PATH.fullmatch(parsed.path.rstrip("/"))
    return match.group("id") if match is not None else None


def _handle_from_status_url(value: str) -> str | None:
    try:
        match = _STATUS_PATH.fullmatch(urlsplit(value).path.rstrip("/"))
    except ValueError:
        return None
    return match.group("user") if match is not None else None


def _parse_engagement(value: str | None) -> int:
    if not value:
        return 0
    match = _NUMBER.search(value.replace(",", ""))
    if match is None:
        return 0
    amount = float(match.group("number"))
    multiplier = {"k": 1_000, "m": 1_000_000, "b": 1_000_000_000}.get(
        (match.group("suffix") or "").casefold(), 1
    )
    return max(0, min(int(amount * multiplier), 10_000_000))


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _non_whitespace_length(value: str) -> int:
    return sum(not character.isspace() for character in value)


__all__ = ["XExampleCollector"]
