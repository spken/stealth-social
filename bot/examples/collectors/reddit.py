"""Old-Reddit browser collector for configured public examples and targets."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime
from urllib.parse import quote, urljoin, urlsplit

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

_ORIGIN = "https://old.reddit.com"


class RedditExampleCollector:
    """Read public old-Reddit HTML through one configured browser profile."""

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
        if request.platform is not Platform.REDDIT:
            raise ValueError("Reddit collector requires a Reddit request")
        account_name = request.account_name
        if self._account_name is not None and account_name != self._account_name:
            raise ValueError("collection request account does not match collector account")
        account = self._require_account(account_name)
        allowed = {item.casefold() for item in account.allowed_subreddits}
        for subreddit in request.subreddits:
            if subreddit.casefold() not in allowed:
                raise ValueError(f"subreddit {subreddit!r} is not allowlisted")

        examples: list[CollectedExample] = []
        seen_ids: set[str] = set()
        routes = self.build_routes(request, allowed_subreddits=allowed)
        async with self._browser.get_page(Platform.REDDIT, account_name) as page:
            for route in routes:
                route_items: list[CollectedExample] = []
                route_seen: set[str] = set()

                def add_route_items(items: list[CollectedExample]) -> None:
                    for item in items:
                        if item.external_id and item.external_id not in route_seen:
                            route_seen.add(item.external_id)
                            route_items.append(item)

                await navigate_public_page(page, route, Platform.REDDIT)
                explicit_post = route in {
                    validate_public_url(url, Platform.REDDIT, target_kind="post")
                    for url in request.post_urls
                }
                add_route_items(await self._extract_posts(
                    page,
                    request,
                    allowed_subreddits=allowed,
                    include_comments=explicit_post,
                ))
                if len(route_items) < request.maximum_items_per_source:
                    await bounded_scroll(
                        page,
                        item_ids=lambda: self._post_ids(page),
                        maximum_items=request.maximum_items_per_source,
                        maximum_scrolls=8,
                        stop_predicate=lambda: self._page_stopped(page),
                    )
                    await raise_for_collection_page(page, platform=Platform.REDDIT)
                    add_route_items(
                        await self._extract_posts(
                            page,
                            request,
                            allowed_subreddits=allowed,
                            include_comments=False,
                        )
                    )
                for item in route_items[: request.maximum_items_per_source]:
                    if item.external_id and item.external_id not in seen_ids:
                        seen_ids.add(item.external_id)
                        examples.append(item)
        return examples

    async def resolve_target(self, request: TargetContextRequest) -> TargetContext:
        if request.platform is not Platform.REDDIT:
            raise ValueError("Reddit resolver requires a Reddit request")
        account = self._require_account(request.account_name)
        if request.target_url is None:
            raise ExampleTargetUnavailableError("Reddit target URL is required")
        target_kind = "comment" if request.target_kind in {"comment", "reply", "t1"} else "post"
        canonical = validate_public_url(
            request.target_url,
            Platform.REDDIT,
            target_kind=target_kind,
        )
        subreddit = _subreddit_from_path(urlsplit(canonical).path)
        if subreddit is None or subreddit.casefold() not in {
            item.casefold() for item in account.allowed_subreddits
        }:
            raise ExampleTargetUnavailableError("Reddit target subreddit is not allowlisted")
        post_id, comment_id = _ids_from_path(urlsplit(canonical).path)
        async with self._browser.get_page(Platform.REDDIT, request.account_name) as page:
            await navigate_public_page(page, canonical, Platform.REDDIT)
            post = page.locator(f'div.thing[data-fullname="t3_{post_id}"]').first
            if await post.count() == 0:
                raise ExampleTargetUnavailableError("Reddit target post was not visible")
            title = await safe_text(post.locator("a.title").first)
            body = await safe_text(post.locator("div.usertext-body .md").first)
            comments = await self._discussion_comments(
                page,
                self._settings.example_collection.maximum_comments_per_post,
                root_title=title,
                root_body=body,
            )
            parent_text = None
            target_container = post
            if comment_id is not None:
                target_container = page.locator(
                    f'div.comment.thing[data-fullname="t1_{comment_id}"]'
                ).first
                if await target_container.count() == 0:
                    raise ExampleTargetUnavailableError("Reddit target comment was not visible")
                parent_text = f"{title}\n{body}".strip()
                body = await safe_text(
                    target_container.locator("div.usertext-body .md").first
                )
            await raise_for_collection_page(page, platform=Platform.REDDIT)
            return TargetContext(
                platform=Platform.REDDIT,
                canonical_url=canonical,
                external_id=f"t1_{comment_id}" if comment_id else f"t3_{post_id}",
                title=title,
                body=body,
                parent_text=parent_text,
                discussion_comments=tuple(comments),
                subreddit=subreddit,
                parent_post_id=f"t3_{post_id}",
                parent_comment_id=f"t1_{comment_id}" if comment_id else None,
                metadata={"discussion_count": len(comments)},
            )

    def build_routes(
        self,
        request: ExampleCollectionRequest,
        *,
        allowed_subreddits: set[str] | frozenset[str],
    ) -> tuple[str, ...]:
        routes: list[str] = []
        for subreddit in request.subreddits:
            if subreddit.casefold() not in {item.casefold() for item in allowed_subreddits}:
                raise ValueError(f"subreddit {subreddit!r} is not allowlisted")
            safe_subreddit = _normalize_subreddit(subreddit)
            if request.queries:
                for query in request.queries:
                    routes.append(
                        f"{_ORIGIN}/r/{safe_subreddit}/search?q={quote(query)}&restrict_sr=on"
                        f"&sort=relevance&t={quote(request.time_filter)}"
                    )
            elif request.sort == "top":
                routes.append(
                    f"{_ORIGIN}/r/{safe_subreddit}/top/?t={quote(request.time_filter)}"
                )
            else:
                routes.append(f"{_ORIGIN}/r/{safe_subreddit}/{request.sort}/")
        for raw_url in request.post_urls:
            canonical = validate_public_url(raw_url, Platform.REDDIT, target_kind="post")
            subreddit = _subreddit_from_path(urlsplit(canonical).path)
            if subreddit is None or subreddit.casefold() not in {
                item.casefold() for item in allowed_subreddits
            }:
                raise ValueError("explicit Reddit post URL is not allowlisted")
            routes.append(canonical)
        return tuple(dict.fromkeys(routes))

    async def _extract_posts(
        self,
        page: Page,
        request: ExampleCollectionRequest,
        *,
        allowed_subreddits: set[str],
        include_comments: bool,
    ) -> list[CollectedExample]:
        results: list[CollectedExample] = []
        locator = page.locator('div.thing[data-fullname^="t3_"]')
        for index in range(min(await locator.count(), request.maximum_items_per_source)):
            thing = locator.nth(index)
            classes = (await safe_attribute(thing, "class") or "").casefold()
            if any(marker in classes for marker in ("deleted", "removed", "promoted", "advertising")):
                continue
            item = await self._post_example(thing, request, allowed_subreddits)
            if item is None:
                continue
            results.append(item)
            if include_comments:
                results.extend(
                    await self._comment_examples(
                        thing,
                        request,
                        root_title=item.title or "",
                        root_body=item.body,
                    )
                )
            if len(results) >= request.maximum_items_per_source:
                break
        return results

    async def _comment_examples(
        self,
        post: Locator,
        request: ExampleCollectionRequest,
        *,
        root_title: str,
        root_body: str,
    ) -> list[CollectedExample]:
        results: list[CollectedExample] = []
        locator = post.locator('xpath=following::div.comment.thing[data-fullname^="t1_"]')
        for index in range(min(await locator.count(), request.maximum_comments_per_post)):
            comment = locator.nth(index)
            body = normalize_authored_text(
                await safe_text(comment.locator("div.usertext-body .md").first)
            )
            if _non_whitespace_length(body) < 20:
                continue
            fullname = await safe_attribute(comment, "data-fullname")
            permalink = await safe_attribute(comment.locator("a.bylink").first, "href")
            if not fullname or not permalink:
                continue
            try:
                canonical = validate_public_url(
                    urljoin(_ORIGIN, permalink), Platform.REDDIT, target_kind="comment"
                )
            except ValueError:
                continue
            author = await safe_text(comment.locator("a.author").first, maximum_length=100)
            score = _parse_score(
                await safe_attribute(comment, "data-score"),
                await safe_attribute(comment.locator(".score[title]").first, "title"),
            )
            if score < request.minimum_score:
                continue
            results.append(
                CollectedExample(
                    platform=Platform.REDDIT,
                    content_type=ExampleType.REDDIT_COMMENT,
                    external_id=fullname,
                    source_url=canonical,
                    author_identifier=author_identifier(Platform.REDDIT, author) if author else None,
                    body=body,
                    parent_text=f"{root_title}\n{root_body}".strip(),
                    subreddit=_subreddit_from_path(urlsplit(canonical).path),
                    published_at=_parse_datetime(
                        await safe_attribute(comment.locator("time[datetime]").first, "datetime")
                    ),
                    engagement_score=score,
                    metadata={"score": score},
                )
            )
        return results

    async def _post_example(
        self,
        thing: Locator,
        request: ExampleCollectionRequest,
        allowed_subreddits: set[str],
    ) -> CollectedExample | None:
        fullname = await safe_attribute(thing, "data-fullname")
        if not fullname or not fullname.casefold().startswith("t3_"):
            return None
        title = normalize_authored_text(await safe_text(thing.locator("a.title").first))
        body = normalize_authored_text(
            await safe_text(thing.locator("div.usertext-body .md").first)
        )
        permalink = await safe_attribute(
            thing.locator('a[data-event-action="comments"]').first,
            "href",
        ) or await safe_attribute(thing.locator("a.bylink").first, "href")
        if permalink is None:
            return None
        canonical = validate_public_url(
            urljoin(_ORIGIN, permalink), Platform.REDDIT, target_kind="post"
        )
        subreddit = _subreddit_from_path(urlsplit(canonical).path)
        if subreddit is None or subreddit.casefold() not in {
            item.casefold() for item in allowed_subreddits
        }:
            return None
        if _non_whitespace_length(f"{title}\n{body}") < 20:
            return None
        author = await safe_text(thing.locator("a.author").first, maximum_length=100)
        published_at = _parse_datetime(
            await safe_attribute(thing.locator("time[datetime]").first, "datetime")
        )
        score = _parse_score(
            await safe_attribute(thing, "data-score"),
            await safe_attribute(thing.locator(".score[title]").first, "title"),
        )
        if score < request.minimum_score:
            return None
        return CollectedExample(
            platform=Platform.REDDIT,
            content_type=ExampleType.REDDIT_POST,
            external_id=fullname,
            source_url=canonical,
            author_identifier=author_identifier(Platform.REDDIT, author) if author else None,
            title=title or None,
            body=body,
            subreddit=subreddit,
            published_at=published_at,
            engagement_score=score,
            metadata={"score": score},
        )

    async def _discussion_comments(
        self,
        page: Page,
        maximum_comments_per_post: int,
        *,
        root_title: str,
        root_body: str,
    ) -> list[str]:
        comments: list[str] = []
        locator = page.locator('div.comment.thing[data-fullname^="t1_"]')
        for index in range(min(await locator.count(), maximum_comments_per_post)):
            comment = locator.nth(index)
            classes = (await safe_attribute(comment, "class") or "").casefold()
            if "deleted" in classes or "removed" in classes:
                continue
            body = normalize_authored_text(
                await safe_text(comment.locator("div.usertext-body .md").first)
            )
            if _non_whitespace_length(body) < 20:
                continue
            comments.append(body)
        return comments

    async def _post_ids(self, page: Page) -> set[str]:
        locator = page.locator('div.thing[data-fullname^="t3_"]')
        values: set[str] = set()
        for index in range(min(await locator.count(), 500)):
            value = await safe_attribute(locator.nth(index), "data-fullname")
            if value:
                values.add(value)
        return values

    async def _page_stopped(self, page: Page) -> bool:
        await raise_for_collection_page(page, platform=Platform.REDDIT)
        return False

    def _require_account(self, account_name: str):
        account = self._settings.accounts.reddit.get(account_name)
        if account is None:
            raise ValueError(f"no configured Reddit account named {account_name!r}")
        if not account.enabled:
            raise ValueError(f"configured Reddit account {account_name!r} is disabled")
        return account


def _normalize_subreddit(value: str) -> str:
    normalized = value.strip()
    if normalized.casefold().startswith("r/"):
        normalized = normalized[2:]
    if not normalized or len(normalized) > 21 or not normalized.replace("_", "a").isalnum():
        raise ValueError("invalid subreddit")
    return normalized


def _subreddit_from_path(path: str) -> str | None:
    parts = path.strip("/").split("/")
    if len(parts) >= 2 and parts[0].casefold() == "r":
        return parts[1]
    return None


def _ids_from_path(path: str) -> tuple[str, str | None]:
    parts = path.strip("/").split("/")
    try:
        comments_index = next(index for index, value in enumerate(parts) if value.casefold() == "comments")
        post_id = parts[comments_index + 1]
        comment_id = parts[comments_index + 3] if len(parts) > comments_index + 3 else None
    except (StopIteration, IndexError):
        raise ExampleTargetUnavailableError("Reddit target URL did not contain an id") from None
    return post_id.casefold(), comment_id.casefold() if comment_id else None


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


def _parse_score(*values: str | None) -> int:
    for value in values:
        if not value:
            continue
        text = value.casefold().replace(",", "")
        try:
            return max(0, min(int(float(text.split()[0])), 10_000_000))
        except (ValueError, IndexError):
            continue
    return 0


def _non_whitespace_length(value: str) -> int:
    return sum(not character.isspace() for character in value)


__all__ = ["RedditExampleCollector"]
