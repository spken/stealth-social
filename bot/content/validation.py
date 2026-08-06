"""Pure deterministic validation for generated and edited candidates."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Iterable

from bot.config import SubredditContentRulesSettings
from bot.content.models import (
    CandidateDraft,
    ContentPurpose,
    ContentRequest,
    FactRequirement,
    ValidationFinding,
    ValidationResult,
    ValidationSeverity,
)
from bot.content.prompt_safety import inspect_untrusted_text
from bot.examples.models import ContentExample
from bot.models import ActionType, Platform, SocialAction

SIBLING_DUPLICATE_RATIO = 0.90
PUBLISHED_DUPLICATE_RATIO = 0.85
EXAMPLE_COPY_ERROR_RATIO = 0.85
EXAMPLE_COPY_WARNING_RATIO = 0.70
COPIED_TOKEN_RUN = 8

_URL = re.compile(r"https?://[^\s)]+", re.IGNORECASE)
_MENTION = re.compile(r"(?<![\w@])@[A-Za-z0-9_]{1,15}\b")
_HASHTAG = re.compile(r"(?<!\w)#[A-Za-z0-9_]+")
_SUBREDDIT_LINK = re.compile(r"/r/([A-Za-z0-9_]{2,21})\b", re.IGNORECASE)
_NUMERIC_CLAIM = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:%|percent|users?|customers?|clients?|downloads?|revenue|million|billion)\b|[$€£]\s*\d",
    re.IGNORECASE,
)
_CREDENTIAL_CLAIM = re.compile(
    r"\b(?:i\s*(?:am|'m|have|hold|earned|possess(?:es)?)|"
    r"we\s*(?:are|'re|have|hold|earned|possess(?:es)?)|"
    r"(?:our\s+)?(?:team|company|founder|staff)\s+"
    r"(?:is|are|has|holds?|earned|possesses?))\s+"
    r"(?:an?\s+)?(?:certified|licensed|accredited|qualified|registered|award[- ]winning|"
    r"expert|doctor|phd|professional|degree|credential|certification|license|"
    r"licence|accreditation)\b|"
    r"\b(?:certified|licensed|accredited|registered|award[- ]winning)\s+"
    r"(?:professional|expert|specialist|by)\b",
    re.IGNORECASE,
)
_TESTIMONIAL_CLAIM = re.compile(
    r"\b(?:customers?|clients?|users?|teams?|reviewers?|people)\s+"
    r"(?:love|loves|loved|recommend|recommends|recommended|endorse|endorses|"
    r"endorsed|report|reports|reported|say|says|said|tell|tells|told|prefer|"
    r"prefers|trust|trusts|trusted|praise|praises|rave|raves)\b|"
    r"\b(?:a|one of our|many)\s+(?:customer|client|user|team)s?\s+"
    r"(?:said|reported|recommended|loved|trusted|praised|raved|told)\b|"
    r"\b(?:customer|client|user)\s+(?:testimonial|review|quote)s?\b",
    re.IGNORECASE,
)
_ENDORSEMENT_CLAIM = re.compile(
    r"\b(?:endorsed|recommended|backed|trusted|adopted|partnered|selected|"
    r"chosen|used|deployed|relied)\s+by\b|"
    r"\b(?:official\s+partner|in\s+partnership\s+with|partnered\s+with|"
    r"used\s+at|adopted\s+by|chosen\s+by)\b",
    re.IGNORECASE,
)
_PERFORMANCE_CLAIM = re.compile(
    r"\b(?:i|we|our\s+(?:product|tool|service|team|workflow)|this\s+"
    r"(?:product|tool|service|workflow)|teams?|companies?|organizations?)\s+"
    r"(?:increased|improved|boosted|reduced|saved|grew|cut|accelerated|"
    r"doubled|tripled|outperformed|optimized|streamlined|delivered)\b|"
    r"\b(?:increase|improve|boost|reduce|save|grow|accelerate|faster|better|"
    r"best|fastest|guaranteed|more\s+efficient|less\s+time|half\s+the\s+time|"
    r"twice\s+as\s+fast)\b[^.!?\n]{0,100}\b"
    r"(?:\d+(?:\.\d+)?\s*(?:%|percent|x|times)|#\s*1|number\s+one|"
    r"half\s+the\s+time|twice\s+as\s+fast)\b",
    re.IGNORECASE,
)
_EXPERIENCE_CLAIM = re.compile(
    r"\b(?:i|we|our\s+(?:team|company|founder|staff))(?:['\u2019]ve)?\s+"
    r"(?:used|use|built|achieved|got|saw|helped|saved|made|tried|tested|"
    r"worked\s+with|have\s+worked\s+with|had\s+experience|have\s+experience|"
    r"has\s+experience)\b|"
    r"\b(?:i|we)(?:['\u2019]ve)?\s+(?:(?:have|has|had)\s+)?(?:hands?[- ]on\s+)?"
    r"(?:experience|expertise)\b",
    re.IGNORECASE,
)
_NUMERIC_CLAIM_BOUNDARY = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:%|percent|users?|customers?|clients?|downloads?|"
    r"revenue|million|billion)(?=$|[^\w])|[$€£]\s*\d",
    re.IGNORECASE,
)
_GENERIC_PREAMBLE = re.compile(
    r"^(?:here(?:'s| is)\s+(?:your|the)\s+(?:post|reply|comment)|sure[,!]?\s+here)",
    re.IGNORECASE,
)
_STOPWORDS = frozenset("a an and are as at be by for from has have in is it of on or that the this to was were with you your we our i me my".split())
_CLAIM_FILLERS = _STOPWORDS | frozenset(
    "am an been being can could did does had how if into its may might more most must no not should so than they them their then there these those what when where which who why will would"
    .split()
)
_ANCHOR_TOKEN = re.compile(r"\d+(?:\.\d+)?%?|[a-z][a-z0-9_-]{2,}", re.IGNORECASE)

_UNSUPPORTED_CLAIM_DETECTORS = (
    (
        "unsupported_credential",
        _CREDENTIAL_CLAIM,
        frozenset(
            {
                "certified",
                "licensed",
                "accredited",
                "qualified",
                "registered",
                "award-winning",
                "expert",
                "doctor",
                "phd",
                "professional",
                "degree",
                "credential",
                "certification",
                "license",
                "licence",
                "accreditation",
            }
        ),
    ),
    (
        "unsupported_testimonial",
        _TESTIMONIAL_CLAIM,
        frozenset(
            {
                "love",
                "loves",
                "loved",
                "recommend",
                "recommends",
                "recommended",
                "endorse",
                "endorses",
                "endorsed",
                "report",
                "reports",
                "reported",
                "say",
                "says",
                "said",
                "tell",
                "tells",
                "told",
                "prefer",
                "prefers",
                "trust",
                "trusts",
                "trusted",
                "praise",
                "praises",
                "rave",
                "raves",
                "testimonial",
                "review",
                "quote",
            }
        ),
    ),
    (
        "unsupported_endorsement",
        _ENDORSEMENT_CLAIM,
        frozenset(
            {
                "endorse",
                "endorsed",
                "recommended",
                "backed",
                "trusted",
                "adopted",
                "partnered",
                "selected",
                "chosen",
                "used",
                "deployed",
                "relied",
                "official",
                "partner",
                "partnership",
            }
        ),
    ),
    (
        "unsupported_performance",
        _PERFORMANCE_CLAIM,
        frozenset(
            {
                "increased",
                "improved",
                "boosted",
                "reduced",
                "saved",
                "grew",
                "cut",
                "accelerated",
                "doubled",
                "tripled",
                "outperformed",
                "optimized",
                "streamlined",
                "delivered",
                "increase",
                "improve",
                "boost",
                "reduce",
                "save",
                "grow",
                "accelerate",
                "faster",
                "better",
                "best",
                "fastest",
                "guaranteed",
                "efficient",
                "twice",
            }
        ),
    ),
    (
        "unsupported_experience",
        _EXPERIENCE_CLAIM,
        frozenset(
            {
                "used",
                "use",
                "built",
                "achieved",
                "saw",
                "helped",
                "saved",
                "made",
                "tried",
                "tested",
                "worked",
                "experience",
                "expertise",
            }
        ),
    ),
    ("unsupported_numeric_claim", _NUMERIC_CLAIM_BOUNDARY, frozenset()),
)


@dataclass(frozen=True, slots=True)
class ValidationContext:
    request: ContentRequest
    selected_examples: tuple[ContentExample, ...] = ()
    recent_published_contents: tuple[str, ...] = ()
    sibling_candidates: tuple[CandidateDraft, ...] = ()
    request_facts: tuple[str, ...] = ()
    allowed_subreddits: frozenset[str] = frozenset()
    community_rules: dict[str, SubredditContentRulesSettings] = field(default_factory=dict)
    supplied_urls: frozenset[str] = frozenset()
    maximum_hashtags: int = 3


class CandidateValidator:
    """Return structured errors/warnings without rewriting the draft."""

    def validate(
        self,
        candidate: CandidateDraft,
        context: ValidationContext,
    ) -> ValidationResult:
        request = context.request
        findings: list[ValidationFinding] = []
        body = candidate.body
        normalized_body = _normalize(body)
        if not body.strip():
            _add(findings, "empty_body", ValidationSeverity.ERROR, "body must not be blank", "body")
        if request.generation_type.value == "reddit_post" and not candidate.title:
            _add(findings, "required_title", ValidationSeverity.ERROR, "Reddit posts require a title", "title")
        if request.generation_type.value != "reddit_post" and candidate.title is not None:
            _add(findings, "forbidden_title", ValidationSeverity.ERROR, "this action type does not accept a title", "title")

        try:
            action = _prospective_action(candidate, request)
            SocialAction.model_validate(action.model_dump(exclude={"fingerprint"}))
        except Exception as error:
            message = " ".join(str(error).split())[:240]
            code = "platform_length" if "280" in message or "characters" in message else "social_action_shape"
            _add(findings, code, ValidationSeverity.ERROR, message)

        for sibling in context.sibling_candidates:
            ratio = SequenceMatcher(None, normalized_body, _normalize(sibling.body)).ratio()
            if ratio >= SIBLING_DUPLICATE_RATIO:
                _add(findings, "duplicate_candidate", ValidationSeverity.ERROR, "candidate duplicates a sibling", "body")
                break
        for published in context.recent_published_contents:
            ratio = SequenceMatcher(None, normalized_body, _normalize(published)).ratio()
            if ratio >= PUBLISHED_DUPLICATE_RATIO:
                _add(findings, "published_similarity", ValidationSeverity.ERROR, "candidate is too similar to recent published content", "body")
                break

        for example in context.selected_examples:
            example_text = _normalize(f"{example.title or ''} {example.body}")
            ratio = SequenceMatcher(None, normalized_body, example_text).ratio()
            if ratio >= EXAMPLE_COPY_ERROR_RATIO or _has_copied_token_run(normalized_body, example_text):
                _add(findings, "example_copy", ValidationSeverity.ERROR, "candidate copies too much selected example text", "body")
                break
            if ratio >= EXAMPLE_COPY_WARNING_RATIO:
                _add(findings, "example_copy", ValidationSeverity.WARNING, "candidate resembles a selected example", "body")

        forbidden = (
            tuple(request.forbidden_claims)
            + tuple(request.forbidden_phrases)
            + tuple(request.account_context.forbidden_claims)
        )
        for phrase in forbidden:
            if _normalize(phrase) in normalized_body:
                _add(findings, "forbidden_phrase", ValidationSeverity.ERROR, "candidate contains a forbidden phrase", "body")
                break
        for fact in request.required_facts:
            anchors = fact.required_terms or tuple(_tokens(fact.statement))
            if anchors and not all(_normalize(term) in normalized_body for term in anchors):
                _add(findings, "required_fact", ValidationSeverity.ERROR, "candidate omitted a required fact", "body")
        trusted_claims = (
            tuple(request.account_context.verified_facts)
            + context.request_facts
            + tuple(item.statement for item in request.required_facts)
            + tuple(
                term
                for item in request.required_facts
                for term in item.required_terms
            )
        )
        for code, detector, support_anchors in _UNSUPPORTED_CLAIM_DETECTORS:
            for match in detector.finditer(body):
                if not _claim_is_supported(
                    _claim_excerpt(body, match),
                    trusted_claims,
                    detector=detector,
                    detected_claim=match.group(0),
                    support_anchors=support_anchors,
                ):
                    _add(
                        findings,
                        code,
                        ValidationSeverity.ERROR,
                        "candidate contains an unsupported claim",
                        "body",
                    )
                    break

        disclosures = tuple(request.account_context.required_disclosures)
        purpose = request.content_purpose
        community = _community_rule(context, request.subreddit)
        if purpose in {
            ContentPurpose.PRODUCT_UPDATE,
            ContentPurpose.BUILDER_UPDATE,
            ContentPurpose.PROMOTIONAL,
            ContentPurpose.CUSTOMER_SUPPORT,
        }:
            if not request.account_context.identity:
                _add(findings, "disclosure_required", ValidationSeverity.ERROR, "this purpose requires configured account affiliation")
            else:
                affiliation_markers = disclosures or (request.account_context.identity,)
                if not any(
                    _normalize(marker) in normalized_body
                    for marker in affiliation_markers
                ):
                    _add(
                        findings,
                        "disclosure_required",
                        ValidationSeverity.ERROR,
                        "candidate did not identify the configured account affiliation",
                    )
            for disclosure in disclosures:
                if _normalize(disclosure) not in normalized_body:
                    _add(findings, "disclosure_required", ValidationSeverity.ERROR, "candidate omitted a required disclosure")
        if request.platform is Platform.REDDIT and purpose is ContentPurpose.PROMOTIONAL:
            if community is None or not community.allow_promotional_content:
                _add(findings, "community_promotion", ValidationSeverity.ERROR, "destination community does not explicitly permit promotion")
            elif community is not None:
                for disclosure in community.required_disclosures:
                    if _normalize(disclosure) not in normalized_body:
                        _add(findings, "disclosure_required", ValidationSeverity.ERROR, "candidate omitted a community disclosure")
        if purpose is ContentPurpose.ORGANIC_DISCUSSION and request.call_to_action:
            _add(findings, "purpose_violation", ValidationSeverity.ERROR, "organic discussion cannot contain a promotional call to action")
        if purpose is ContentPurpose.ORGANIC_DISCUSSION:
            for product in request.account_context.products:
                if _normalize(product) in normalized_body:
                    _add(findings, "purpose_violation", ValidationSeverity.ERROR, "organic discussion cannot promote a configured product", "body")
                    break
        if community is not None:
            for phrase in community.forbidden_phrases:
                if _normalize(phrase) in normalized_body:
                    _add(findings, "forbidden_phrase", ValidationSeverity.ERROR, "candidate contains a community-forbidden phrase")
            if candidate.title and community.maximum_title_characters and len(candidate.title) > community.maximum_title_characters:
                _add(findings, "platform_length", ValidationSeverity.ERROR, "candidate title exceeds the community limit", "title")
            if community.maximum_body_characters and len(body) > community.maximum_body_characters:
                _add(findings, "platform_length", ValidationSeverity.ERROR, "candidate body exceeds the community limit", "body")

        for url in _URL.findall(body):
            if url.rstrip(".,!?)]") not in context.supplied_urls:
                _add(findings, "unexpected_url", ValidationSeverity.ERROR, "candidate contains an unexpected URL", "body")
        for mention in _MENTION.findall(body):
            if not re.fullmatch(r"@[A-Za-z0-9_]{1,15}", mention):
                _add(findings, "unexpected_mention", ValidationSeverity.ERROR, "candidate contains an invalid mention", "body")
        hashtags = _HASHTAG.findall(body)
        if hashtags and not any(keyword.startswith("#") for keyword in request.keywords):
            _add(findings, "unexpected_hashtag", ValidationSeverity.ERROR, "hashtags were not requested", "body")
        if len(hashtags) > context.maximum_hashtags:
            _add(findings, "unexpected_hashtag", ValidationSeverity.ERROR, "candidate contains too many hashtags", "body")
        if request.platform is Platform.REDDIT:
            for subreddit in _SUBREDDIT_LINK.findall(body):
                allowed = {item.casefold() for item in context.allowed_subreddits}
                if request.subreddit and subreddit.casefold() != request.subreddit.casefold() and subreddit.casefold() not in allowed:
                    _add(findings, "subreddit_violation", ValidationSeverity.ERROR, "candidate references a disallowed subreddit", "body")
            if re.search(r"<\s*(?:script|table)|</?script|```[^\n]*```", body, re.IGNORECASE):
                _add(findings, "unsupported_markdown", ValidationSeverity.ERROR, "candidate contains unsupported Reddit markup", "body")

        safety = inspect_untrusted_text(f"{candidate.title or ''}\n{body}")
        if safety.findings or any(tag in body for tag in ("<TASK>", "</TASK>", "<OUTPUT_SCHEMA>", "</OUTPUT_SCHEMA>")):
            _add(findings, "prompt_leakage", ValidationSeverity.ERROR, "candidate contains prompt-control or role-instruction text")
        if _GENERIC_PREAMBLE.search(body):
            _add(findings, "model_preamble", ValidationSeverity.ERROR, "candidate contains a model preamble", "body")
        return ValidationResult(findings=tuple(findings))


def _prospective_action(candidate: CandidateDraft, request: ContentRequest) -> SocialAction:
    generation_type = request.generation_type.value
    target = request.target_url or (
        request.target_context.canonical_url if request.target_context else None
    )
    parent_post_id = request.target_context.parent_post_id if request.target_context else None
    parent_comment_id = request.target_context.parent_comment_id if request.target_context else None
    if generation_type in {"x_post", "x_reply"}:
        return SocialAction(
            action_type=ActionType.X_POST,
            platform=Platform.X,
            account_name=request.account_name,
            content=candidate.body,
            target_url=target,
            parent_post_id=parent_post_id,
        )
    if generation_type == "reddit_post":
        return SocialAction(
            action_type=ActionType.REDDIT_POST,
            platform=Platform.REDDIT,
            account_name=request.account_name,
            title=candidate.title,
            content=candidate.body,
            subreddit=request.subreddit,
        )
    if generation_type == "reddit_comment":
        return SocialAction(
            action_type=ActionType.REDDIT_COMMENT,
            platform=Platform.REDDIT,
            account_name=request.account_name,
            content=candidate.body,
            subreddit=request.subreddit,
            target_url=target,
            parent_post_id=parent_post_id,
        )
    return SocialAction(
        action_type=ActionType.REDDIT_REPLY,
        platform=Platform.REDDIT,
        account_name=request.account_name,
        content=candidate.body,
        subreddit=request.subreddit,
        target_url=target,
        parent_comment_id=parent_comment_id,
    )


def _community_rule(
    context: ValidationContext,
    subreddit: str | None,
) -> SubredditContentRulesSettings | None:
    if subreddit is None:
        return None
    for name, rule in context.community_rules.items():
        if name.casefold() == subreddit.casefold():
            return rule
    return None


def _normalize(value: str) -> str:
    return " ".join(value.casefold().split())


def _claim_anchor_tokens(value: str) -> set[str]:
    anchors: set[str] = set()
    for token in _ANCHOR_TOKEN.findall(value.casefold()):
        if token in _CLAIM_FILLERS:
            continue
        if token.endswith("%"):
            anchors.add(token[:-1])
            anchors.add("percent")
        else:
            anchors.add(token)
    return anchors


def _claim_excerpt(value: str, match: re.Match[str]) -> str:
    """Include the detected claim's bounded sentence for anchor comparison."""

    maximum_length = 240
    start = max(
        value.rfind(".", 0, match.start()),
        value.rfind("!", 0, match.start()),
        value.rfind("?", 0, match.start()),
        value.rfind("\n", 0, match.start()),
    ) + 1
    boundaries = [
        position
        for position in (
            value.find(".", match.end()),
            value.find("!", match.end()),
            value.find("?", match.end()),
            value.find("\n", match.end()),
        )
        if position >= 0
    ]
    end = min(boundaries, default=len(value))
    excerpt = value[start:end].strip()
    if len(excerpt) <= maximum_length:
        return excerpt
    offset = match.start() - start
    left = max(0, offset - maximum_length // 2)
    return excerpt[left : left + maximum_length]


def _claim_is_supported(
    claim: str,
    trusted_claims: Iterable[str],
    *,
    detector: re.Pattern[str],
    detected_claim: str,
    support_anchors: frozenset[str] = frozenset(),
) -> bool:
    """Require the same trusted claim type and all salient claim anchors."""

    claim_anchors = _claim_anchor_tokens(claim)
    if not claim_anchors:
        return False
    detected_anchors = _claim_anchor_tokens(detected_claim)
    numeric_anchors = {item for item in claim_anchors if item[:1].isdigit()}
    semantic_anchors = claim_anchors - numeric_anchors
    predicate_anchors = detected_anchors & support_anchors
    if support_anchors and not predicate_anchors:
        return False
    context_anchors = semantic_anchors - detected_anchors
    for trusted in trusted_claims:
        for trusted_match in detector.finditer(trusted):
            trusted_excerpt = _claim_excerpt(trusted, trusted_match)
            trusted_anchors = _claim_anchor_tokens(trusted_excerpt)
            trusted_detected = _claim_anchor_tokens(trusted_match.group(0))
            if detected_anchors and not detected_anchors.issubset(trusted_detected):
                continue
            if numeric_anchors and not numeric_anchors.issubset(trusted_anchors):
                continue
            if predicate_anchors and not predicate_anchors.issubset(trusted_anchors):
                continue
            if context_anchors and not context_anchors.issubset(trusted_anchors):
                continue
            if numeric_anchors or predicate_anchors:
                return True
    return False


def _tokens(value: str) -> tuple[str, ...]:
    return tuple(
        token
        for token in re.findall(r"[a-z0-9]{3,}", _normalize(value))
        if token not in _STOPWORDS
    )


def _has_copied_token_run(candidate: str, example: str) -> bool:
    candidate_tokens = _tokens(candidate)
    example_tokens = _tokens(example)
    if len(candidate_tokens) < COPIED_TOKEN_RUN:
        return False
    for start in range(len(candidate_tokens) - COPIED_TOKEN_RUN + 1):
        run = candidate_tokens[start : start + COPIED_TOKEN_RUN]
        if " ".join(run) in " ".join(example_tokens):
            return True
    return False


def _add(
    findings: list[ValidationFinding],
    code: str,
    severity: ValidationSeverity,
    message: str,
    field: str | None = None,
) -> None:
    if any(item.code == code and item.field == field for item in findings):
        return
    findings.append(
        ValidationFinding(code=code, severity=severity, message=message, field=field)
    )


__all__ = [
    "CandidateValidator",
    "COPIED_TOKEN_RUN",
    "EXAMPLE_COPY_ERROR_RATIO",
    "EXAMPLE_COPY_WARNING_RATIO",
    "PUBLISHED_DUPLICATE_RATIO",
    "SIBLING_DUPLICATE_RATIO",
    "ValidationContext",
]
