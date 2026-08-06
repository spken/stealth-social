"""Deterministic heuristic ranking and optional explanation-only Ollama ranking."""

from __future__ import annotations

import re
from collections.abc import Sequence
from difflib import SequenceMatcher
from typing import Protocol, runtime_checkable
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StringConstraints
from typing_extensions import Annotated

from bot.content.models import (
    ContentRequest,
    RankedCandidate,
    RankingMode,
    RankingResult,
    ValidatedCandidate,
)
from bot.content.prompt_builder import PromptBuilder
from bot.ollama.client import OllamaClient, OllamaMessage

NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


@runtime_checkable
class CandidateRanker(Protocol):
    async def rank(
        self,
        candidates: Sequence[ValidatedCandidate],
        request: ContentRequest,
    ) -> list[RankedCandidate]:
        raise NotImplementedError


class HeuristicCandidateRanker:
    """Rank valid candidates without external state or model calls."""

    async def rank(
        self,
        candidates: Sequence[ValidatedCandidate],
        request: ContentRequest,
    ) -> list[RankedCandidate]:
        ranked: list[RankedCandidate] = []
        for candidate in candidates:
            if candidate.validation.has_errors:
                continue
            components = _components(candidate, request)
            warning_penalty = 0.75 * sum(
                item.severity.value == "warning" for item in candidate.validation.findings
            )
            score = max(0.0, min(10.0, sum(components.values()) - warning_penalty))
            explanation = _explanation(components, warning_penalty)
            ranked.append(
                RankedCandidate(
                    candidate=candidate,
                    ranking=RankingResult(
                        score=score,
                        explanation=explanation,
                        mode=RankingMode.HEURISTIC,
                        components={**components, "warning_penalty": warning_penalty},
                    ),
                )
            )
        return sorted(
            ranked,
            key=lambda item: (-item.ranking.score, item.candidate.ordinal),
        )


class RankingError(RuntimeError):
    """Configured ranking could not produce a complete explanation-only result."""


class OllamaRankingItem(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    candidate_id: UUID
    score: float = Field(ge=0, le=10)
    explanation: NonEmptyString


class OllamaRankingResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    rankings: tuple[OllamaRankingItem, ...]


class OllamaCandidateRanker:
    """Ask Ollama only for bounded scores/explanations, never alternative text."""

    def __init__(self, client: OllamaClient, prompt_builder: PromptBuilder) -> None:
        self._client = client
        self._prompts = prompt_builder

    async def rank(
        self,
        candidates: Sequence[ValidatedCandidate],
        request: ContentRequest,
    ) -> list[RankedCandidate]:
        valid = [item for item in candidates if not item.validation.has_errors]
        if not valid:
            return []
        schema = OllamaRankingResponse.model_json_schema()
        package = self._prompts.build_ranking(request, valid, schema)
        model = str(request.resolved_parameters.get("model", "qwen3:8b"))
        await self._client.require_model(model)
        response = await self._client.chat(
            model=model,
            messages=(
                OllamaMessage(role="system", content=package.system_message),
                OllamaMessage(role="user", content=package.user_message),
            ),
            format_schema=package.output_schema,
            options={"temperature": 0.0, "top_p": 1.0},
            think=False,
        )
        try:
            parsed = OllamaRankingResponse.model_validate_json(response.content)
        except (TypeError, ValueError) as error:
            raise RankingError("Ollama ranking response was malformed") from error
        expected = {item.id for item in valid}
        actual = [item.candidate_id for item in parsed.rankings]
        if set(actual) != expected or len(actual) != len(expected):
            raise RankingError("Ollama ranking response did not cover every valid candidate")
        by_id = {item.id: item for item in valid}
        return [
            RankedCandidate(
                candidate=by_id[item.candidate_id],
                ranking=RankingResult(
                    score=item.score,
                    explanation=item.explanation,
                    mode=RankingMode.OLLAMA,
                    components={},
                ),
            )
            for item in parsed.rankings
        ]


def _components(candidate: ValidatedCandidate, request: ContentRequest) -> dict[str, float]:
    body = candidate.draft.body
    body_tokens = _tokens(body)
    topic_tokens = _tokens(" ".join(item for item in (request.topic, *request.keywords) if item))
    target = request.source_post_text or request.source_comment_text or (
        request.target_context.body if request.target_context else ""
    )
    target_tokens = _tokens(target)
    topic_relevance = _overlap(body_tokens, topic_tokens) * 2.0
    target_overlap = _overlap(body_tokens, target_tokens)
    target_awareness = min(1.5, target_overlap * 1.5)
    if target_overlap > 0.75:
        target_awareness *= 0.5
    community_fit = 1.5 if (
        request.generation_type.value != "reddit_post" or request.subreddit
    ) else 0.0
    specificity = min(1.5, len(body_tokens) / 30.0 + _concrete_ratio(body) * 0.5)
    usefulness = min(1.5, _useful_words(body) / 4.0)
    originality = max(0.0, 1.0 - _generic_ratio(body))
    naturalness = max(0.0, 1.0 - min(1.0, _format_penalty(body) + _generic_ratio(body)))
    return {
        "topic_relevance": min(2.0, topic_relevance),
        "target_awareness": target_awareness,
        "platform_community_fit": community_fit,
        "specificity": specificity,
        "usefulness": usefulness,
        "originality": originality,
        "naturalness": naturalness,
    }


def _explanation(components: dict[str, float], warning_penalty: float) -> str:
    strongest = [name.replace("_", " ") for name, _ in sorted(components.items(), key=lambda item: (-item[1], item[0]))[:2]]
    weakest = [name.replace("_", " ") for name, _ in sorted(components.items(), key=lambda item: (item[1], item[0]))[:2]]
    warning_text = f"; {int(warning_penalty / 0.75)} warning(s) reduced the score" if warning_penalty else ""
    return f"Strongest: {', '.join(strongest)}. Weakest: {', '.join(weakest)}{warning_text}."


def _tokens(value: str) -> set[str]:
    return {
        item
        for item in re.findall(r"[a-z0-9]{3,}", value.casefold())
        if item not in {"the", "and", "for", "with", "this", "that"}
    }


def _overlap(left: set[str], right: set[str]) -> float:
    return len(left & right) / len(left | right) if left and right else 0.0


def _concrete_ratio(value: str) -> float:
    tokens = _tokens(value)
    concrete = sum(item.endswith(("ing", "tion", "er", "ed")) for item in tokens)
    return concrete / len(tokens) if tokens else 0.0


def _useful_words(value: str) -> int:
    return sum(
        word in value.casefold()
        for word in ("how", "because", "step", "example", "tradeoff", "consider", "useful", "lesson")
    )


def _generic_ratio(value: str) -> float:
    phrases = ("as an ai", "in today's world", "it's important to note", "delve into", "leverage")
    lowered = value.casefold()
    return min(1.0, sum(phrase in lowered for phrase in phrases) / 3.0)


def _format_penalty(value: str) -> float:
    headings = len(re.findall(r"^#+\s|^[-*]\s", value, re.MULTILINE))
    dashes = value.count("—") + value.count("--")
    return min(1.0, headings / 8.0 + dashes / 12.0)


__all__ = [
    "CandidateRanker",
    "HeuristicCandidateRanker",
    "OllamaCandidateRanker",
    "OllamaRankingResponse",
    "RankingError",
]
