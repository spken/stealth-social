"""Deterministic inspection and safe framing for untrusted public text."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from bot.examples.models import (
    PromptInjectionCategory,
    PromptInjectionFinding,
    PromptInjectionSeverity,
    PromptSafetyResult,
)


class PromptInjectionPolicyError(ValueError):
    """The application could not preserve an untrusted-data boundary."""


_PATTERNS: tuple[
    tuple[PromptInjectionCategory, PromptInjectionSeverity, re.Pattern[str]], ...
] = (
    (
        PromptInjectionCategory.AUTHORITY_OVERRIDE,
        PromptInjectionSeverity.HIGH,
        re.compile(r"\b(?:ignore|disregard|forget|override)\s+(?:all\s+)?(?:previous|prior|above|system)\s+instructions?\b", re.I),
    ),
    (
        PromptInjectionCategory.ROLE_IMPERSONATION,
        PromptInjectionSeverity.HIGH,
        re.compile(r"\b(?:you are now|act as|pretend to be)\s+(?:the\s+)?(?:system|developer|assistant|administrator)\b", re.I),
    ),
    (
        PromptInjectionCategory.DELIMITER_INJECTION,
        PromptInjectionSeverity.HIGH,
        re.compile(r"(?:</?[A-Z][A-Z0-9_ -]{1,40}>|\bBEGIN\s+(?:SYSTEM|DEVELOPER)\b|\bEND\s+(?:SYSTEM|DEVELOPER)\b)", re.I),
    ),
    (
        PromptInjectionCategory.PROMPT_EXTRACTION,
        PromptInjectionSeverity.MEDIUM,
        re.compile(r"\b(?:reveal|show|print|dump|extract)\s+(?:the\s+)?(?:system|developer|hidden)?\s*(?:prompt|instructions?|schema)\b", re.I),
    ),
    (
        PromptInjectionCategory.TOOL_COMMAND_REQUEST,
        PromptInjectionSeverity.HIGH,
        re.compile(r"\b(?:run|execute|call|invoke)\s+(?:this\s+)?(?:command|tool|script|shell)\b|\b(?:curl|powershell|rm\s+-|del\s+)", re.I),
    ),
    (
        PromptInjectionCategory.TASK_REDIRECTION,
        PromptInjectionSeverity.MEDIUM,
        re.compile(r"\b(?:instead|new\s+task|your\s+real\s+task)\s*[:,-]?\s*(?:write|return|send|do|publish|post)\b", re.I),
    ),
)


def inspect_untrusted_text(text: str) -> PromptSafetyResult:
    if not isinstance(text, str):
        raise PromptInjectionPolicyError("untrusted text must be a string")
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    findings: list[PromptInjectionFinding] = []
    for category, severity, pattern in _PATTERNS:
        for match in pattern.finditer(normalized):
            start, end = match.span()
            evidence = " ".join(match.group(0).split())[:96]
            findings.append(
                PromptInjectionFinding(
                    category=category,
                    severity=severity,
                    evidence=evidence,
                    start_offset=start,
                    end_offset=end,
                )
            )
    findings.sort(key=lambda item: (item.start_offset, item.end_offset, item.category.value))
    return PromptSafetyResult(findings=tuple(findings))


def encode_untrusted_json(value: Any) -> str:
    """Serialize untrusted values and escape tag-significant characters."""

    try:
        encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError) as error:
        raise PromptInjectionPolicyError("untrusted data was not JSON serializable") from error
    return (
        encoded.replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )


__all__ = [
    "PromptInjectionPolicyError",
    "encode_untrusted_json",
    "inspect_untrusted_text",
]
