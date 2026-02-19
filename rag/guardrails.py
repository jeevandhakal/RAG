"""Guardrails for input validation, output validation, and execution limits.

This module provides input guardrails (query length, off-topic detection, PII sanitization),
output guardrails (response length limits), and a structured error taxonomy for the RAG system.
"""

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

logger = logging.getLogger(__name__)


class ErrorCode(str, Enum):
    """Error taxonomy for guardrail violations."""

    QUERY_TOO_LONG = "QUERY_TOO_LONG"
    OFF_TOPIC = "OFF_TOPIC"
    PII_DETECTED = "PII_DETECTED"
    RETRIEVAL_EMPTY = "RETRIEVAL_EMPTY"
    LLM_TIMEOUT = "LLM_TIMEOUT"
    POLICY_BLOCK = "POLICY_BLOCK"
    EMPTY_QUERY = "EMPTY_QUERY"


# Driving/road rules keywords for off-topic detection
DRIVING_TOPIC_KEYWORDS: List[str] = [
    "drive",
    "driving",
    "driver",
    "vehicle",
    "car",
    "road",
    "highway",
    "intersection",
    "traffic",
    "signal",
    "pedestrian",
    "crosswalk",
    "bus",
    "school bus",
    "emergency",
    "park",
    "parking",
    "yield",
    "stop",
    "speed",
    "license",
    "pass",
    "lane",
    "rules",
    "law",
    "regulation",
    "nova scotia",
    "ns",
    "highway traffic act",
    "crosswalk guard",
    "emergency vehicle",
    "right of way",
    "turn",
    "merge",
]

# PII detection patterns
PHONE_PATTERN = re.compile(
    r"\b(?:\d{3}[-.\s]?\d{3}[-.\s]?\d{4}|\(\d{3}\)\s*\d{3}[-.\s]?\d{4}|\d{10})\b"
)
EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
LICENSE_PLATE_PATTERN = re.compile(
    r"\b[A-Z]{2,3}\s*\d{2,4}\b|\b\d{2,4}\s*[A-Z]{2,3}\b",
    re.IGNORECASE,
)


@dataclass
class GuardrailResult:
    """Result of a single guardrail check."""

    passed: bool
    error_code: Optional[ErrorCode] = None
    triggered: List[str] = field(default_factory=list)
    sanitized_query: Optional[str] = None
    warning: Optional[str] = None


def check_query_length(query: str, max_length: int = 500) -> GuardrailResult:
    """Reject empty queries and queries exceeding max_length characters.

    Args:
        query: The user's input query.
        max_length: Maximum allowed query length in characters.

    Returns:
        GuardrailResult with passed=False if empty or too long, else passed=True.
    """
    if len(query.strip()) == 0:
        logger.warning("Guardrail triggered: Empty query")
        return GuardrailResult(
            passed=False,
            error_code=ErrorCode.EMPTY_QUERY,
            triggered=["empty_query"],
        )
    if len(query) > max_length:
        logger.warning(
            "Guardrail triggered: Query too long (%d > %d)", len(query), max_length
        )
        return GuardrailResult(
            passed=False,
            error_code=ErrorCode.QUERY_TOO_LONG,
            triggered=["query_length_limit"],
        )
    return GuardrailResult(passed=True)


def check_off_topic(query: str) -> GuardrailResult:
    """Check if query is related to driving/road rules (Nova Scotia context).

    Args:
        query: The user's input query.

    Returns:
        GuardrailResult with passed=False if off-topic, else passed=True.
    """
    query_lower = query.lower().strip()
    if not query_lower:
        return GuardrailResult(
            passed=False,
            error_code=ErrorCode.EMPTY_QUERY,
            triggered=["empty_query"],
        )

    for keyword in DRIVING_TOPIC_KEYWORDS:
        if keyword in query_lower:
            return GuardrailResult(passed=True)

    # Allow very short queries that might be abbreviations
    if len(query_lower) < 10:
        return GuardrailResult(passed=True)

    logger.warning("Guardrail triggered: Off-topic query")
    return GuardrailResult(
        passed=False,
        error_code=ErrorCode.OFF_TOPIC,
        triggered=["off_topic_detection"],
    )


def check_and_sanitize_pii(query: str) -> GuardrailResult:
    """Detect and strip PII (phone, email, license plate) from query.

    Args:
        query: The user's input query.

    Returns:
        GuardrailResult with sanitized_query and warning if PII found; otherwise passed.
    """
    sanitized = query
    pii_found: List[str] = []

    if PHONE_PATTERN.search(query):
        pii_found.append("phone")
        sanitized = PHONE_PATTERN.sub("[REDACTED_PHONE]", sanitized)

    if EMAIL_PATTERN.search(query):
        pii_found.append("email")
        sanitized = EMAIL_PATTERN.sub("[REDACTED_EMAIL]", sanitized)

    if LICENSE_PLATE_PATTERN.search(query):
        pii_found.append("license_plate")
        sanitized = LICENSE_PLATE_PATTERN.sub("[REDACTED_PLATE]", sanitized)

    if pii_found:
        logger.warning("Guardrail triggered: PII detected (%s)", ", ".join(pii_found))
        return GuardrailResult(
            passed=True,
            error_code=ErrorCode.PII_DETECTED,
            triggered=["pii_detection"],
            sanitized_query=sanitized,
            warning=(
                f"PII detected ({', '.join(pii_found)}) was removed. "
                "I can only answer questions about Nova Scotia driving rules."
            ),
        )
    return GuardrailResult(passed=True)


def check_response_length(response: str, max_words: int = 500) -> GuardrailResult:
    """Cap responses to max_words.

    Args:
        response: The LLM response text.
        max_words: Maximum allowed word count.

    Returns:
        GuardrailResult with sanitized_query (truncated) if over limit.
    """
    word_count = len(response.split())
    if word_count > max_words:
        logger.warning(
            "Guardrail triggered: Response too long (%d > %d words)",
            word_count,
            max_words,
        )
        truncated = " ".join(response.split()[:max_words]) + "... [response truncated]"
        return GuardrailResult(
            passed=True,
            triggered=["response_length_limit"],
            sanitized_query=truncated,
        )
    return GuardrailResult(passed=True)


@dataclass
class InputGuardrailResult:
    """Result of applying all input guardrails."""

    should_proceed: bool
    refusal_message: Optional[str] = None
    error_code: Optional[ErrorCode] = None
    triggered: List[str] = field(default_factory=list)
    sanitized_query: Optional[str] = None
    warning: Optional[str] = None


def apply_input_guardrails(query: str) -> InputGuardrailResult:
    """Apply all input guardrails in sequence.

    Checks: (1) query length/empty, (2) off-topic, (3) PII sanitization.

    Args:
        query: The user's input query.

    Returns:
        InputGuardrailResult with should_proceed, refusal_message, and sanitized_query.
    """
    # 1. Empty / length check
    result = check_query_length(query)
    if not result.passed:
        return InputGuardrailResult(
            should_proceed=False,
            refusal_message="Please enter a question.",
            error_code=result.error_code,
            triggered=result.triggered,
        )

    # 2. Off-topic
    result = check_off_topic(query)
    if not result.passed:
        return InputGuardrailResult(
            should_proceed=False,
            refusal_message="I can only answer questions about Nova Scotia driving rules.",
            error_code=result.error_code,
            triggered=result.triggered,
        )

    # 3. PII - sanitize and continue
    result = check_and_sanitize_pii(query)
    sanitized = result.sanitized_query if result.sanitized_query else query

    return InputGuardrailResult(
        should_proceed=True,
        error_code=result.error_code,
        triggered=result.triggered,
        sanitized_query=sanitized,
        warning=result.warning,
    )
