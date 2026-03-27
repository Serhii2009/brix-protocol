"""Error classification for RetryGuard.

Classifies exceptions into RETRYABLE, FATAL, or UNKNOWN so RetryGuard
can decide whether to retry or raise immediately.
"""

from __future__ import annotations

import asyncio
import warnings
from enum import Enum


class ErrorClass(Enum):
    """Classification of an LLM call error."""

    RETRYABLE = "retryable"
    """Transient error — safe to retry with backoff."""

    FATAL = "fatal"
    """Permanent error — retrying will not help; raise immediately."""

    UNKNOWN = "unknown"
    """Unrecognized error — treat as RETRYABLE with a warning."""


# HTTP status codes that are safe to retry
_RETRYABLE_STATUS_CODES: frozenset[int] = frozenset({429, 500, 502, 503, 504})

# HTTP status codes that are fatal (retrying will not help)
_FATAL_STATUS_CODES: frozenset[int] = frozenset({401, 403})

# Substrings in the exception message that indicate a fatal error
_FATAL_MESSAGE_SUBSTRINGS: tuple[str, ...] = (
    "invalid_request_error",
    "content_policy_violation",
    "model_not_found",
    "context_length_exceeded",
    "invalid_api_key",
    "permission_denied",
)


def _get_status_code(exc: BaseException) -> int | None:
    """Extract HTTP status code from an exception, if available."""
    # Standard attribute on HTTP errors from openai/anthropic SDKs
    for attr in ("status_code", "status", "http_status"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value

    # Parse from string representation as last resort
    msg = str(exc)
    for code in [429, 500, 502, 503, 504, 401, 403]:
        if str(code) in msg:
            return code

    return None


def classify_error(exc: BaseException) -> ErrorClass:
    """Classify an exception as RETRYABLE, FATAL, or UNKNOWN.

    Classification logic:
    - ``RETRYABLE``: network errors, transient HTTP failures (429/500/502/503/504)
    - ``FATAL``: auth failures (401/403), invalid requests, content policy violations
    - ``UNKNOWN``: anything else — treated as RETRYABLE with a warning emitted

    Args:
        exc: The exception to classify.

    Returns:
        An :class:`ErrorClass` value.
    """
    # Explicit retryable exception types
    if isinstance(exc, (ConnectionError, TimeoutError, asyncio.TimeoutError, OSError)):
        return ErrorClass.RETRYABLE

    # Check HTTP status code
    status = _get_status_code(exc)
    if status is not None:
        if status in _FATAL_STATUS_CODES:
            return ErrorClass.FATAL
        if status in _RETRYABLE_STATUS_CODES:
            return ErrorClass.RETRYABLE

    # Check error message for fatal substrings
    msg = str(exc).lower()
    for substring in _FATAL_MESSAGE_SUBSTRINGS:
        if substring in msg:
            return ErrorClass.FATAL

    # Unknown error — warn and treat as retryable
    warnings.warn(
        f"RetryGuard: unrecognized error type {type(exc).__name__!r}, treating as retryable. "
        "Add it to _retry_classifier.py if this is incorrect.",
        stacklevel=3,
    )
    return ErrorClass.UNKNOWN


__all__ = ["ErrorClass", "classify_error"]
