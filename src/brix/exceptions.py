"""BRIX top-level exception hierarchy.

Every failure mode in the library has its own exception class so engineers
can handle them specifically without catching broad base exceptions.
"""

from __future__ import annotations


class BrixError(Exception):
    """Base exception for all BRIX errors."""


class BrixConfigurationError(BrixError):
    """Raised when BRIX is configured incorrectly.

    Examples: missing required parameters, incompatible options,
    unrecognized LLM client interface.
    """


class BrixGuardError(BrixError):
    """Raised when a Guard raises an unexpected internal error.

    Args:
        guard_name: The name of the Guard that failed.
        message: Human-readable description of the failure.
    """

    def __init__(self, guard_name: str, message: str) -> None:
        self.guard_name = guard_name
        super().__init__(f"[{guard_name}] {message}")


class BrixInternalError(BrixError):
    """Raised for unexpected errors inside the InterceptorChain or BrixClient.

    Wraps lower-level exceptions that are not Guard-specific.
    """


class BrixGuardBlockedError(BrixError):
    """Raised when a Guard blocks the request.

    Engineers should catch this exception to handle blocked requests
    gracefully (e.g., return a fallback response, log the event, etc.).

    Args:
        guard_name: The name of the Guard that blocked the request.
        reason: Human-readable explanation of why the request was blocked.
    """

    def __init__(self, guard_name: str, reason: str) -> None:
        self.guard_name = guard_name
        self.reason = reason
        super().__init__(f"[{guard_name}] blocked: {reason}")


class BrixTimeoutError(BrixGuardBlockedError):
    """Raised by TimeoutGuard when a call or pipeline exceeds the time limit."""

    def __init__(self, guard_name: str = "timeout", reason: str = "time limit exceeded") -> None:
        super().__init__(guard_name, reason)


class BrixBudgetError(BrixGuardBlockedError):
    """Raised by BudgetGuard when the session cost limit would be exceeded."""

    def __init__(self, guard_name: str = "budget", reason: str = "cost limit exceeded") -> None:
        super().__init__(guard_name, reason)


class BrixRateLimitError(BrixGuardBlockedError):
    """Raised by RateLimitGuard when the request rate exceeds the configured limit."""

    def __init__(
        self, guard_name: str = "rate_limit", reason: str = "rate limit exceeded"
    ) -> None:
        super().__init__(guard_name, reason)


class BrixLoopError(BrixGuardBlockedError):
    """Raised by LoopGuard when a repetition loop is detected in the pipeline."""

    def __init__(self, guard_name: str = "loop", reason: str = "loop detected") -> None:
        super().__init__(guard_name, reason)


class BrixSchemaError(BrixGuardBlockedError):
    """Raised by SchemaGuard when structured output cannot be produced after retries."""

    def __init__(
        self, guard_name: str = "schema", reason: str = "schema validation failed"
    ) -> None:
        super().__init__(guard_name, reason)


__all__ = [
    "BrixError",
    "BrixConfigurationError",
    "BrixGuardError",
    "BrixInternalError",
    "BrixGuardBlockedError",
    "BrixTimeoutError",
    "BrixBudgetError",
    "BrixRateLimitError",
    "BrixLoopError",
    "BrixSchemaError",
]
