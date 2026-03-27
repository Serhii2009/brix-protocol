"""RetryGuard — handles transient LLM failures with smart classification.

The 100% guarantee: either the call eventually succeeds, or the engineer
receives a :class:`~brix.exceptions.BrixGuardError` with the complete retry
history (attempts made, delays used, final error). No call silently disappears.

RetryGuard is implemented as a **short-circuit guard**: it calls the LLM
itself with retry logic and returns a CallResponse from pre_call, causing the
InterceptorChain to skip its own LLM call. This is necessary because the chain
propagates LLM exceptions before post_call can run, making post_call retries
architecturally impossible in the current design.

The retry loop:
1. Attempts the LLM call (with per_call_timeout from TimeoutGuard if set).
2. On success: stores retry metadata and returns the real response (with usage).
3. On FATAL error: raises immediately — retrying won't help.
4. On RETRYABLE error: applies exponential backoff with jitter.
5. On UNKNOWN error: warns and treats as retryable.
6. If the retry_budget_seconds is exhausted: raises BrixGuardError with history.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from random import random
from typing import Any

from brix.context import ExecutionContext
from brix.exceptions import BrixGuardError
from brix.guards._retry_classifier import ErrorClass, classify_error
from brix.guards.protocol import CallRequest, CallResponse


class RetryGuard:
    """Guard that retries transient LLM failures with exponential backoff.

    Classifies errors as RETRYABLE, FATAL, or UNKNOWN and acts accordingly.
    Respects ``context.metadata["_per_call_timeout"]`` set by TimeoutGuard —
    each individual attempt is wrapped with ``asyncio.wait_for`` if a timeout
    is configured.

    **Important:** RetryGuard must be the last guard before RegulatedGuard in
    the chain (registered last by BRIX.wrap()). It short-circuits the chain's
    own LLM call. The real LLM response object is returned as-is, preserving
    ``response.usage`` so BudgetGuard.post_call can calculate the actual cost.

    Args:
        llm_callable: The async callable that sends a request to the LLM and
            returns a :class:`~brix.guards.protocol.CallResponse`. Built by
            ``build_llm_callable(llm_client)`` in BRIX.wrap().
        max_retries: Maximum number of retry attempts (not counting the first).
            Total attempts = max_retries + 1. Default 3.
        backoff_base: Base for exponential backoff. Default 2.0.
        max_backoff: Maximum backoff delay in seconds. Default 60.0.
        retry_budget_seconds: Total time budget for all retry delays combined.
            If sleeping for the next delay would exceed this budget, raises
            immediately with the full retry history. Default 120.0.
        retry_on: Optional list of HTTP status codes to treat as retryable in
            addition to the default RETRYABLE set. Unused if None.
    """

    name: str = "retry"

    def __init__(
        self,
        llm_callable: Callable[[CallRequest], Awaitable[CallResponse]],
        *,
        max_retries: int = 3,
        backoff_base: float = 2.0,
        max_backoff: float = 60.0,
        retry_budget_seconds: float = 120.0,
        retry_on: list[int] | None = None,
    ) -> None:
        self._llm_callable = llm_callable
        self._max_retries = max_retries
        self._backoff_base = backoff_base
        self._max_backoff = max_backoff
        self._retry_budget = retry_budget_seconds
        self._extra_retryable_codes: frozenset[int] = (
            frozenset(retry_on) if retry_on else frozenset()
        )

    async def pre_call(
        self,
        request: CallRequest,
        context: ExecutionContext,
    ) -> CallResponse:
        """Call the LLM with retry logic and return the real response.

        This method short-circuits the chain — the InterceptorChain will not
        make its own LLM call when a guard returns a CallResponse from pre_call.

        The actual response object from the LLM callable is returned as-is,
        preserving ``response.usage`` for BudgetGuard.post_call to calculate
        actual costs.

        Args:
            request: The outbound request.
            context: Mutable session state. ``_per_call_timeout`` is read from
                ``context.metadata`` if set by TimeoutGuard.

        Returns:
            The LLM response (success path).

        Raises:
            BrixGuardError: If all retries are exhausted or the retry budget
                is exceeded. The exception carries the full retry history.
            The original exception: If the error is classified as FATAL.
        """
        budget_start = time.perf_counter()
        history: list[dict[str, Any]] = []
        last_exc: BaseException | None = None

        for attempt in range(self._max_retries + 1):
            per_call_timeout: float | None = context.metadata.get("_per_call_timeout")

            try:
                if per_call_timeout is not None:
                    response = await asyncio.wait_for(
                        self._llm_callable(request),
                        timeout=per_call_timeout,
                    )
                else:
                    response = await self._llm_callable(request)

                # Success — store metadata and return the real response object
                # (preserves response.usage so BudgetGuard can calculate actual cost)
                context.metadata["retry_count"] = attempt
                context.metadata["retry_history"] = history
                return response

            except asyncio.TimeoutError as exc:
                last_exc = exc
                err_class = ErrorClass.RETRYABLE

            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                err_class = classify_error(exc)

                # Check if the status code is in the user-provided extra list
                from brix.guards._retry_classifier import _get_status_code  # noqa: PLC0415

                if self._extra_retryable_codes:
                    status = _get_status_code(exc)
                    if status in self._extra_retryable_codes:
                        err_class = ErrorClass.RETRYABLE

            # Fatal errors: raise immediately, no retry
            if err_class == ErrorClass.FATAL:
                assert last_exc is not None
                raise last_exc

            # Compute backoff delay with jitter
            delay = min(
                self._backoff_base**attempt * (0.5 + random() * 0.5),
                self._max_backoff,
            )

            # Record attempt in history — use last_exc (Python 3.11+ deletes
            # except-bound variable after the except block exits)
            history.append(
                {
                    "attempt": attempt,
                    "error": str(last_exc),
                    "error_type": type(last_exc).__name__,
                    "delay": delay,
                }
            )

            # Check retry budget before sleeping
            elapsed = time.perf_counter() - budget_start
            if elapsed + delay > self._retry_budget:
                assert last_exc is not None
                raise BrixGuardError(
                    self.name,
                    f"retry budget of {self._retry_budget:.1f}s exhausted after "
                    f"{attempt + 1} attempt(s); history: {history}",
                ) from last_exc

            await asyncio.sleep(delay)

        # All retries exhausted
        assert last_exc is not None
        raise BrixGuardError(
            self.name,
            f"max retries ({self._max_retries}) exceeded; history: {history}",
        ) from last_exc

    async def post_call(
        self,
        request: CallRequest,
        response: CallResponse,
        context: ExecutionContext,
    ) -> CallResponse:
        """Pass-through post_call. RetryGuard does all work in pre_call."""
        return response


__all__ = ["RetryGuard"]
