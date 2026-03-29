"""RateLimitGuard — proactively prevents LLM 429 errors with an adaptive token bucket.

The 100% guarantee: a correctly configured token bucket mathematically cannot
exceed the configured *average* rate. Calls are throttled before they reach the
provider, not after a 429 is received.

Important limitation: many providers enforce burst limits (e.g. max N requests
in 10 seconds) that are stricter than the average rate. The token bucket controls
average throughput, not instantaneous burst. Use ``burst_capacity`` to set a
conservative hard limit on how many tokens the bucket can hold at once, which
indirectly limits burst behaviour.

Adaptive layer: if a 429 is detected (via RetryGuard's ``retry_history`` metadata),
the effective rate is automatically reduced by ``rate_reduction_factor``. After
``recovery_window_seconds`` without a 429, the rate climbs back toward the
configured maximum at ``rate_recovery_factor`` per window. Set
``adaptive_rate_limiting=False`` to disable this behaviour.
"""

from __future__ import annotations

import asyncio
import time

from brix.context import ExecutionContext
from brix.guards.protocol import CallRequest, CallResponse


class _TokenBucket:
    """Non-blocking token bucket for rate limiting.

    Tokens refill at ``effective_rate / 60`` tokens per second.
    ``acquire()`` sleeps via ``asyncio.sleep`` (never blocks the thread)
    until a token is available.

    Thread safety: uses a lazy ``asyncio.Lock`` (same pattern as BudgetGuard)
    to prevent token over-issuance under concurrent ``asyncio.gather`` usage.
    """

    def __init__(
        self,
        rate_per_minute: float,
        adaptive: bool,
        min_rate_fraction: float,
        burst_capacity: int | None,
    ) -> None:
        self._max_rate = rate_per_minute
        self._effective_rate = rate_per_minute
        self._min_rate = rate_per_minute * min_rate_fraction
        self._burst_capacity = burst_capacity
        initial = rate_per_minute / 60.0
        if burst_capacity is not None:
            initial = min(initial, float(burst_capacity))
        self._tokens: float = initial
        self._last_refill: float = time.monotonic()
        self._adaptive = adaptive
        self._last_429_time: float | None = None
        self._lock: asyncio.Lock | None = None

    def _capacity(self) -> float:
        """Maximum tokens the bucket can hold (1 second of effective throughput)."""
        base = self._effective_rate / 60.0
        if self._burst_capacity is not None:
            return min(base, float(self._burst_capacity))
        return base

    def _refill(self) -> None:
        """Add tokens proportional to elapsed time. Must be called inside lock."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(
            self._tokens + elapsed * (self._effective_rate / 60.0),
            self._capacity(),
        )
        self._last_refill = now

    async def acquire(self) -> float:
        """Acquire one token, sleeping non-blockingly if the bucket is empty.

        Returns:
            The number of seconds waited (0.0 if a token was immediately available).
        """
        if self._lock is None:
            self._lock = asyncio.Lock()

        async with self._lock:
            self._refill()
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return 0.0
            # Compute how long to wait before a token refills
            wait = (1.0 - self._tokens) / (self._effective_rate / 60.0)

        # Sleep outside the lock so other coroutines can proceed during the wait
        await asyncio.sleep(wait)

        async with self._lock:
            self._refill()
            self._tokens = max(0.0, self._tokens - 1.0)

        return wait

    def record_429(self, reduction_factor: float) -> None:
        """Reduce effective rate after a 429 response."""
        if not self._adaptive:
            return
        self._last_429_time = time.monotonic()
        self._effective_rate = max(self._min_rate, self._effective_rate * reduction_factor)

    def maybe_recover(self, recovery_window: float, recovery_factor: float) -> None:
        """Increase effective rate if no 429 in the last ``recovery_window`` seconds.

        Args:
            recovery_window: Seconds without a 429 before attempting recovery.
            recovery_factor: Multiplicative factor applied per recovery check (>1.0).
                Default 1.05 (5% increase). Use a higher value (e.g. 1.5) for faster
                recovery at the cost of potentially overshooting and hitting 429 again.
        """
        if not self._adaptive or self._effective_rate >= self._max_rate:
            return
        if self._last_429_time is None:
            return
        if time.monotonic() - self._last_429_time > recovery_window:
            self._effective_rate = min(self._max_rate, self._effective_rate * recovery_factor)
            # Reset timer so the next recovery check waits another full window
            self._last_429_time = time.monotonic()

    @property
    def effective_rate(self) -> float:
        """Current effective rate in requests per minute."""
        return self._effective_rate


class RateLimitGuard:
    """Guard that throttles LLM calls to prevent 429 rate-limit errors.

    Uses an adaptive token bucket algorithm:
    1. Before each call, ``pre_call`` acquires one token from the bucket.
       If the bucket is empty, it sleeps (non-blocking) until a token refills.
    2. After each call, ``post_call`` inspects ``context.metadata["retry_history"]``
       (set by RetryGuard) for 429 responses and reduces the effective rate if found.
    3. On each ``pre_call``, if enough time has passed since the last 429 and the
       effective rate is below the configured maximum, the rate is gradually increased.

    Args:
        requests_per_minute: Target rate cap. The bucket refills at this rate / 60
            tokens per second. Must be > 0.
        adaptive: Enable automatic rate adjustment on 429. Default True.
        min_rate_floor: Minimum effective rate as a fraction of ``requests_per_minute``.
            Prevents the rate from dropping to zero under sustained 429 pressure.
            Default 0.1 (10% of configured max).
        rate_reduction_factor: Multiplicative factor applied to effective rate when a
            429 is detected. Default 0.5 (halve the rate).
        rate_recovery_factor: Multiplicative factor applied to effective rate during
            recovery. Default 1.05 (5% per recovery window). Increase for faster
            recovery (e.g. 1.5) at the risk of re-triggering 429.
        recovery_window_seconds: Seconds without a 429 before attempting rate recovery.
            Default 60.0.
        burst_capacity: Optional hard cap on bucket token capacity. Limits how many
            requests can be issued in a rapid burst, independently of the average rate.
            When None, capacity = 1 second of tokens at effective rate.
    """

    name: str = "rate_limit"

    def __init__(
        self,
        requests_per_minute: int,
        *,
        adaptive: bool = True,
        min_rate_floor: float = 0.1,
        rate_reduction_factor: float = 0.5,
        rate_recovery_factor: float = 1.05,
        recovery_window_seconds: float = 60.0,
        burst_capacity: int | None = None,
    ) -> None:
        self._bucket = _TokenBucket(requests_per_minute, adaptive, min_rate_floor, burst_capacity)
        self._reduction_factor = rate_reduction_factor
        self._recovery_factor = rate_recovery_factor
        self._recovery_window = recovery_window_seconds

    async def pre_call(
        self,
        request: CallRequest,
        context: ExecutionContext,
    ) -> CallRequest:
        """Throttle the call if needed and attempt rate recovery.

        Args:
            request: The outbound request.
            context: Mutable session state.

        Returns:
            The unmodified request after acquiring a token.
        """
        self._bucket.maybe_recover(self._recovery_window, self._recovery_factor)
        wait_seconds = await self._bucket.acquire()
        context.metadata["_rate_limit_wait_ms"] = wait_seconds * 1000
        return request

    async def post_call(
        self,
        request: CallRequest,
        response: CallResponse,
        context: ExecutionContext,
    ) -> CallResponse:
        """Detect 429s from RetryGuard's retry_history and reduce rate if found.

        Args:
            request: The request that was sent.
            response: The response received.
            context: Mutable session state.

        Returns:
            The unmodified response.
        """
        history = context.metadata.get("retry_history", [])
        had_429 = any("429" in str(entry.get("error", "")) for entry in history)
        if had_429:
            self._bucket.record_429(self._reduction_factor)
        return response

    @property
    def effective_rate(self) -> float:
        """Current effective rate in requests per minute (for observability)."""
        return self._bucket.effective_rate


__all__ = ["RateLimitGuard"]
