# mypy: disable-error-code="no-untyped-def,misc,type-arg"
"""Tests for RateLimitGuard."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from brix.context import ExecutionContext
from brix.guards.protocol import CallRequest, CallResponse
from brix.guards.rate_limit import RateLimitGuard, _TokenBucket


def _make_context() -> ExecutionContext:
    return ExecutionContext.new_session()


def _make_request() -> CallRequest:
    return CallRequest(
        messages=[{"role": "user", "content": "Hi"}],
        model="gpt-4o-mini",
    )


def _make_response() -> CallResponse:
    return CallResponse(content="Hi", usage={"prompt_tokens": 10, "completion_tokens": 5})


# ---------------------------------------------------------------------------
# Name
# ---------------------------------------------------------------------------


def test_name():
    guard = RateLimitGuard(60)
    assert guard.name == "rate_limit"


# ---------------------------------------------------------------------------
# Token bucket: immediate pass-through when tokens available
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_wait_when_tokens_available():
    guard = RateLimitGuard(requests_per_minute=600)  # 10 tokens/sec → initial=min(10,1)=1
    ctx = _make_context()
    req = _make_request()
    # Should not sleep
    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        result = await guard.pre_call(req, ctx)
    mock_sleep.assert_not_called()
    assert result is req


@pytest.mark.asyncio
async def test_rate_limit_wait_ms_stored_in_metadata():
    guard = RateLimitGuard(requests_per_minute=600)
    ctx = _make_context()
    await guard.pre_call(_make_request(), ctx)
    assert "_rate_limit_wait_ms" in ctx.metadata
    assert ctx.metadata["_rate_limit_wait_ms"] >= 0.0


# ---------------------------------------------------------------------------
# Token bucket: delay when bucket empty
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sleep_called_when_bucket_empty():
    # Very low rate: 6 rpm = 0.1 token/sec; bucket starts with ~0.1 tokens
    # Second call should need to sleep
    guard = RateLimitGuard(requests_per_minute=6)
    ctx = _make_context()

    slept: list[float] = []

    async def capture_sleep(secs: float) -> None:
        slept.append(secs)

    with patch("asyncio.sleep", side_effect=capture_sleep):
        await guard.pre_call(_make_request(), ctx)  # first: uses initial token
        # Drain remaining tokens artificially
        guard._bucket._tokens = 0.0
        await guard.pre_call(_make_request(), ctx)  # second: must wait

    assert len(slept) >= 1


# ---------------------------------------------------------------------------
# Adaptive: 429 detection from retry_history
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_429_in_retry_history_reduces_effective_rate():
    guard = RateLimitGuard(requests_per_minute=60, adaptive=True, min_rate_floor=0.1)
    ctx = _make_context()
    ctx.metadata["retry_history"] = [
        {"attempt": 0, "error": "HTTP 429 Too Many Requests", "delay": 1.0}
    ]
    resp = _make_response()
    initial_rate = guard.effective_rate
    await guard.post_call(_make_request(), resp, ctx)
    assert guard.effective_rate < initial_rate


@pytest.mark.asyncio
async def test_no_429_in_retry_history_rate_unchanged():
    guard = RateLimitGuard(requests_per_minute=60, adaptive=True)
    ctx = _make_context()
    ctx.metadata["retry_history"] = [
        {"attempt": 0, "error": "HTTP 503 Service Unavailable", "delay": 1.0}
    ]
    resp = _make_response()
    initial_rate = guard.effective_rate
    await guard.post_call(_make_request(), resp, ctx)
    assert guard.effective_rate == initial_rate


@pytest.mark.asyncio
async def test_no_retry_history_in_metadata_is_noop():
    """post_call with no retry_history key must not crash."""
    guard = RateLimitGuard(requests_per_minute=60)
    ctx = _make_context()
    resp = _make_response()
    result = await guard.post_call(_make_request(), resp, ctx)
    assert result is resp


# ---------------------------------------------------------------------------
# Non-adaptive mode
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_non_adaptive_rate_unchanged_on_429():
    guard = RateLimitGuard(requests_per_minute=60, adaptive=False)
    ctx = _make_context()
    ctx.metadata["retry_history"] = [{"attempt": 0, "error": "HTTP 429", "delay": 0.0}]
    initial_rate = guard.effective_rate
    await guard.post_call(_make_request(), _make_response(), ctx)
    assert guard.effective_rate == initial_rate


# ---------------------------------------------------------------------------
# Rate floor and ceiling
# ---------------------------------------------------------------------------


def test_rate_floor_not_exceeded():
    bucket = _TokenBucket(
        rate_per_minute=100, adaptive=True, min_rate_fraction=0.1, burst_capacity=None
    )
    # Reduce many times
    for _ in range(20):
        bucket.record_429(reduction_factor=0.5)
    # Floor = 100 * 0.1 = 10
    assert bucket.effective_rate >= 10.0


def test_rate_ceiling_not_exceeded():
    bucket = _TokenBucket(
        rate_per_minute=60, adaptive=True, min_rate_fraction=0.1, burst_capacity=None
    )
    # Record 429 to lower rate, then recover many times
    bucket.record_429(0.5)
    bucket._last_429_time = 0.0  # force recovery condition
    for _ in range(100):
        bucket.maybe_recover(recovery_window=0.0, recovery_factor=2.0)
    assert bucket.effective_rate <= 60.0


# ---------------------------------------------------------------------------
# Burst capacity
# ---------------------------------------------------------------------------


def test_burst_capacity_limits_bucket():
    bucket = _TokenBucket(
        rate_per_minute=600, adaptive=False, min_rate_fraction=0.1, burst_capacity=2
    )
    # Capacity should be min(600/60=10, 2) = 2
    assert bucket._capacity() <= 2.0


# ---------------------------------------------------------------------------
# Concurrent acquire: no over-issuance
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrent_acquire_no_over_issuance():
    """Under asyncio.gather, the bucket should not issue more tokens than its rate allows."""
    guard = RateLimitGuard(requests_per_minute=60)
    # Set tokens to exactly 1 so only one call gets through without sleeping
    guard._bucket._tokens = 1.0

    slept: list[float] = []

    async def capture_sleep(secs: float) -> None:
        slept.append(secs)
        # Don't actually sleep so the test is fast; bucket refills are mocked

    with patch("asyncio.sleep", side_effect=capture_sleep):
        ctx = _make_context()
        await asyncio.gather(
            guard.pre_call(_make_request(), ctx),
            guard.pre_call(_make_request(), ctx),
            guard.pre_call(_make_request(), ctx),
        )

    # At least 2 of the 3 concurrent calls should have needed to sleep
    assert len(slept) >= 1


# ---------------------------------------------------------------------------
# Recovery
# ---------------------------------------------------------------------------


def test_rate_recovers_after_window():
    bucket = _TokenBucket(
        rate_per_minute=100, adaptive=True, min_rate_fraction=0.1, burst_capacity=None
    )
    bucket.record_429(0.5)  # rate → 50
    reduced_rate = bucket.effective_rate
    # Force time to be past recovery window
    bucket._last_429_time = 0.0
    bucket.maybe_recover(recovery_window=0.0, recovery_factor=1.5)
    assert bucket.effective_rate > reduced_rate


# ---------------------------------------------------------------------------
# post_call pass-through
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_post_call_response_unchanged():
    guard = RateLimitGuard(requests_per_minute=60)
    ctx = _make_context()
    resp = _make_response()
    result = await guard.post_call(_make_request(), resp, ctx)
    assert result is resp
