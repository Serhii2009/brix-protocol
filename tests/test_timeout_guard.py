# mypy: disable-error-code="no-untyped-def,misc,type-arg"
"""Tests for TimeoutGuard."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from brix.context import ExecutionContext
from brix.exceptions import BrixTimeoutError
from brix.guards.protocol import CallRequest, CallResponse
from brix.guards.timeout import TimeoutGuard


def _make_context(age_seconds: float = 0.0) -> ExecutionContext:
    ctx = ExecutionContext.new_session()
    if age_seconds:
        ctx.session_start = datetime.now(timezone.utc) - timedelta(seconds=age_seconds)
    return ctx


def _make_request() -> CallRequest:
    return CallRequest(
        messages=[{"role": "user", "content": "Hi"}],
        model="gpt-4o-mini",
    )


def _make_response() -> CallResponse:
    return CallResponse(content="Hi")


# ---------------------------------------------------------------------------
# Basic name and init
# ---------------------------------------------------------------------------


def test_name():
    guard = TimeoutGuard(per_call=5.0)
    assert guard.name == "timeout"


def test_invalid_on_timeout_raises():
    with pytest.raises(ValueError, match="on_timeout"):
        TimeoutGuard(per_call=5.0, on_timeout="ignore")


# ---------------------------------------------------------------------------
# Total session timeout
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_total_timeout_fires_when_exceeded():
    guard = TimeoutGuard(total=1.0, on_timeout="raise")
    ctx = _make_context(age_seconds=5.0)  # already 5s old
    req = _make_request()
    with pytest.raises(BrixTimeoutError) as exc_info:
        await guard.pre_call(req, ctx)
    assert "total" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_total_timeout_return_partial():
    guard = TimeoutGuard(total=1.0, on_timeout="return_partial")
    ctx = _make_context(age_seconds=5.0)
    req = _make_request()
    result = await guard.pre_call(req, ctx)
    assert isinstance(result, CallResponse)
    assert result.raw["timed_out"] is True
    assert result.raw["level"] == "total"
    assert result.usage is None


@pytest.mark.asyncio
async def test_total_timeout_does_not_fire_when_within_limit():
    guard = TimeoutGuard(total=60.0, on_timeout="raise")
    ctx = _make_context(age_seconds=1.0)
    req = _make_request()
    result = await guard.pre_call(req, ctx)
    assert result is req


# ---------------------------------------------------------------------------
# Per-step timeout
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_per_step_timeout_fires_when_exceeded():
    guard = TimeoutGuard(per_step=1.0, on_timeout="raise")
    ctx = _make_context()
    # Simulate previous step that started 10s ago
    ctx.metadata["_step_start"] = datetime.now(timezone.utc) - timedelta(seconds=10.0)
    req = _make_request()
    with pytest.raises(BrixTimeoutError) as exc_info:
        await guard.pre_call(req, ctx)
    assert "step" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_per_step_timeout_does_not_fire_on_first_call():
    """First call has no _step_start in metadata."""
    guard = TimeoutGuard(per_step=1.0, on_timeout="raise")
    ctx = _make_context()
    req = _make_request()
    result = await guard.pre_call(req, ctx)
    assert result is req


@pytest.mark.asyncio
async def test_per_step_fires_but_total_does_not():
    """per_step fires, but total (larger) does not."""
    guard = TimeoutGuard(per_step=1.0, total=9999.0, on_timeout="raise")
    ctx = _make_context(age_seconds=0.1)
    ctx.metadata["_step_start"] = datetime.now(timezone.utc) - timedelta(seconds=10.0)
    req = _make_request()
    with pytest.raises(BrixTimeoutError) as exc_info:
        await guard.pre_call(req, ctx)
    assert "step" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_step_start_set_after_successful_pre_call():
    guard = TimeoutGuard(per_step=60.0)
    ctx = _make_context()
    req = _make_request()
    await guard.pre_call(req, ctx)
    assert "_step_start" in ctx.metadata


@pytest.mark.asyncio
async def test_step_start_cleared_by_post_call():
    guard = TimeoutGuard(per_step=60.0)
    ctx = _make_context()
    req = _make_request()
    resp = _make_response()
    await guard.pre_call(req, ctx)
    assert "_step_start" in ctx.metadata
    await guard.post_call(req, resp, ctx)
    assert "_step_start" not in ctx.metadata


# ---------------------------------------------------------------------------
# Per-call timeout metadata
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_per_call_timeout_stored_in_metadata():
    guard = TimeoutGuard(per_call=5.0)
    ctx = _make_context()
    req = _make_request()
    await guard.pre_call(req, ctx)
    assert ctx.metadata.get("_per_call_timeout") == 5.0


@pytest.mark.asyncio
async def test_per_call_timeout_absent_when_not_configured():
    guard = TimeoutGuard(per_step=60.0)  # no per_call
    ctx = _make_context()
    req = _make_request()
    await guard.pre_call(req, ctx)
    assert "_per_call_timeout" not in ctx.metadata


@pytest.mark.asyncio
async def test_stale_per_call_timeout_cleared():
    """If a previous guard config set _per_call_timeout, a new guard without it clears it."""
    guard = TimeoutGuard(per_step=60.0)  # no per_call
    ctx = _make_context()
    ctx.metadata["_per_call_timeout"] = 99.0  # stale value from previous run
    req = _make_request()
    await guard.pre_call(req, ctx)
    assert "_per_call_timeout" not in ctx.metadata


# ---------------------------------------------------------------------------
# Timeout config observability
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_timeout_config_stored_in_metadata():
    guard = TimeoutGuard(per_call=5.0, per_step=30.0, total=300.0)
    ctx = _make_context()
    req = _make_request()
    await guard.pre_call(req, ctx)
    cfg = ctx.metadata.get("_timeout_config", {})
    assert cfg["per_call"] == 5.0
    assert cfg["per_step"] == 30.0
    assert cfg["total"] == 300.0


# ---------------------------------------------------------------------------
# All three levels simultaneously
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_all_three_timeout_levels_pass_on_clean_call():
    guard = TimeoutGuard(per_call=10.0, per_step=60.0, total=300.0, on_timeout="raise")
    ctx = _make_context(age_seconds=1.0)
    req = _make_request()
    result = await guard.pre_call(req, ctx)
    assert result is req


@pytest.mark.asyncio
async def test_post_call_pass_through():
    guard = TimeoutGuard(per_call=10.0)
    ctx = _make_context()
    req = _make_request()
    resp = _make_response()
    result = await guard.post_call(req, resp, ctx)
    assert result is resp
