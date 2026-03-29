# mypy: disable-error-code="no-untyped-def,misc,type-arg"
"""Tests for RetryGuard."""

from __future__ import annotations

import asyncio
import warnings
from unittest.mock import AsyncMock

import pytest

from brix.context import ExecutionContext
from brix.exceptions import BrixBudgetError, BrixGuardError
from brix.guards.budget import BudgetGuard
from brix.guards.protocol import CallRequest, CallResponse
from brix.guards.retry import RetryGuard
from brix.guards.timeout import TimeoutGuard


def _make_context() -> ExecutionContext:
    return ExecutionContext.new_session()


def _make_request() -> CallRequest:
    return CallRequest(
        messages=[{"role": "user", "content": "Hi"}],
        model="gpt-4o-mini",
    )


def _make_response() -> CallResponse:
    return CallResponse(
        content="Hello",
        usage={"prompt_tokens": 10, "completion_tokens": 5},
    )


def _make_callable(responses: list) -> AsyncMock:
    """Build a callable that raises or returns items from responses in order."""
    mock = AsyncMock(side_effect=responses)
    return mock


class _FakeHTTPError(Exception):
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code
        super().__init__(f"HTTP {status_code}")


# ---------------------------------------------------------------------------
# Basic name and construction
# ---------------------------------------------------------------------------


def test_name():
    guard = RetryGuard(AsyncMock())
    assert guard.name == "retry"


# ---------------------------------------------------------------------------
# Success path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_success_on_first_attempt():
    resp = _make_response()
    guard = RetryGuard(_make_callable([resp]), max_retries=3)
    ctx = _make_context()
    req = _make_request()
    result = await guard.pre_call(req, ctx)
    assert result is resp
    assert ctx.metadata["retry_count"] == 0
    assert ctx.metadata["retry_history"] == []


@pytest.mark.asyncio
async def test_success_on_second_attempt_stores_retry_count():
    resp = _make_response()
    error = _FakeHTTPError(503)
    guard = RetryGuard(
        _make_callable([error, resp]),
        max_retries=3,
        backoff_base=0.0,  # zero delay for speed
    )
    ctx = _make_context()
    req = _make_request()
    result = await guard.pre_call(req, ctx)
    assert result is resp
    assert ctx.metadata["retry_count"] == 1
    assert len(ctx.metadata["retry_history"]) == 1


# ---------------------------------------------------------------------------
# Retry on transient errors
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retryable_error_retries_up_to_max():
    resp = _make_response()
    errors = [_FakeHTTPError(429)] * 3
    guard = RetryGuard(
        _make_callable([*errors, resp]),
        max_retries=3,
        backoff_base=0.0,
    )
    ctx = _make_context()
    req = _make_request()
    result = await guard.pre_call(req, ctx)
    assert result is resp
    assert ctx.metadata["retry_count"] == 3


@pytest.mark.asyncio
async def test_connection_error_is_retryable():
    resp = _make_response()
    guard = RetryGuard(
        _make_callable([ConnectionError("network"), resp]),
        max_retries=2,
        backoff_base=0.0,
    )
    ctx = _make_context()
    req = _make_request()
    result = await guard.pre_call(req, ctx)
    assert result is resp


@pytest.mark.asyncio
async def test_asyncio_timeout_error_is_retryable():
    resp = _make_response()
    guard = RetryGuard(
        _make_callable([asyncio.TimeoutError(), resp]),
        max_retries=2,
        backoff_base=0.0,
    )
    ctx = _make_context()
    req = _make_request()
    result = await guard.pre_call(req, ctx)
    assert result is resp


# ---------------------------------------------------------------------------
# Fatal errors
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fatal_error_raises_immediately_no_retry():
    fatal = _FakeHTTPError(401)
    mock = _make_callable([fatal])
    guard = RetryGuard(mock, max_retries=3, backoff_base=0.0)
    ctx = _make_context()
    req = _make_request()
    with pytest.raises(_FakeHTTPError):
        await guard.pre_call(req, ctx)
    assert mock.call_count == 1  # no retry


@pytest.mark.asyncio
async def test_403_is_fatal():
    mock = _make_callable([_FakeHTTPError(403)])
    guard = RetryGuard(mock, max_retries=3, backoff_base=0.0)
    ctx = _make_context()
    req = _make_request()
    with pytest.raises(_FakeHTTPError):
        await guard.pre_call(req, ctx)
    assert mock.call_count == 1


# ---------------------------------------------------------------------------
# Max retries exhausted
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_max_retries_exhausted_raises_brix_guard_error():
    error = _FakeHTTPError(503)
    guard = RetryGuard(
        _make_callable([error] * 10),
        max_retries=2,
        backoff_base=0.0,
    )
    ctx = _make_context()
    req = _make_request()
    with pytest.raises(BrixGuardError) as exc_info:
        await guard.pre_call(req, ctx)
    assert "max retries" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_retry_history_in_exception():
    error = _FakeHTTPError(503)
    guard = RetryGuard(
        _make_callable([error] * 10),
        max_retries=2,
        backoff_base=0.0,
    )
    ctx = _make_context()
    req = _make_request()
    with pytest.raises(BrixGuardError) as exc_info:
        await guard.pre_call(req, ctx)
    assert "history" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_zero_max_retries_makes_exactly_one_attempt():
    error = _FakeHTTPError(503)
    mock = _make_callable([error])
    guard = RetryGuard(mock, max_retries=0, backoff_base=0.0)
    ctx = _make_context()
    req = _make_request()
    with pytest.raises(BrixGuardError):
        await guard.pre_call(req, ctx)
    assert mock.call_count == 1


# ---------------------------------------------------------------------------
# Retry budget
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retry_budget_exhaustion_raises():
    error = _FakeHTTPError(503)
    guard = RetryGuard(
        _make_callable([error] * 10),
        max_retries=5,
        backoff_base=100.0,  # large delay
        retry_budget_seconds=0.001,  # tiny budget
    )
    ctx = _make_context()
    req = _make_request()
    with pytest.raises(BrixGuardError) as exc_info:
        await guard.pre_call(req, ctx)
    assert "budget" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# Unknown errors
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unknown_error_triggers_warning_and_is_retried():
    resp = _make_response()

    class WeirdError(Exception):
        pass

    guard = RetryGuard(
        _make_callable([WeirdError("unexpected"), resp]),
        max_retries=2,
        backoff_base=0.0,
    )
    ctx = _make_context()
    req = _make_request()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = await guard.pre_call(req, ctx)
    assert result is resp
    assert any("RetryGuard" in str(w.message) for w in caught)


# ---------------------------------------------------------------------------
# extra retry_on codes
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_extra_retry_on_codes():
    """HTTP 418 is not in default RETRYABLE set, but can be added via retry_on."""
    resp = _make_response()
    guard = RetryGuard(
        _make_callable([_FakeHTTPError(418), resp]),
        max_retries=2,
        backoff_base=0.0,
        retry_on=[418],
    )
    ctx = _make_context()
    req = _make_request()
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        result = await guard.pre_call(req, ctx)
    assert result is resp


# ---------------------------------------------------------------------------
# post_call is pass-through
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_post_call_pass_through():
    guard = RetryGuard(AsyncMock())
    ctx = _make_context()
    req = _make_request()
    resp = _make_response()
    result = await guard.post_call(req, resp, ctx)
    assert result is resp


# ---------------------------------------------------------------------------
# Per-call timeout from context.metadata
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_respects_per_call_timeout_from_metadata():
    """RetryGuard reads _per_call_timeout from context.metadata."""

    async def slow_callable(request: CallRequest) -> CallResponse:
        await asyncio.sleep(10)
        return _make_response()

    guard = RetryGuard(slow_callable, max_retries=1, backoff_base=0.0)
    ctx = _make_context()
    ctx.metadata["_per_call_timeout"] = 0.05  # very short
    req = _make_request()
    with pytest.raises(BrixGuardError):
        await guard.pre_call(req, ctx)


# ---------------------------------------------------------------------------
# Integration: BudgetGuard + TimeoutGuard + RetryGuard ordering
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_budget_guard_blocks_before_retry_guard_calls_llm():
    """BudgetGuard.pre_call runs first; RetryGuard's LLM callable must not be called."""
    from brix.chain import InterceptorChain
    from brix.client import build_llm_callable

    class _MockLLM:
        async def complete(self, prompt, *, system, temperature, max_tokens):
            return "ok"

    llm_client = _MockLLM()
    llm_callable = build_llm_callable(llm_client)
    retry_llm_mock = AsyncMock(return_value=_make_response())

    budget_guard = BudgetGuard(max_cost_usd=0.0, strategy="block")
    retry_guard = RetryGuard(retry_llm_mock, max_retries=3, backoff_base=0.0)

    chain = InterceptorChain([budget_guard, retry_guard])
    ctx = _make_context()
    req = CallRequest(
        messages=[{"role": "user", "content": "Hi"}],
        model="gpt-4o",  # non-zero price so budget is exceeded
    )

    with pytest.raises(BrixBudgetError):
        await chain.execute(req, ctx, llm_callable)

    retry_llm_mock.assert_not_called()


@pytest.mark.asyncio
async def test_timeout_guard_sets_metadata_before_retry_reads_it():
    """TimeoutGuard.pre_call runs before RetryGuard.pre_call, so _per_call_timeout is set."""
    captured_timeout: list[float | None] = []

    async def capturing_callable(request: CallRequest) -> CallResponse:
        return _make_response()

    class _TimeoutCapturingRetryGuard(RetryGuard):
        async def pre_call(self, request, context):
            captured_timeout.append(context.metadata.get("_per_call_timeout"))
            return await super().pre_call(request, context)

    timeout_guard = TimeoutGuard(per_call=7.5)
    retry_guard = _TimeoutCapturingRetryGuard(capturing_callable, max_retries=0, backoff_base=0.0)

    from brix.chain import InterceptorChain
    from brix.client import build_llm_callable

    class _MockLLM:
        async def complete(self, prompt, *, system, temperature, max_tokens):
            return "ok"

    chain = InterceptorChain([timeout_guard, retry_guard])
    ctx = _make_context()
    req = _make_request()

    await chain.execute(req, ctx, build_llm_callable(_MockLLM()))

    assert captured_timeout == [7.5]
