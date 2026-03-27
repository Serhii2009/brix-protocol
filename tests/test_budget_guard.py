"""Tests for BudgetGuard."""

from __future__ import annotations

import asyncio
import warnings
from unittest.mock import patch

import pytest

from brix.context import ExecutionContext
from brix.exceptions import BrixBudgetError
from brix.guards.budget import BudgetGuard
from brix.guards.protocol import CallRequest, CallResponse


def _make_context() -> ExecutionContext:
    return ExecutionContext.new_session()


def _make_request(model: str = "gpt-4o-mini") -> CallRequest:
    return CallRequest(
        messages=[{"role": "user", "content": "Hello"}],
        model=model,
    )


def _make_response(prompt_tokens: int = 10, completion_tokens: int = 5) -> CallResponse:
    return CallResponse(
        content="Hi",
        usage={
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    )


# ---------------------------------------------------------------------------
# Basic name and init
# ---------------------------------------------------------------------------


def test_name():
    guard = BudgetGuard(max_cost_usd=1.0)
    assert guard.name == "budget"


def test_invalid_strategy_raises():
    with pytest.raises(ValueError, match="strategy"):
        BudgetGuard(1.0, strategy="ignore")


def test_invalid_warning_threshold_raises():
    with pytest.raises(ValueError, match="warning_threshold"):
        BudgetGuard(1.0, warning_threshold=0.0)

    with pytest.raises(ValueError, match="warning_threshold"):
        BudgetGuard(1.0, warning_threshold=1.1)


# ---------------------------------------------------------------------------
# pre_call: budget checks
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_passes_when_under_budget():
    guard = BudgetGuard(max_cost_usd=10.0)
    ctx = _make_context()
    req = _make_request()
    result = await guard.pre_call(req, ctx)
    assert result is req


@pytest.mark.asyncio
async def test_blocks_when_budget_exceeded():
    guard = BudgetGuard(max_cost_usd=0.0)
    ctx = _make_context()
    req = _make_request(model="gpt-4o")  # non-zero price
    with pytest.raises(BrixBudgetError):
        await guard.pre_call(req, ctx)


@pytest.mark.asyncio
async def test_zero_budget_blocks_first_call():
    guard = BudgetGuard(max_cost_usd=0.0)
    ctx = _make_context()
    req = _make_request(model="gpt-4o")
    with pytest.raises(BrixBudgetError) as exc_info:
        await guard.pre_call(req, ctx)
    # Reason should mention both amounts
    assert "exceeding" in str(exc_info.value).lower() or "limit" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_strategy_warn_continues_despite_exceeding():
    guard = BudgetGuard(max_cost_usd=0.0, strategy="warn")
    ctx = _make_context()
    req = _make_request(model="gpt-4o")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = await guard.pre_call(req, ctx)
    assert result is req
    assert any("BudgetGuard" in str(w.message) for w in caught)


@pytest.mark.asyncio
async def test_warning_emitted_at_threshold():
    guard = BudgetGuard(max_cost_usd=1.0, warning_threshold=0.8)
    ctx = _make_context()
    ctx.session_cost_usd = 0.85  # already at 85%
    req = _make_request()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        await guard.pre_call(req, ctx)
    assert any("BudgetGuard" in str(w.message) for w in caught)


@pytest.mark.asyncio
async def test_warning_threshold_configurable():
    """At 50% threshold, warning fires at half budget."""
    guard = BudgetGuard(max_cost_usd=1.0, warning_threshold=0.5)
    ctx = _make_context()
    ctx.session_cost_usd = 0.55
    req = _make_request()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        await guard.pre_call(req, ctx)
    assert any("BudgetGuard" in str(w.message) for w in caught)


@pytest.mark.asyncio
async def test_estimated_cost_stored_in_metadata():
    guard = BudgetGuard(max_cost_usd=10.0)
    ctx = _make_context()
    req = _make_request(model="gpt-4o")
    await guard.pre_call(req, ctx)
    assert "estimated" in ctx.metadata.get("last_call_cost", {})
    assert ctx.metadata["last_call_cost"]["estimated"] >= 0.0


# ---------------------------------------------------------------------------
# post_call: actual cost accumulation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_actual_cost_stored_in_metadata_after_post_call():
    guard = BudgetGuard(max_cost_usd=10.0)
    ctx = _make_context()
    req = _make_request(model="gpt-4o-mini")
    resp = _make_response(prompt_tokens=100, completion_tokens=50)
    await guard.post_call(req, resp, ctx)
    assert ctx.metadata["last_call_cost"]["actual"] > 0.0
    assert ctx.metadata["last_call_cost"]["prompt_tokens"] == 100
    assert ctx.metadata["last_call_cost"]["completion_tokens"] == 50


@pytest.mark.asyncio
async def test_session_cost_accumulates_across_calls():
    guard = BudgetGuard(max_cost_usd=10.0)
    ctx = _make_context()
    req = _make_request(model="gpt-4o-mini")
    resp = _make_response(prompt_tokens=1000, completion_tokens=500)

    await guard.post_call(req, resp, ctx)
    cost_after_first = ctx.session_cost_usd
    assert cost_after_first > 0.0

    await guard.post_call(req, resp, ctx)
    assert ctx.session_cost_usd == pytest.approx(cost_after_first * 2)


@pytest.mark.asyncio
async def test_post_call_with_usage_none_is_noop():
    """Timed-out responses with usage=None must not crash or update cost."""
    guard = BudgetGuard(max_cost_usd=10.0)
    ctx = _make_context()
    req = _make_request()
    resp = CallResponse(content="", usage=None)
    result = await guard.post_call(req, resp, ctx)
    assert result is resp
    assert ctx.session_cost_usd == 0.0


@pytest.mark.asyncio
async def test_unknown_model_uses_zero_price_fallback():
    guard = BudgetGuard(max_cost_usd=10.0)
    ctx = _make_context()
    req = _make_request(model="totally-unknown-model-xyz")
    # Should not raise; zero-price fallback
    result = await guard.pre_call(req, ctx)
    assert result is req
    assert ctx.metadata["last_call_cost"]["estimated"] == 0.0


@pytest.mark.asyncio
async def test_session_cost_breakdown_populated():
    guard = BudgetGuard(max_cost_usd=10.0)
    ctx = _make_context()
    req = _make_request(model="gpt-4o-mini")
    resp = _make_response(100, 50)

    await guard.post_call(req, resp, ctx)
    breakdown = ctx.metadata.get("session_cost_breakdown", [])
    assert len(breakdown) == 1
    assert "actual" in breakdown[0]
    assert "model" in breakdown[0]


@pytest.mark.asyncio
async def test_anthropic_style_usage_keys():
    """post_call must handle input_tokens/output_tokens keys (Anthropic)."""
    guard = BudgetGuard(max_cost_usd=10.0)
    ctx = _make_context()
    req = _make_request(model="claude-3-5-sonnet-20241022")
    resp = CallResponse(
        content="Hi",
        usage={"input_tokens": 100, "output_tokens": 50},
    )
    await guard.post_call(req, resp, ctx)
    assert ctx.session_cost_usd > 0.0
