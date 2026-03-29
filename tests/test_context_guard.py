"""Tests for ContextGuard."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

import pytest

from brix.context import ExecutionContext
from brix.exceptions import BrixConfigurationError
from brix.guards.context import ContextGuard, _count_tokens
from brix.guards.protocol import CallRequest, CallResponse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_context() -> ExecutionContext:
    return ExecutionContext.new_session()


def _make_request(messages: list[dict[str, Any]] | None = None) -> CallRequest:
    if messages is None:
        messages = [{"role": "user", "content": "hello"}]
    return CallRequest(messages=messages, model="gpt-4o", kwargs={})


def _make_response() -> CallResponse:
    return CallResponse(content="ok")


def _make_llm_callable(
    summary: str,
) -> Callable[[CallRequest], Awaitable[CallResponse]]:
    """Return an async callable that always responds with the given summary."""
    call_count = [0]

    async def _callable(request: CallRequest) -> CallResponse:
        call_count[0] += 1
        return CallResponse(content=summary)

    _callable.call_count = call_count  # type: ignore[attr-defined]
    return _callable  # type: ignore[return-value]


def _over_budget_messages(n: int = 8) -> list[dict[str, Any]]:
    """Generate messages that will exceed a small (50-token) budget."""
    msgs: list[dict[str, Any]] = []
    for i in range(n):
        role = "user" if i % 2 == 0 else "assistant"
        msgs.append({"role": role, "content": f"This is message number {i} with content."})
    return msgs


# ---------------------------------------------------------------------------
# Construction / configuration
# ---------------------------------------------------------------------------


def test_name() -> None:
    guard = ContextGuard(1000)
    assert guard.name == "context"


def test_summarize_requires_llm_callable() -> None:
    with pytest.raises(BrixConfigurationError, match="llm_callable"):
        ContextGuard(1000, strategy="summarize", llm_callable=None)


def test_invalid_strategy_raises_config_error() -> None:
    with pytest.raises(BrixConfigurationError, match="strategy"):
        ContextGuard(1000, strategy="unknown")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# No compression when under limit
# ---------------------------------------------------------------------------


async def test_no_compression_under_limit() -> None:
    guard = ContextGuard(10_000, reserve_tokens=0)
    ctx = _make_context()
    req = _make_request([{"role": "user", "content": "hi"}])

    result = await guard.pre_call(req, ctx)
    assert result.messages == req.messages
    assert "_context_compressed" not in ctx.metadata


# ---------------------------------------------------------------------------
# sliding_window strategy
# ---------------------------------------------------------------------------


async def test_sliding_window_removes_excess() -> None:
    guard = ContextGuard(50, strategy="sliding_window", reserve_tokens=0)
    ctx = _make_context()
    messages = _over_budget_messages(8)
    req = _make_request(messages)

    result = await guard.pre_call(req, ctx)
    assert len(result.messages) < len(messages)
    assert ctx.metadata.get("_context_compressed") is True


async def test_system_prompt_preserved() -> None:
    guard = ContextGuard(50, strategy="sliding_window", reserve_tokens=0)
    ctx = _make_context()
    system_msg = {"role": "system", "content": "You are helpful."}
    messages = [system_msg] + _over_budget_messages(6)
    req = _make_request(messages)

    result = await guard.pre_call(req, ctx)
    assert any(m["role"] == "system" and m["content"] == "You are helpful." for m in result.messages)


async def test_current_query_preserved() -> None:
    guard = ContextGuard(50, strategy="sliding_window", reserve_tokens=0)
    ctx = _make_context()
    messages = _over_budget_messages(6)
    messages.append({"role": "user", "content": "FINAL QUERY"})
    req = _make_request(messages)

    result = await guard.pre_call(req, ctx)
    assert result.messages[-1]["role"] == "user"
    assert result.messages[-1]["content"] == "FINAL QUERY"


async def test_token_count_never_exceeds_limit() -> None:
    max_tokens = 80
    guard = ContextGuard(max_tokens, strategy="sliding_window", reserve_tokens=0)
    ctx = _make_context()
    messages = _over_budget_messages(12)
    req = _make_request(messages)

    result = await guard.pre_call(req, ctx)
    assert _count_tokens(result.messages, req.model) <= max_tokens


async def test_reserve_tokens_respected() -> None:
    max_tokens = 100
    reserve = 30
    guard = ContextGuard(max_tokens, strategy="sliding_window", reserve_tokens=reserve)
    ctx = _make_context()
    messages = _over_budget_messages(10)
    req = _make_request(messages)

    result = await guard.pre_call(req, ctx)
    assert _count_tokens(result.messages, req.model) <= max_tokens - reserve


# ---------------------------------------------------------------------------
# Metadata recording
# ---------------------------------------------------------------------------


async def test_metadata_records_stats() -> None:
    guard = ContextGuard(50, strategy="sliding_window", reserve_tokens=0)
    ctx = _make_context()
    req = _make_request(_over_budget_messages(8))

    await guard.pre_call(req, ctx)

    assert ctx.metadata["_context_compressed"] is True
    assert ctx.metadata["_context_strategy_used"] == "sliding_window"
    assert isinstance(ctx.metadata["_tokens_before"], int)
    assert isinstance(ctx.metadata["_tokens_after"], int)
    assert ctx.metadata["_tokens_after"] < ctx.metadata["_tokens_before"]


# ---------------------------------------------------------------------------
# summarize strategy
# ---------------------------------------------------------------------------


async def test_summarize_strategy_calls_llm() -> None:
    llm = _make_llm_callable("Brief summary.")
    guard = ContextGuard(50, strategy="summarize", reserve_tokens=0, llm_callable=llm)
    ctx = _make_context()
    req = _make_request(_over_budget_messages(8))

    await guard.pre_call(req, ctx)
    assert llm.call_count[0] >= 1  # type: ignore[attr-defined]


async def test_summarize_fallback_to_sliding_window() -> None:
    """When the summary + remaining history is still over budget, sliding_window is applied."""
    # Summary is concise (~15 tokens) but the remaining history is large enough
    # that summary + remaining_history + last_user still exceeds the budget (80).
    concise_summary = "France capital is Paris. Germany capital is Berlin."
    llm = _make_llm_callable(concise_summary)
    # budget = 80 tokens; last_user + summary ≈ 30 tokens, so history can tip it over
    guard = ContextGuard(80, strategy="summarize", reserve_tokens=0, llm_callable=llm)
    ctx = _make_context()

    # Build messages whose total exceeds budget=80
    messages = [
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris, a beautiful city."},
        {"role": "user", "content": "And Germany? Tell me more about it."},
        {"role": "assistant", "content": "Germany capital is Berlin, a vibrant metropolis."},
        {"role": "user", "content": "And Italy? Elaborate please with many details here."},
        {"role": "assistant", "content": "The capital of Italy is Rome, an ancient and storied city."},
        {"role": "user", "content": "Final question"},
    ]
    req = _make_request(messages)

    # Verify the messages are actually over budget (precondition for this test)
    assert _count_tokens(messages, req.model) > 80

    result = await guard.pre_call(req, ctx)
    # The fallback should have fired (summary + remaining history > budget)
    # and the final result must be within budget
    assert ctx.metadata.get("_context_compressed") is True
    # Verify fallback was triggered and/or token limit was met
    assert _count_tokens(result.messages, req.model) <= 80


# ---------------------------------------------------------------------------
# importance strategy
# ---------------------------------------------------------------------------


async def test_importance_keeps_tool_calls() -> None:
    guard = ContextGuard(60, strategy="importance", reserve_tokens=0)
    ctx = _make_context()

    tool_msg = {"role": "assistant", "content": "calling tool", "tool_calls": [{"id": "1"}]}
    filler_msg = {"role": "assistant", "content": "filler response"}
    messages = [filler_msg, filler_msg, tool_msg, {"role": "user", "content": "go"}]
    req = _make_request(messages)

    result = await guard.pre_call(req, ctx)
    # The tool_calls message should be present (score=3 > filler score=1)
    assert any("tool_calls" in m for m in result.messages)


async def test_importance_keeps_tool_results() -> None:
    guard = ContextGuard(60, strategy="importance", reserve_tokens=0)
    ctx = _make_context()

    tool_result = {"role": "tool", "content": "tool output"}
    filler_msg = {"role": "assistant", "content": "filler filler filler filler"}
    messages = [filler_msg, filler_msg, tool_result, {"role": "user", "content": "done"}]
    req = _make_request(messages)

    result = await guard.pre_call(req, ctx)
    assert any(m.get("role") == "tool" for m in result.messages)
