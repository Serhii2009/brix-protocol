"""Tests for LoopGuard."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from brix.context import ExecutionContext
from brix.exceptions import BrixConfigurationError, BrixLoopError
from brix.guards.loop import LoopGuard
from brix.guards.protocol import CallRequest, CallResponse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_context() -> ExecutionContext:
    return ExecutionContext.new_session()


def _make_request() -> CallRequest:
    return CallRequest(messages=[{"role": "user", "content": "test"}], model="gpt-4o", kwargs={})


def _make_response(content: str = "response") -> CallResponse:
    return CallResponse(content=content)


def _make_guard(**kwargs: object) -> LoopGuard:
    return LoopGuard(**kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Construction / configuration
# ---------------------------------------------------------------------------


def test_name() -> None:
    guard = _make_guard()
    assert guard.name == "loop"


def test_invalid_on_loop_raises_config_error() -> None:
    with pytest.raises(BrixConfigurationError, match="on_loop"):
        _make_guard(on_loop="unknown")


def test_semantic_raises_config_error_when_not_installed() -> None:
    with patch.dict("sys.modules", {"sentence_transformers": None}):
        with pytest.raises(BrixConfigurationError, match="sentence-transformers"):
            _make_guard(semantic_detection=True)


# ---------------------------------------------------------------------------
# Exact loop detection
# ---------------------------------------------------------------------------


async def test_no_loop_below_threshold() -> None:
    guard = _make_guard(exact_threshold=3, on_loop="raise")
    ctx = _make_context()
    req = _make_request()
    resp = _make_response("same")

    await guard.post_call(req, resp, ctx)  # count = 1
    await guard.post_call(req, resp, ctx)  # count = 2
    # Threshold is 3 — no raise yet
    assert not ctx.metadata.get("_loop_inject_next", False)


async def test_exact_loop_detected_at_threshold() -> None:
    guard = _make_guard(exact_threshold=3, on_loop="raise")
    ctx = _make_context()
    req = _make_request()
    resp = _make_response("same")

    await guard.post_call(req, resp, ctx)
    await guard.post_call(req, resp, ctx)
    with pytest.raises(BrixLoopError):
        await guard.post_call(req, resp, ctx)


async def test_exact_loop_uses_rolling_window() -> None:
    """Loop should be detected even after a run of unique responses."""
    guard = _make_guard(exact_threshold=3, on_loop="raise")
    ctx = _make_context()
    req = _make_request()

    for i in range(5):
        await guard.post_call(req, _make_response(f"unique_{i}"), ctx)

    resp = _make_response("dup")
    await guard.post_call(req, resp, ctx)
    await guard.post_call(req, resp, ctx)
    with pytest.raises(BrixLoopError):
        await guard.post_call(req, resp, ctx)


async def test_window_bounded() -> None:
    """Responses outside the loop_window should not contribute to the count."""
    guard = _make_guard(exact_threshold=3, on_loop="raise", loop_window=4)
    ctx = _make_context()
    req = _make_request()

    # Two duplicates
    await guard.post_call(req, _make_response("dup"), ctx)
    await guard.post_call(req, _make_response("dup"), ctx)
    # Five uniques — push both "dup" hashes out of the window
    for i in range(5):
        await guard.post_call(req, _make_response(f"unique_{i}"), ctx)
    # Now two more duplicates — old "dup" hashes are gone; count only = 2 < 3
    await guard.post_call(req, _make_response("dup"), ctx)
    # Should NOT raise (only 2 "dup" in window)
    await guard.post_call(req, _make_response("dup"), ctx)


async def test_zero_overhead_clean_calls() -> None:
    """All-different responses should never raise or set the inject flag."""
    guard = _make_guard(exact_threshold=3, on_loop="raise")
    ctx = _make_context()
    req = _make_request()

    for i in range(20):
        await guard.post_call(req, _make_response(f"response_{i}"), ctx)

    assert not ctx.metadata.get("_loop_inject_next", False)


async def test_loop_error_contains_history() -> None:
    """BrixLoopError reason should include hash/window info."""
    guard = _make_guard(exact_threshold=2, on_loop="raise")
    ctx = _make_context()
    req = _make_request()
    resp = _make_response("dup")

    await guard.post_call(req, resp, ctx)
    with pytest.raises(BrixLoopError) as exc_info:
        await guard.post_call(req, resp, ctx)

    assert "loop" in str(exc_info.value).lower()
    assert exc_info.value.reason is not None


# ---------------------------------------------------------------------------
# on_loop="raise" — no diversity injection
# ---------------------------------------------------------------------------


async def test_on_loop_raise_skips_diversity() -> None:
    guard = _make_guard(exact_threshold=2, on_loop="raise")
    ctx = _make_context()
    req = _make_request()
    resp = _make_response("dup")

    await guard.post_call(req, resp, ctx)
    with pytest.raises(BrixLoopError):
        await guard.post_call(req, resp, ctx)

    # No diversity injection flag should have been set
    assert not ctx.metadata.get("_loop_inject_next", False)


# ---------------------------------------------------------------------------
# Diversity injection / recovery
# ---------------------------------------------------------------------------


async def test_diversity_injection_in_pre_call() -> None:
    guard = _make_guard(exact_threshold=2, on_loop="inject_diversity", diversity_attempts=2)
    ctx = _make_context()
    req = _make_request()
    resp = _make_response("dup")

    await guard.post_call(req, resp, ctx)
    await guard.post_call(req, resp, ctx)  # triggers first injection flag

    assert ctx.metadata.get("_loop_inject_next") is True

    # pre_call should inject the diversity prompt and clear the flag
    modified_req = await guard.pre_call(req, ctx)
    assert any(
        "[BRIX-LOOP-RECOVERY]" in str(m.get("content", "")) for m in modified_req.messages
    )
    assert not ctx.metadata.get("_loop_inject_next", False)


async def test_raise_after_diversity_exhausted() -> None:
    guard = _make_guard(exact_threshold=2, on_loop="inject_diversity", diversity_attempts=1)
    ctx = _make_context()
    req = _make_request()
    resp = _make_response("dup")

    # First detection: inject diversity (count=0 < 1)
    await guard.post_call(req, resp, ctx)
    await guard.post_call(req, resp, ctx)
    assert ctx.metadata.get("_loop_diversity_count") == 1

    # Second detection: exhausted (count=1 >= 1) → raise
    with pytest.raises(BrixLoopError) as exc_info:
        await guard.post_call(req, resp, ctx)

    assert "diversity" in str(exc_info.value).lower() or "loop" in str(exc_info.value).lower()


async def test_diversity_count_in_metadata() -> None:
    guard = _make_guard(exact_threshold=2, on_loop="inject_diversity", diversity_attempts=3)
    ctx = _make_context()
    req = _make_request()
    resp = _make_response("dup")

    await guard.post_call(req, resp, ctx)
    await guard.post_call(req, resp, ctx)  # count → 1
    assert ctx.metadata.get("_loop_diversity_count") == 1

    # Clear inject flag so post_call runs again without pre_call short-circuit
    ctx.metadata["_loop_inject_next"] = False
    await guard.post_call(req, resp, ctx)  # count → 2
    assert ctx.metadata.get("_loop_diversity_count") == 2


async def test_metadata_keys_namespaced() -> None:
    """All metadata keys written by LoopGuard must start with '_loop_'."""
    guard = _make_guard(exact_threshold=3, on_loop="inject_diversity")
    ctx = _make_context()
    req = _make_request()

    for i in range(3):
        await guard.post_call(req, _make_response(f"r{i}"), ctx)

    loop_keys = [k for k in ctx.metadata if k.startswith("_loop_")]
    all_guard_keys = [k for k in ctx.metadata if k.startswith("_")]
    # Every key starting with "_" that we wrote should be a _loop_ key
    for key in all_guard_keys:
        assert key.startswith("_loop_"), f"unexpected metadata key: {key!r}"
    assert len(loop_keys) >= 1


# ---------------------------------------------------------------------------
# Custom diversity prompt
# ---------------------------------------------------------------------------


async def test_loop_guard_accepts_custom_diversity_prompt() -> None:
    custom_prompt = "CUSTOM DIVERSITY INSTRUCTION"
    guard = _make_guard(
        exact_threshold=2,
        on_loop="inject_diversity",
        diversity_attempts=2,
        diversity_prompt=custom_prompt,
    )
    ctx = _make_context()
    req = _make_request()
    resp = _make_response("dup")

    await guard.post_call(req, resp, ctx)
    await guard.post_call(req, resp, ctx)  # triggers injection

    modified_req = await guard.pre_call(req, ctx)
    assert any(
        custom_prompt in str(m.get("content", "")) for m in modified_req.messages
    )
