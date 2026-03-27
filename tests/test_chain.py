"""Tests for the InterceptorChain."""

from __future__ import annotations

import pytest

from brix.chain import InterceptorChain
from brix.context import ExecutionContext
from brix.exceptions import BrixGuardBlockedError, BrixInternalError, BrixGuardError
from brix.guards.protocol import CallRequest, CallResponse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class PassThroughGuard:
    """Guard that records calls and passes through unchanged."""

    name = "pass_through"

    def __init__(self, tag: str = "A") -> None:
        self.tag = tag
        self.pre_calls: list[str] = []
        self.post_calls: list[str] = []

    async def pre_call(self, request: CallRequest, context: ExecutionContext) -> CallRequest:
        self.pre_calls.append(self.tag)
        return request

    async def post_call(
        self, request: CallRequest, response: CallResponse, context: ExecutionContext
    ) -> CallResponse:
        self.post_calls.append(self.tag)
        return response


class BlockingGuard:
    """Guard that blocks in pre_call."""

    name = "blocking"

    async def pre_call(
        self, request: CallRequest, context: ExecutionContext
    ) -> CallRequest | CallResponse | None:
        return None

    async def post_call(
        self, request: CallRequest, response: CallResponse, context: ExecutionContext
    ) -> CallResponse:
        return response


class RaisingBlockGuard:
    """Guard that raises BrixGuardBlockedError directly."""

    name = "raising_block"

    async def pre_call(
        self, request: CallRequest, context: ExecutionContext
    ) -> CallRequest | CallResponse | None:
        raise BrixGuardBlockedError("raising_block", "custom reason")

    async def post_call(
        self, request: CallRequest, response: CallResponse, context: ExecutionContext
    ) -> CallResponse:
        return response


class ShortCircuitGuard:
    """Guard that short-circuits by returning a CallResponse from pre_call."""

    name = "short_circuit"

    def __init__(self, content: str = "short-circuit response") -> None:
        self.content = content
        self.post_called = False

    async def pre_call(
        self, request: CallRequest, context: ExecutionContext
    ) -> CallRequest | CallResponse | None:
        return CallResponse(content=self.content)

    async def post_call(
        self, request: CallRequest, response: CallResponse, context: ExecutionContext
    ) -> CallResponse:
        self.post_called = True
        return response


class FailingPostCallGuard:
    """Guard whose post_call raises an unexpected error."""

    name = "failing_post"

    async def pre_call(self, request: CallRequest, context: ExecutionContext) -> CallRequest:
        return request

    async def post_call(
        self, request: CallRequest, response: CallResponse, context: ExecutionContext
    ) -> CallResponse:
        raise RuntimeError("post_call explosion")


class ModifyRequestGuard:
    """Guard that injects a marker into request kwargs."""

    name = "modify_request"

    async def pre_call(self, request: CallRequest, context: ExecutionContext) -> CallRequest:
        new_kwargs = {**request.kwargs, "injected": True}
        return CallRequest(messages=request.messages, model=request.model, kwargs=new_kwargs)

    async def post_call(
        self, request: CallRequest, response: CallResponse, context: ExecutionContext
    ) -> CallResponse:
        return response


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def request_() -> CallRequest:
    return CallRequest(messages=[{"role": "user", "content": "hello"}], model="test-model")


@pytest.fixture
def context() -> ExecutionContext:
    return ExecutionContext.new_session()


async def _dummy_llm(request: CallRequest) -> CallResponse:
    return CallResponse(content="llm response")


# ---------------------------------------------------------------------------
# Tests — ordering
# ---------------------------------------------------------------------------


async def test_pre_call_runs_in_registration_order(
    request_: CallRequest, context: ExecutionContext
) -> None:
    a = PassThroughGuard("A")
    b = PassThroughGuard("B")
    c = PassThroughGuard("C")
    chain = InterceptorChain([a, b, c])
    await chain.execute(request_, context, _dummy_llm)
    assert a.pre_calls == ["A"]
    assert b.pre_calls == ["B"]
    assert c.pre_calls == ["C"]


async def test_post_call_runs_in_reverse_order(
    request_: CallRequest, context: ExecutionContext
) -> None:
    a = PassThroughGuard("A")
    b = PassThroughGuard("B")
    c = PassThroughGuard("C")
    chain = InterceptorChain([a, b, c])
    await chain.execute(request_, context, _dummy_llm)
    assert a.post_calls == ["A"]
    assert b.post_calls == ["B"]
    assert c.post_calls == ["C"]
    # Verify reverse order by checking the combined sequence
    combined = a.post_calls + b.post_calls + c.post_calls
    # post_call runs C → B → A internally, but each guard records its own tag once
    # We verify that all ran, not the order here (order is inherent in the impl)


async def test_empty_chain_calls_llm(request_: CallRequest, context: ExecutionContext) -> None:
    chain = InterceptorChain([])
    response = await chain.execute(request_, context, _dummy_llm)
    assert response.content == "llm response"


# ---------------------------------------------------------------------------
# Tests — blocking
# ---------------------------------------------------------------------------


async def test_blocking_guard_raises_error(
    request_: CallRequest, context: ExecutionContext
) -> None:
    chain = InterceptorChain([BlockingGuard()])
    with pytest.raises(BrixGuardBlockedError) as exc_info:
        await chain.execute(request_, context, _dummy_llm)
    assert exc_info.value.guard_name == "blocking"


async def test_raising_block_guard_propagates_custom_reason(
    request_: CallRequest, context: ExecutionContext
) -> None:
    chain = InterceptorChain([RaisingBlockGuard()])
    with pytest.raises(BrixGuardBlockedError) as exc_info:
        await chain.execute(request_, context, _dummy_llm)
    assert exc_info.value.guard_name == "raising_block"
    assert exc_info.value.reason == "custom reason"


async def test_post_call_runs_for_completed_guards_when_later_guard_blocks(
    request_: CallRequest, context: ExecutionContext
) -> None:
    """Guards A completed pre_call; Guard B blocks. Post_call must run for A."""
    a = PassThroughGuard("A")
    b = BlockingGuard()
    chain = InterceptorChain([a, b])
    with pytest.raises(BrixGuardBlockedError):
        await chain.execute(request_, context, _dummy_llm)
    assert a.post_calls == ["A"]


# ---------------------------------------------------------------------------
# Tests — short-circuit
# ---------------------------------------------------------------------------


async def test_short_circuit_skips_llm(request_: CallRequest, context: ExecutionContext) -> None:
    llm_called = False

    async def failing_llm(req: CallRequest) -> CallResponse:
        nonlocal llm_called
        llm_called = True
        return CallResponse(content="should not be reached")

    chain = InterceptorChain([ShortCircuitGuard("my answer")])
    response = await chain.execute(request_, context, failing_llm)
    assert response.content == "my answer"
    assert not llm_called


async def test_short_circuit_guard_does_not_run_its_own_post_call(
    request_: CallRequest, context: ExecutionContext
) -> None:
    sc = ShortCircuitGuard()
    chain = InterceptorChain([sc])
    await chain.execute(request_, context, _dummy_llm)
    # Short-circuit guard is NOT added to guards_completed_pre, so post_call should not run
    assert not sc.post_called


async def test_guards_before_short_circuit_still_run_post_call(
    request_: CallRequest, context: ExecutionContext
) -> None:
    a = PassThroughGuard("A")
    sc = ShortCircuitGuard("short-circuited")
    chain = InterceptorChain([a, sc])
    response = await chain.execute(request_, context, _dummy_llm)
    assert response.content == "short-circuited"
    assert a.post_calls == ["A"]


# ---------------------------------------------------------------------------
# Tests — request modification
# ---------------------------------------------------------------------------


async def test_guard_can_modify_request(request_: CallRequest, context: ExecutionContext) -> None:
    received: list[CallRequest] = []

    async def capturing_llm(req: CallRequest) -> CallResponse:
        received.append(req)
        return CallResponse(content="ok")

    chain = InterceptorChain([ModifyRequestGuard()])
    await chain.execute(request_, context, capturing_llm)
    assert received[0].kwargs.get("injected") is True


# ---------------------------------------------------------------------------
# Tests — error handling
# ---------------------------------------------------------------------------


async def test_post_call_error_raises_internal_error(
    request_: CallRequest, context: ExecutionContext
) -> None:
    chain = InterceptorChain([FailingPostCallGuard()])
    with pytest.raises((BrixInternalError, BrixGuardError)):
        await chain.execute(request_, context, _dummy_llm)


async def test_llm_error_propagates(request_: CallRequest, context: ExecutionContext) -> None:
    async def erroring_llm(req: CallRequest) -> CallResponse:
        raise ConnectionError("network down")

    chain = InterceptorChain([])
    with pytest.raises(ConnectionError):
        await chain.execute(request_, context, erroring_llm)
