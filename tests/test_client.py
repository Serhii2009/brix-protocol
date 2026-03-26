"""Tests for BrixClient and BRIX.wrap()."""

from __future__ import annotations

import warnings

import pytest

from brix.client import BRIX, BrixClient
from brix.context import ExecutionContext
from brix.exceptions import BrixConfigurationError
from brix.guards.protocol import CallRequest, CallResponse


# ---------------------------------------------------------------------------
# Minimal mock LLM client (legacy LLMClient protocol)
# ---------------------------------------------------------------------------


class SimpleMockLLM:
    """Implements the BRIX LLMClient protocol for testing."""

    def __init__(self, response: str = "mock response") -> None:
        self._response = response
        self.calls: list[str] = []

    async def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> str:
        self.calls.append(prompt)
        return self._response


# ---------------------------------------------------------------------------
# BRIX.wrap() tests
# ---------------------------------------------------------------------------


def test_wrap_returns_brix_client() -> None:
    llm = SimpleMockLLM()
    client = BRIX.wrap(llm)
    assert isinstance(client, BrixClient)


def test_wrap_with_no_guards_creates_empty_chain() -> None:
    llm = SimpleMockLLM()
    client = BRIX.wrap(llm)
    assert client.chain.guards == []


def test_wrap_warns_for_unimplemented_max_cost_usd() -> None:
    llm = SimpleMockLLM()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        BRIX.wrap(llm, max_cost_usd=10.0)
    assert any("max_cost_usd" in str(warning.message) for warning in w)


def test_wrap_warns_for_unimplemented_max_time_seconds() -> None:
    llm = SimpleMockLLM()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        BRIX.wrap(llm, max_time_seconds=30.0)
    assert any("max_time_seconds" in str(warning.message) for warning in w)


def test_wrap_does_not_warn_when_params_are_default() -> None:
    llm = SimpleMockLLM()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        BRIX.wrap(llm)
    brix_warnings = [x for x in w if "BRIX" in str(x.message)]
    assert len(brix_warnings) == 0


def test_wrap_raises_for_unrecognized_llm() -> None:
    class UnknownClient:
        pass

    with pytest.raises(BrixConfigurationError):
        BRIX.wrap(UnknownClient())


# ---------------------------------------------------------------------------
# BrixClient.complete() tests
# ---------------------------------------------------------------------------


async def test_complete_returns_string() -> None:
    llm = SimpleMockLLM("hello world")
    client = BRIX.wrap(llm)
    result = await client.complete([{"role": "user", "content": "hi"}])
    assert result == "hello world"


async def test_complete_increments_call_count() -> None:
    llm = SimpleMockLLM()
    client = BRIX.wrap(llm)
    assert client.context.call_count == 0
    await client.complete([{"role": "user", "content": "first"}])
    assert client.context.call_count == 1
    await client.complete([{"role": "user", "content": "second"}])
    assert client.context.call_count == 2


async def test_complete_appends_call_history() -> None:
    llm = SimpleMockLLM()
    client = BRIX.wrap(llm)
    await client.complete([{"role": "user", "content": "test"}])
    assert len(client.context.call_history) == 1
    record = client.context.call_history[0]
    assert record.response.content == "mock response"
    assert record.latency_ms >= 0


async def test_session_id_is_stable_across_calls() -> None:
    llm = SimpleMockLLM()
    client = BRIX.wrap(llm)
    await client.complete([{"role": "user", "content": "a"}])
    session_id = client.context.session_id
    await client.complete([{"role": "user", "content": "b"}])
    assert client.context.session_id == session_id


async def test_run_id_changes_per_call() -> None:
    llm = SimpleMockLLM()
    client = BRIX.wrap(llm)
    await client.complete([{"role": "user", "content": "a"}])
    run_id_1 = client.context.run_id
    await client.complete([{"role": "user", "content": "b"}])
    run_id_2 = client.context.run_id
    assert run_id_1 != run_id_2


async def test_acomplete_is_alias_for_complete() -> None:
    llm = SimpleMockLLM("alias works")
    client = BRIX.wrap(llm)
    result = await client.acomplete([{"role": "user", "content": "test"}])
    assert result == "alias works"


# ---------------------------------------------------------------------------
# Context tests
# ---------------------------------------------------------------------------


def test_context_session_start_is_set() -> None:
    llm = SimpleMockLLM()
    client = BRIX.wrap(llm)
    assert client.context.session_start is not None


def test_context_is_execution_context_instance() -> None:
    llm = SimpleMockLLM()
    client = BRIX.wrap(llm)
    assert isinstance(client.context, ExecutionContext)


# ---------------------------------------------------------------------------
# Guard integration tests
# ---------------------------------------------------------------------------


async def test_guard_receives_context() -> None:
    """Guard should receive the same context the client holds."""
    received_contexts: list[ExecutionContext] = []

    class ContextCapturingGuard:
        name = "ctx_capture"

        async def pre_call(
            self, request: CallRequest, context: ExecutionContext
        ) -> CallRequest:
            received_contexts.append(context)
            return request

        async def post_call(
            self, request: CallRequest, response: CallResponse, context: ExecutionContext
        ) -> CallResponse:
            return response

    llm = SimpleMockLLM()
    client = BrixClient(llm, guards=[ContextCapturingGuard()])
    await client.complete([{"role": "user", "content": "test"}])

    assert len(received_contexts) == 1
    assert received_contexts[0] is client.context
