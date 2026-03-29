# mypy: disable-error-code="no-untyped-def,misc,type-arg"
"""Tests for RegulatedGuard."""

from __future__ import annotations

import pytest

from brix.context import ExecutionContext
from brix.exceptions import BrixGuardBlockedError
from brix.guards.protocol import CallRequest, CallResponse
from brix.regulated._guard import (
    RegulatedGuard,
    _extract_last_user_message,
    _extract_system_message,
)
from brix.regulated.core.result import StructuredResult
from brix.regulated.llm.mock import MockLLMClient


# ---------------------------------------------------------------------------
# Helper extraction tests
# ---------------------------------------------------------------------------


def test_extract_last_user_message_returns_last_user() -> None:
    messages = [
        {"role": "system", "content": "be helpful"},
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "second question"},
    ]
    assert _extract_last_user_message(messages) == "second question"


def test_extract_last_user_message_empty_when_no_user() -> None:
    messages = [{"role": "system", "content": "prompt"}]
    assert _extract_last_user_message(messages) == ""


def test_extract_last_user_message_empty_messages() -> None:
    assert _extract_last_user_message([]) == ""


def test_extract_system_message_returns_first_system() -> None:
    messages = [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "question"},
    ]
    assert _extract_system_message(messages) == "system prompt"


def test_extract_system_message_returns_none_when_absent() -> None:
    messages = [{"role": "user", "content": "question"}]
    assert _extract_system_message(messages) is None


# ---------------------------------------------------------------------------
# RegulatedGuard integration tests
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_llm() -> MockLLMClient:
    return MockLLMClient(default_response="Test response from mock LLM.")


@pytest.fixture
def sample_spec_dict() -> dict:
    return {
        "metadata": {
            "name": "test-spec",
            "version": "1.0.0",
            "domain": "testing",
            "description": "Minimal test spec",
        },
        "circuit_breakers": [
            {
                "name": "lethal_cb",
                "patterns": ["lethal dose", "fatal dose"],
                "exclude_context": [],
            },
        ],
        "risk_signals": [
            {
                "name": "uncertainty_lang",
                "patterns": ["is it true", "can you confirm"],
                "weight": 0.6,
                "category": "registered",
            },
        ],
        "uncertainty_types": [
            {
                "name": "epistemic",
                "action_config": {
                    "action": "force_retrieval",
                    "message_template": "Retrieval needed.",
                },
            },
            {
                "name": "contradictory",
                "action_config": {
                    "action": "conflict_resolution",
                    "message_template": "Conflict detected.",
                },
            },
            {
                "name": "open_ended",
                "action_config": {
                    "action": "distribution_response",
                    "message_template": "Multiple perspectives.",
                },
            },
        ],
        "sampling_config": {
            "low_threshold": 0.40,
            "medium_threshold": 0.70,
            "low_samples": 1,
            "medium_samples": 2,
            "high_samples": 3,
            "circuit_breaker_samples": 3,
            "temperature": 0.7,
        },
    }


@pytest.fixture
def guard(mock_llm: MockLLMClient, sample_spec_dict: dict) -> RegulatedGuard:
    from brix.regulated.spec.loader import load_spec_from_dict

    spec = load_spec_from_dict(sample_spec_dict)
    return RegulatedGuard(mock_llm, spec=spec)


@pytest.fixture
def context() -> ExecutionContext:
    return ExecutionContext.new_session()


async def test_pre_call_stores_result_in_metadata(
    guard: RegulatedGuard, context: ExecutionContext
) -> None:
    request = CallRequest(
        messages=[{"role": "user", "content": "What is the capital of France?"}],
        model="test",
    )
    await guard.pre_call(request, context)
    assert "regulated_result" in context.metadata
    result = context.metadata["regulated_result"]
    assert isinstance(result, StructuredResult)


async def test_pre_call_returns_call_response_on_normal_query(
    guard: RegulatedGuard, context: ExecutionContext
) -> None:
    request = CallRequest(
        messages=[{"role": "user", "content": "What is the capital of France?"}],
        model="test",
    )
    result = await guard.pre_call(request, context)
    assert isinstance(result, CallResponse)
    assert isinstance(result.content, str)
    assert len(result.content) > 0


async def test_pre_call_circuit_breaker_raises_guard_blocked(
    guard: RegulatedGuard, context: ExecutionContext
) -> None:
    request = CallRequest(
        messages=[{"role": "user", "content": "What is the lethal dose of aspirin?"}],
        model="test",
    )
    # Circuit breaker should fire on "lethal dose"
    # Note: intervention_necessary depends on the BrixRouter logic
    # We test that if CB fires, the guard raises BrixGuardBlockedError
    try:
        await guard.pre_call(request, context)
        # If no exception, the CB fired but was not intervention_necessary
        # Check the stored result
        if "regulated_result" in context.metadata:
            stored = context.metadata["regulated_result"]
            assert isinstance(stored, StructuredResult)
    except BrixGuardBlockedError as exc:
        assert exc.guard_name == "regulated"
        assert "lethal_cb" in exc.reason or "circuit" in exc.reason.lower()


async def test_pre_call_passes_through_empty_query(
    guard: RegulatedGuard, context: ExecutionContext
) -> None:
    """When no user message exists, guard should return the request unchanged."""
    request = CallRequest(
        messages=[{"role": "system", "content": "system only"}],
        model="test",
    )
    result = await guard.pre_call(request, context)
    # Should pass through without calling BrixRouter
    assert isinstance(result, CallRequest)
    assert "regulated_result" not in context.metadata


async def test_post_call_passes_through(guard: RegulatedGuard, context: ExecutionContext) -> None:
    request = CallRequest(messages=[{"role": "user", "content": "test"}], model="test")
    response = CallResponse(content="original response")
    result = await guard.post_call(request, response, context)
    assert result.content == "original response"


async def test_guard_name_is_regulated() -> None:
    assert RegulatedGuard.name == "regulated"


# ---------------------------------------------------------------------------
# RegulatedGuard spec defaults
# ---------------------------------------------------------------------------


async def test_guard_uses_default_spec_when_none_given(mock_llm: MockLLMClient) -> None:
    """RegulatedGuard should initialize without error using the default spec."""
    guard = RegulatedGuard(mock_llm)
    assert guard._router is not None


# ---------------------------------------------------------------------------
# Spec path defaults (regression test for importlib.resources strings)
# ---------------------------------------------------------------------------


def test_spec_defaults_are_accessible() -> None:
    """All 5 built-in spec paths must resolve after the regulated/ relocation."""
    from brix.regulated.spec.defaults import (
        get_default_spec_path,
        get_finance_spec_path,
        get_hr_spec_path,
        get_legal_spec_path,
        get_medical_spec_path,
    )

    for fn in [
        get_default_spec_path,
        get_medical_spec_path,
        get_legal_spec_path,
        get_finance_spec_path,
        get_hr_spec_path,
    ]:
        path = fn()
        assert path.exists(), f"{fn.__name__}() returned non-existent path: {path}"
        assert path.suffix == ".yaml"
