# mypy: disable-error-code="no-untyped-def,misc,type-arg"
"""Tests for SchemaGuard."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel, Field

from brix.context import ExecutionContext
from brix.exceptions import BrixSchemaError
from brix.guards.protocol import CallRequest, CallResponse
from brix.guards.schema import SchemaGuard, _extract_json


# ---------------------------------------------------------------------------
# Test schema
# ---------------------------------------------------------------------------


class SimpleSchema(BaseModel):
    answer: str
    confidence: float


class RequiredFieldSchema(BaseModel):
    name: str
    score: int = Field(ge=0, le=100)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_context() -> ExecutionContext:
    return ExecutionContext.new_session()


def _make_request(messages=None) -> CallRequest:
    if messages is None:
        messages = [{"role": "user", "content": "Test"}]
    return CallRequest(messages=messages, model="gpt-4o-mini")


def _make_response(content: str) -> CallResponse:
    return CallResponse(content=content, usage={"prompt_tokens": 10, "completion_tokens": 20})


def _make_llm_callable(responses: list) -> AsyncMock:
    return AsyncMock(
        side_effect=[CallResponse(content=r) if isinstance(r, str) else r for r in responses]
    )


# ---------------------------------------------------------------------------
# Name
# ---------------------------------------------------------------------------


def test_name():
    guard = SchemaGuard(AsyncMock(), SimpleSchema)
    assert guard.name == "schema"


# ---------------------------------------------------------------------------
# _extract_json helper
# ---------------------------------------------------------------------------


def test_extract_json_plain():
    assert _extract_json('{"answer": "yes"}') == '{"answer": "yes"}'


def test_extract_json_json_block():
    text = '```json\n{"answer": "yes"}\n```'
    assert '"answer"' in _extract_json(text)


def test_extract_json_generic_block():
    text = '```\n{"answer": "yes"}\n```'
    assert '"answer"' in _extract_json(text)


def test_extract_json_leading_trailing_text():
    text = 'Here is the result: {"answer": "yes", "confidence": 0.9} Hope that helps!'
    extracted = _extract_json(text)
    data = json.loads(extracted)
    assert data["answer"] == "yes"


# ---------------------------------------------------------------------------
# pre_call: schema injection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_schema_marker_injected_in_system_prompt():
    guard = SchemaGuard(AsyncMock(), SimpleSchema)
    ctx = _make_context()
    req = _make_request()
    result = await guard.pre_call(req, ctx)
    system_msgs = [m for m in result.messages if m.get("role") == "system"]
    assert len(system_msgs) == 1
    assert SchemaGuard._MARKER in system_msgs[0]["content"]


@pytest.mark.asyncio
async def test_schema_injection_creates_system_message_when_absent():
    guard = SchemaGuard(AsyncMock(), SimpleSchema)
    ctx = _make_context()
    req = _make_request([{"role": "user", "content": "Test"}])
    result = await guard.pre_call(req, ctx)
    system_msgs = [m for m in result.messages if m.get("role") == "system"]
    assert len(system_msgs) == 1


@pytest.mark.asyncio
async def test_schema_injection_appends_to_existing_system_message():
    guard = SchemaGuard(AsyncMock(), SimpleSchema)
    ctx = _make_context()
    req = _make_request(
        [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Test"},
        ]
    )
    result = await guard.pre_call(req, ctx)
    system_msgs = [m for m in result.messages if m.get("role") == "system"]
    assert len(system_msgs) == 1
    assert "You are helpful." in system_msgs[0]["content"]
    assert SchemaGuard._MARKER in system_msgs[0]["content"]


@pytest.mark.asyncio
async def test_schema_injection_idempotent():
    guard = SchemaGuard(AsyncMock(), SimpleSchema)
    ctx = _make_context()
    req = _make_request()
    result1 = await guard.pre_call(req, ctx)
    result2 = await guard.pre_call(result1, ctx)
    # Marker should appear exactly once
    system_content = next(m["content"] for m in result2.messages if m.get("role") == "system")
    assert system_content.count(SchemaGuard._MARKER) == 1


@pytest.mark.asyncio
async def test_inject_schema_false_no_modification():
    guard = SchemaGuard(AsyncMock(), SimpleSchema, inject_schema=False)
    ctx = _make_context()
    req = _make_request()
    result = await guard.pre_call(req, ctx)
    assert result is req


# ---------------------------------------------------------------------------
# post_call: validation success
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_valid_json_returns_pydantic_instance():
    guard = SchemaGuard(AsyncMock(), SimpleSchema)
    ctx = _make_context()
    req = _make_request()
    resp = _make_response('{"answer": "yes", "confidence": 0.95}')
    result = await guard.post_call(req, resp, ctx)
    assert isinstance(result.content, SimpleSchema)
    assert result.content.answer == "yes"
    assert result.content.confidence == pytest.approx(0.95)


@pytest.mark.asyncio
async def test_valid_json_in_markdown_block():
    guard = SchemaGuard(AsyncMock(), SimpleSchema)
    ctx = _make_context()
    req = _make_request()
    resp = _make_response('```json\n{"answer": "hello", "confidence": 0.5}\n```')
    result = await guard.post_call(req, resp, ctx)
    assert isinstance(result.content, SimpleSchema)


@pytest.mark.asyncio
async def test_schema_validated_true_in_metadata():
    guard = SchemaGuard(AsyncMock(), SimpleSchema)
    ctx = _make_context()
    resp = _make_response('{"answer": "yes", "confidence": 0.9}')
    await guard.post_call(_make_request(), resp, ctx)
    assert ctx.metadata["schema_validated"] is True
    assert ctx.metadata["schema_attempts"] == 1


@pytest.mark.asyncio
async def test_usage_preserved_after_validation():
    guard = SchemaGuard(AsyncMock(), SimpleSchema)
    ctx = _make_context()
    resp = _make_response('{"answer": "yes", "confidence": 0.9}')
    result = await guard.post_call(_make_request(), resp, ctx)
    assert result.usage == {"prompt_tokens": 10, "completion_tokens": 20}


# ---------------------------------------------------------------------------
# post_call: self-healing re-prompts
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_validation_error_triggers_reprompt():
    valid_json = '{"answer": "fixed", "confidence": 0.8}'
    llm_callable = _make_llm_callable([valid_json])
    guard = SchemaGuard(llm_callable, SimpleSchema, max_retries=1)
    ctx = _make_context()
    resp = _make_response("not json at all")
    result = await guard.post_call(_make_request(), resp, ctx)
    assert isinstance(result.content, SimpleSchema)
    assert llm_callable.call_count == 1  # one re-prompt


@pytest.mark.asyncio
async def test_all_retries_fail_raises_schema_error():
    llm_callable = _make_llm_callable(["bad", "also bad", "still bad"])
    guard = SchemaGuard(llm_callable, SimpleSchema, max_retries=2)
    ctx = _make_context()
    resp = _make_response("not valid")
    with pytest.raises(BrixSchemaError) as exc_info:
        await guard.post_call(_make_request(), resp, ctx)
    assert "attempt" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_schema_error_contains_history():
    llm_callable = _make_llm_callable(["bad", "also bad"])
    guard = SchemaGuard(llm_callable, SimpleSchema, max_retries=1)
    ctx = _make_context()
    resp = _make_response("not valid")
    with pytest.raises(BrixSchemaError) as exc_info:
        await guard.post_call(_make_request(), resp, ctx)
    assert "history" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# post_call: healing time limit
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_max_healing_seconds_exceeded_raises():
    async def slow_callable(req):
        await asyncio.sleep(0.2)
        return CallResponse(content="not valid")

    guard = SchemaGuard(slow_callable, SimpleSchema, max_retries=5, max_healing_seconds=0.01)
    ctx = _make_context()
    resp = _make_response("not valid")
    with pytest.raises(BrixSchemaError) as exc_info:
        await guard.post_call(_make_request(), resp, ctx)
    assert "time limit" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_max_healing_seconds_none_uses_only_retry_count():
    valid = '{"answer": "ok", "confidence": 0.7}'
    llm_callable = _make_llm_callable(["bad", valid])
    guard = SchemaGuard(llm_callable, SimpleSchema, max_retries=2, max_healing_seconds=None)
    ctx = _make_context()
    resp = _make_response("bad")
    result = await guard.post_call(_make_request(), resp, ctx)
    assert isinstance(result.content, SimpleSchema)
