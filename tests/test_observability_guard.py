"""Tests for ObservabilityGuard."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from brix.context import ExecutionContext
from brix.exceptions import BrixGuardError
from brix.guards.observability import ObservabilityGuard
from brix.guards.protocol import CallRequest, CallResponse


def _make_context(session_id: str = "test-session") -> ExecutionContext:
    ctx = ExecutionContext.new_session()
    ctx.session_id = session_id
    ctx.run_id = "run-001"
    ctx.call_count = 1
    return ctx


def _make_request() -> CallRequest:
    return CallRequest(
        messages=[{"role": "user", "content": "Hi"}],
        model="gpt-4o-mini",
    )


def _make_response(content="Hello") -> CallResponse:
    return CallResponse(
        content=content,
        usage={"prompt_tokens": 10, "completion_tokens": 5},
    )


# ---------------------------------------------------------------------------
# Name
# ---------------------------------------------------------------------------


def test_name():
    guard = ObservabilityGuard(log_path=None, guard_names=["budget"])
    assert guard.name == "observability"


# ---------------------------------------------------------------------------
# In-memory buffer
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_buffer_populated_without_log_path():
    guard = ObservabilityGuard(log_path=None, guard_names=["observability"])
    ctx = _make_context()
    req = _make_request()
    resp = _make_response()
    await guard.pre_call(req, ctx)
    await guard.post_call(req, resp, ctx)
    traces = guard.get_traces()
    assert len(traces) == 1


@pytest.mark.asyncio
async def test_get_traces_most_recent_first():
    guard = ObservabilityGuard(log_path=None, guard_names=[], buffer_size=10)
    ctx = _make_context()
    req = _make_request()

    for i in range(3):
        ctx.run_id = f"run-{i}"
        ctx.call_count = i + 1
        await guard.pre_call(req, ctx)
        await guard.post_call(req, _make_response(f"response {i}"), ctx)

    traces = guard.get_traces()
    assert traces[0]["sequence"] == 3  # most recent
    assert traces[-1]["sequence"] == 1  # oldest


@pytest.mark.asyncio
async def test_buffer_size_respected():
    guard = ObservabilityGuard(log_path=None, guard_names=[], buffer_size=3)
    ctx = _make_context()
    req = _make_request()

    for i in range(5):
        ctx.run_id = f"run-{i}"
        ctx.call_count = i + 1
        await guard.pre_call(req, ctx)
        await guard.post_call(req, _make_response(), ctx)

    traces = guard.get_traces()
    assert len(traces) == 3  # oldest 2 evicted


# ---------------------------------------------------------------------------
# Audit entry format
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_audit_entry_written_to_file(tmp_path: Path):
    guard = ObservabilityGuard(log_path=tmp_path, guard_names=["budget", "observability"])
    ctx = _make_context()
    req = _make_request()
    resp = _make_response()
    await guard.pre_call(req, ctx)
    await guard.post_call(req, resp, ctx)

    audit_file = tmp_path / "brix_audit.jsonl"
    assert audit_file.exists()
    entry = json.loads(audit_file.read_text().strip())
    assert entry["run_id"] == ctx.run_id
    assert entry["session_id"] == ctx.session_id
    assert entry["model"] == req.model
    assert "chain_hash" in entry
    assert "prompt_hash" in entry
    assert "response_hash" in entry
    assert entry["latency_ms"] >= 0.0


@pytest.mark.asyncio
async def test_genesis_chain_hash(tmp_path: Path):
    guard = ObservabilityGuard(log_path=tmp_path, guard_names=[])
    session_id = "genesis-test"
    ctx = _make_context(session_id)
    req = _make_request()
    resp = _make_response()
    await guard.pre_call(req, ctx)
    await guard.post_call(req, resp, ctx)

    entry = json.loads((tmp_path / "brix_audit.jsonl").read_text().strip())
    expected_genesis = json.dumps({"genesis": session_id})
    expected_hash = hashlib.sha256(expected_genesis.encode()).hexdigest()
    assert entry["chain_hash"] == expected_hash


@pytest.mark.asyncio
async def test_chain_hash_second_entry(tmp_path: Path):
    guard = ObservabilityGuard(log_path=tmp_path, guard_names=[])
    ctx = _make_context()
    req = _make_request()

    # First call
    ctx.run_id = "run-1"
    ctx.call_count = 1
    await guard.pre_call(req, ctx)
    await guard.post_call(req, _make_response(), ctx)

    # Capture first entry JSON
    first_entry_json_raw = (tmp_path / "brix_audit.jsonl").read_text().splitlines()[0]
    first_entry = json.loads(first_entry_json_raw)
    first_entry_canonical = json.dumps(first_entry, sort_keys=True, default=str)

    # Second call
    ctx.run_id = "run-2"
    ctx.call_count = 2
    await guard.pre_call(req, ctx)
    await guard.post_call(req, _make_response(), ctx)

    lines = (tmp_path / "brix_audit.jsonl").read_text().splitlines()
    second_entry = json.loads(lines[1])
    expected_hash = hashlib.sha256(first_entry_canonical.encode()).hexdigest()
    assert second_entry["chain_hash"] == expected_hash


@pytest.mark.asyncio
async def test_prompt_hash_is_sha256_of_messages():
    guard = ObservabilityGuard(log_path=None, guard_names=[])
    ctx = _make_context()
    req = _make_request()
    resp = _make_response()
    await guard.pre_call(req, ctx)
    await guard.post_call(req, resp, ctx)

    entry = guard.get_traces()[0]
    expected = hashlib.sha256(
        json.dumps(req.messages, sort_keys=True, default=str).encode()
    ).hexdigest()
    assert entry["prompt_hash"] == expected


@pytest.mark.asyncio
async def test_latency_ms_is_positive():
    guard = ObservabilityGuard(log_path=None, guard_names=[])
    ctx = _make_context()
    req = _make_request()
    resp = _make_response()
    await guard.pre_call(req, ctx)
    await asyncio.sleep(0.01)
    await guard.post_call(req, resp, ctx)

    entry = guard.get_traces()[0]
    assert entry["latency_ms"] > 0.0


# ---------------------------------------------------------------------------
# Strict mode
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_write_failure_strict_false_does_not_raise(tmp_path: Path, caplog):
    guard = ObservabilityGuard(log_path=tmp_path, guard_names=[], strict_mode=False)
    ctx = _make_context()
    req = _make_request()
    resp = _make_response()
    await guard.pre_call(req, ctx)

    with patch("builtins.open", side_effect=OSError("disk full")), caplog.at_level(logging.WARNING):
        result = await guard.post_call(req, resp, ctx)

    assert result is resp  # pipeline continues


@pytest.mark.asyncio
async def test_write_failure_strict_true_raises(tmp_path: Path):
    guard = ObservabilityGuard(log_path=tmp_path, guard_names=[], strict_mode=True)
    ctx = _make_context()
    req = _make_request()
    resp = _make_response()
    await guard.pre_call(req, ctx)

    with patch("pathlib.Path.open", side_effect=OSError("disk full")):
        with pytest.raises(BrixGuardError):
            await guard.post_call(req, resp, ctx)


# ---------------------------------------------------------------------------
# DRE session file
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dre_session_file_written(tmp_path: Path):
    guard = ObservabilityGuard(log_path=tmp_path, guard_names=[])
    session_id = "dre-test"
    ctx = _make_context(session_id)
    req = _make_request()
    resp = _make_response()
    await guard.pre_call(req, ctx)
    await guard.post_call(req, resp, ctx)

    session_file = tmp_path / ".brix_sessions" / f"{session_id}.jsonl"
    assert session_file.exists()
    record = json.loads(session_file.read_text().strip())
    assert record["run_id"] == ctx.run_id
    assert record["content"] == "Hello"
    assert record["content_type"] == "str"


@pytest.mark.asyncio
async def test_dre_session_rotation(tmp_path: Path):
    guard = ObservabilityGuard(
        log_path=tmp_path, guard_names=[], max_session_records=2
    )
    ctx = _make_context("rotate-test")
    req = _make_request()

    # Write 3 records; rotation should trigger after 2
    for i in range(3):
        ctx.run_id = f"run-{i}"
        ctx.call_count = i + 1
        await guard.pre_call(req, ctx)
        await guard.post_call(req, _make_response(), ctx)

    sessions_dir = tmp_path / ".brix_sessions"
    jsonl_files = list(sessions_dir.glob("*.jsonl"))
    # Should have at least 2 files: one rotated + one current
    assert len(jsonl_files) >= 2


# ---------------------------------------------------------------------------
# Concurrent chain integrity
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrent_chain_integrity(tmp_path: Path):
    """Chain hashes must be consistent under concurrent asyncio.gather."""
    guard = ObservabilityGuard(log_path=tmp_path, guard_names=[])

    async def make_call(i: int) -> None:
        ctx = ExecutionContext.new_session()
        ctx.session_id = "concurrent-session"
        ctx.run_id = f"run-{i}"
        ctx.call_count = i + 1
        req = _make_request()
        resp = _make_response()
        await guard.pre_call(req, ctx)
        await guard.post_call(req, resp, ctx)

    await asyncio.gather(*[make_call(i) for i in range(5)])

    lines = (tmp_path / "brix_audit.jsonl").read_text().splitlines()
    assert len(lines) == 5

    # Verify chain: each entry's chain_hash = SHA-256 of the previous entry's canonical JSON
    prev_json = json.dumps({"genesis": "concurrent-session"})
    entries_sorted = [json.loads(line) for line in lines]
    entries_sorted.sort(key=lambda e: e["sequence"])

    for entry in entries_sorted:
        expected = hashlib.sha256(prev_json.encode()).hexdigest()
        assert entry["chain_hash"] == expected
        prev_json = json.dumps(entry, sort_keys=True, default=str)
