# mypy: disable-error-code="no-untyped-def,misc,type-arg"
"""Tests for BrixReplayClient and BRIX.replay()."""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest
from pydantic import BaseModel

from brix import BRIX
from brix.exceptions import BrixReplayError
from brix.guards.observability import ObservabilityGuard
from brix.guards.protocol import CallResponse
from brix.replay import BrixReplayClient


class OutputModel(BaseModel):
    answer: str
    confidence: float


def _write_dre_records(sessions_dir: Path, session_id: str, records: list[dict]) -> None:
    """Write DRE records directly to disk for testing."""
    sessions_dir.mkdir(parents=True, exist_ok=True)
    path = sessions_dir / f"{session_id}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# BrixReplayClient basic behaviour
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sequential_replay_returns_responses_in_order(tmp_path: Path):
    session_id = "seq-test"
    sessions_dir = tmp_path / ".brix_sessions"
    _write_dre_records(
        sessions_dir,
        session_id,
        [
            {
                "run_id": "r1",
                "sequence": 1,
                "content": "first",
                "content_type": "str",
                "usage": None,
            },
            {
                "run_id": "r2",
                "sequence": 2,
                "content": "second",
                "content_type": "str",
                "usage": None,
            },
            {
                "run_id": "r3",
                "sequence": 3,
                "content": "third",
                "content_type": "str",
                "usage": None,
            },
        ],
    )
    client = BrixReplayClient(session_id=session_id, log_path=tmp_path)
    assert await client.complete() == "first"
    assert await client.complete() == "second"
    assert await client.complete() == "third"


@pytest.mark.asyncio
async def test_brix_replay_error_on_missing_session(tmp_path: Path):
    with pytest.raises(BrixReplayError, match="no DRE session file"):
        BrixReplayClient(session_id="nonexistent", log_path=tmp_path)


@pytest.mark.asyncio
async def test_brix_replay_error_when_calls_exceed_records(tmp_path: Path):
    session_id = "short-test"
    sessions_dir = tmp_path / ".brix_sessions"
    _write_dre_records(
        sessions_dir,
        session_id,
        [
            {
                "run_id": "r1",
                "sequence": 1,
                "content": "only",
                "content_type": "str",
                "usage": None,
            },
        ],
    )
    client = BrixReplayClient(session_id=session_id, log_path=tmp_path)
    await client.complete()  # first succeeds
    with pytest.raises(BrixReplayError, match="no recorded response"):
        await client.complete()  # second fails


# ---------------------------------------------------------------------------
# BRIX.replay() factory
# ---------------------------------------------------------------------------


def test_brix_replay_returns_replay_client(tmp_path: Path):
    session_id = "factory-test"
    sessions_dir = tmp_path / ".brix_sessions"
    _write_dre_records(
        sessions_dir,
        session_id,
        [
            {"run_id": "r1", "sequence": 1, "content": "hi", "content_type": "str", "usage": None},
        ],
    )
    client = BRIX.replay(session_id=session_id, log_path=tmp_path)
    assert isinstance(client, BrixReplayClient)


@pytest.mark.asyncio
async def test_acomplete_alias(tmp_path: Path):
    session_id = "alias-test"
    sessions_dir = tmp_path / ".brix_sessions"
    _write_dre_records(
        sessions_dir,
        session_id,
        [
            {"run_id": "r1", "sequence": 1, "content": "hi", "content_type": "str", "usage": None},
        ],
    )
    client = BrixReplayClient(session_id=session_id, log_path=tmp_path)
    result = await client.acomplete()
    assert result == "hi"


# ---------------------------------------------------------------------------
# Pydantic model reconstruction
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pydantic_response_with_schema_returns_model_instance(tmp_path: Path):
    session_id = "pydantic-test"
    sessions_dir = tmp_path / ".brix_sessions"
    _write_dre_records(
        sessions_dir,
        session_id,
        [
            {
                "run_id": "r1",
                "sequence": 1,
                "content": {"answer": "yes", "confidence": 0.9},
                "content_type": "pydantic:OutputModel",
                "usage": None,
            },
        ],
    )
    client = BrixReplayClient(session_id=session_id, log_path=tmp_path, schema=OutputModel)
    result = await client.complete()
    assert isinstance(result, OutputModel)
    assert result.answer == "yes"
    assert result.confidence == pytest.approx(0.9)


@pytest.mark.asyncio
async def test_pydantic_response_without_schema_returns_dict_and_warns(tmp_path: Path):
    session_id = "pydantic-no-schema"
    sessions_dir = tmp_path / ".brix_sessions"
    _write_dre_records(
        sessions_dir,
        session_id,
        [
            {
                "run_id": "r1",
                "sequence": 1,
                "content": {"answer": "yes", "confidence": 0.9},
                "content_type": "pydantic:OutputModel",
                "usage": None,
            },
        ],
    )
    client = BrixReplayClient(session_id=session_id, log_path=tmp_path, schema=None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = await client.complete()
    assert isinstance(result, dict)
    assert any(
        "schema" in str(w.message).lower() or "pydantic" in str(w.message).lower() for w in caught
    )


# ---------------------------------------------------------------------------
# Out-of-order records sorted by sequence
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_records_sorted_by_sequence_regardless_of_file_order(tmp_path: Path):
    session_id = "order-test"
    sessions_dir = tmp_path / ".brix_sessions"
    # Write in wrong order
    _write_dre_records(
        sessions_dir,
        session_id,
        [
            {
                "run_id": "r3",
                "sequence": 3,
                "content": "third",
                "content_type": "str",
                "usage": None,
            },
            {
                "run_id": "r1",
                "sequence": 1,
                "content": "first",
                "content_type": "str",
                "usage": None,
            },
            {
                "run_id": "r2",
                "sequence": 2,
                "content": "second",
                "content_type": "str",
                "usage": None,
            },
        ],
    )
    client = BrixReplayClient(session_id=session_id, log_path=tmp_path)
    assert await client.complete() == "first"
    assert await client.complete() == "second"
    assert await client.complete() == "third"


# ---------------------------------------------------------------------------
# Mixed content types
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mixed_str_and_pydantic_responses(tmp_path: Path):
    session_id = "mixed-test"
    sessions_dir = tmp_path / ".brix_sessions"
    _write_dre_records(
        sessions_dir,
        session_id,
        [
            {
                "run_id": "r1",
                "sequence": 1,
                "content": "plain string",
                "content_type": "str",
                "usage": None,
            },
            {
                "run_id": "r2",
                "sequence": 2,
                "content": {"answer": "model", "confidence": 0.7},
                "content_type": "pydantic:OutputModel",
                "usage": None,
            },
            {
                "run_id": "r3",
                "sequence": 3,
                "content": "another string",
                "content_type": "str",
                "usage": None,
            },
        ],
    )
    client = BrixReplayClient(session_id=session_id, log_path=tmp_path, schema=OutputModel)
    r1 = await client.complete()
    r2 = await client.complete()
    r3 = await client.complete()
    assert r1 == "plain string"
    assert isinstance(r2, OutputModel)
    assert r3 == "another string"


# ---------------------------------------------------------------------------
# Round-trip: record via ObservabilityGuard then replay
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_round_trip_record_and_replay(tmp_path: Path):
    """Write a session via ObservabilityGuard, then replay it exactly."""
    from brix.context import ExecutionContext

    session_id = "round-trip"
    guard = ObservabilityGuard(log_path=tmp_path, guard_names=["observability"])

    ctx = ExecutionContext.new_session()
    ctx.session_id = session_id

    responses = ["Hello world", "How can I help?"]
    for i, content in enumerate(responses):
        ctx.run_id = f"run-{i}"
        ctx.call_count = i + 1
        from brix.guards.protocol import CallRequest

        call_req = CallRequest(messages=[{"role": "user", "content": "test"}], model="gpt-4o-mini")
        call_resp = CallResponse(content=content)
        await guard.pre_call(call_req, ctx)
        await guard.post_call(call_req, call_resp, ctx)

    # Replay
    client = BRIX.replay(session_id=session_id, log_path=tmp_path)
    assert client.total_calls == 2
    assert await client.complete() == "Hello world"
    assert await client.complete() == "How can I help?"


# ---------------------------------------------------------------------------
# Client properties
# ---------------------------------------------------------------------------


def test_session_id_property(tmp_path: Path):
    session_id = "prop-test"
    sessions_dir = tmp_path / ".brix_sessions"
    _write_dre_records(
        sessions_dir,
        session_id,
        [
            {"run_id": "r1", "sequence": 1, "content": "hi", "content_type": "str", "usage": None},
        ],
    )
    client = BrixReplayClient(session_id=session_id, log_path=tmp_path)
    assert client.session_id == session_id
    assert client.total_calls == 1
    assert client.calls_remaining == 1


# ---------------------------------------------------------------------------
# BRIX.purge_sessions
# ---------------------------------------------------------------------------


def test_purge_sessions_deletes_old_files(tmp_path: Path):
    sessions_dir = tmp_path / ".brix_sessions"
    sessions_dir.mkdir()
    old_file = sessions_dir / "old-session.jsonl"
    old_file.write_text('{"run_id": "x", "sequence": 1}\n')
    # Make the file appear old
    import os

    old_time = 0.0  # epoch — definitely older than 7 days
    os.utime(old_file, (old_time, old_time))

    deleted = BRIX.purge_sessions(tmp_path, older_than_days=7)
    assert deleted == 1
    assert not old_file.exists()


def test_purge_sessions_keeps_recent_files(tmp_path: Path):
    sessions_dir = tmp_path / ".brix_sessions"
    sessions_dir.mkdir()
    recent_file = sessions_dir / "recent-session.jsonl"
    recent_file.write_text('{"run_id": "x", "sequence": 1}\n')
    # File mtime defaults to now — recent

    deleted = BRIX.purge_sessions(tmp_path, older_than_days=7)
    assert deleted == 0
    assert recent_file.exists()
