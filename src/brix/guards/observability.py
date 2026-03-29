"""ObservabilityGuard — cryptographically chained audit logs with deterministic replay.

The 100% guarantee (refined):
- If a call completes with ObservabilityGuard active, a trace entry is **always**
  added to the in-memory buffer.
- If ``log_path`` is set, the entry is also written to disk. With
  ``strict_mode=False`` (default), a disk write failure is logged at WARNING
  level and the pipeline continues. With ``strict_mode=True``, a write failure
  raises :class:`~brix.exceptions.BrixGuardError` and the pipeline fails fast.
- **Replay requires successful disk writes.** The in-memory buffer is not
  sufficient for replay — it is cleared when the process exits.

Two layers:

Layer 1 — Audit log (``brix_audit.jsonl``):
  Every call is recorded with a complete audit entry including run_id,
  session_id, model, token counts, cost, latency, prompt hash, response hash,
  and a ``chain_hash``. The chain_hash is the SHA-256 of the *previous entry's
  full JSON serialization*, creating a cryptographic chain where any modification
  or deletion of a historical record invalidates all subsequent hashes.

Layer 2 — DRE (Deterministic Replay Engine):
  The response to every call is recorded in
  ``.brix_sessions/{session_id}.jsonl``. Use ``BRIX.replay()`` to replay a
  session without making live LLM calls — every response is replayed exactly
  as recorded, in order, at zero cost.

Session file rotation: when ``max_session_records`` is set, the session file
is renamed with a UTC timestamp suffix once it reaches that many entries, and
a fresh file starts. This bounds individual file size for long-running pipelines.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from brix.context import ExecutionContext
from brix.exceptions import BrixGuardError
from brix.guards.protocol import CallRequest, CallResponse


class ObservabilityGuard:
    """Guard that records every LLM call to an audit log and DRE session file.

    Args:
        log_path: Directory for JSONL audit log and DRE session files.
            When ``None``, writes to the in-memory buffer only (no disk I/O).
        guard_names: Names of all guards active in this session (for the audit
            entry). Pass ``[g.name for g in guards]`` at construction time from
            ``BRIX.wrap()``.
        buffer_size: Maximum number of trace entries to keep in memory.
            Oldest entries are evicted when the buffer is full. Default 1000.
        strict_mode: When ``True``, disk write failures raise
            :class:`~brix.exceptions.BrixGuardError` instead of logging at
            WARNING. Default ``False`` (best-effort writes).
        max_session_records: When set, the DRE session file is rotated after
            this many records. The rotated file is renamed
            ``{session_id}.{YYYYMMDDTHHMMSS}.jsonl``. Default ``None`` (no rotation).
    """

    name: str = "observability"

    def __init__(
        self,
        log_path: Path | None,
        guard_names: list[str],
        *,
        buffer_size: int = 1000,
        strict_mode: bool = False,
        max_session_records: int | None = None,
    ) -> None:
        import asyncio  # noqa: PLC0415

        self._log_path = log_path
        self._guard_names = guard_names
        self._buffer: deque[dict[str, Any]] = deque(maxlen=buffer_size if buffer_size > 0 else None)
        self._prev_entry_json: str | None = None
        self._start_times: dict[str, float] = {}
        self._lock: asyncio.Lock | None = None
        self._strict_mode = strict_mode
        self._max_session_records = max_session_records
        self._session_record_counts: dict[str, int] = {}

    async def pre_call(
        self,
        request: CallRequest,
        context: ExecutionContext,
    ) -> CallRequest:
        """Record the start time for this call.

        Args:
            request: The outbound request.
            context: Mutable session state.

        Returns:
            The unmodified request.
        """
        self._start_times[context.run_id] = time.perf_counter()
        return request

    async def post_call(
        self,
        request: CallRequest,
        response: CallResponse,
        context: ExecutionContext,
    ) -> CallResponse:
        """Build and write the audit entry and DRE session record.

        The asyncio.Lock serializes all writes for this guard instance, ensuring
        the chain_hash sequence is consistent even under concurrent asyncio.gather
        usage.

        Args:
            request: The request that was sent.
            response: The (possibly SchemaGuard-transformed) response.
            context: Mutable session state.

        Returns:
            The unmodified response.
        """
        import asyncio  # noqa: PLC0415

        if self._lock is None:
            self._lock = asyncio.Lock()

        async with self._lock:
            entry = self._build_entry(request, response, context)
            self._buffer.append(entry)
            self._prev_entry_json = json.dumps(entry, sort_keys=True, default=str)

            if self._log_path is not None:
                self._write_audit(entry)
                self._write_dre(context.session_id, context.run_id, context.call_count, response)

        return response

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def get_traces(self) -> list[dict[str, Any]]:
        """Return trace entries from the in-memory buffer, most-recent first."""
        return list(reversed(self._buffer))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_entry(
        self,
        request: CallRequest,
        response: CallResponse,
        context: ExecutionContext,
    ) -> dict[str, Any]:
        run_id = context.run_id
        t_start = self._start_times.pop(run_id, time.perf_counter())
        latency_ms = (time.perf_counter() - t_start) * 1000.0

        messages_json = json.dumps(request.messages, sort_keys=True, default=str)
        prompt_hash = hashlib.sha256(messages_json.encode()).hexdigest()
        response_hash = hashlib.sha256(str(response.content).encode()).hexdigest()

        # chain_hash: SHA-256 of previous entry's full JSON; genesis = SHA-256 of {"genesis": session_id}
        prev_json = self._prev_entry_json or json.dumps({"genesis": context.session_id})
        chain_hash = hashlib.sha256(prev_json.encode()).hexdigest()

        usage = response.usage or {}
        cost = context.metadata.get("last_call_cost", {}).get("actual")

        return {
            "run_id": run_id,
            "session_id": context.session_id,
            "sequence": context.call_count,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": request.model,
            "prompt_tokens": usage.get("prompt_tokens") or usage.get("input_tokens"),
            "completion_tokens": usage.get("completion_tokens") or usage.get("output_tokens"),
            "cost_usd": cost,
            "latency_ms": latency_ms,
            "prompt_hash": prompt_hash,
            "response_hash": response_hash,
            "guards_active": self._guard_names,
            "chain_hash": chain_hash,
        }

    def _write_audit(self, entry: dict[str, Any]) -> None:
        try:
            audit_path = self._log_path / "brix_audit.jsonl"  # type: ignore[operator]
            audit_path.parent.mkdir(parents=True, exist_ok=True)
            with audit_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except Exception as exc:
            if self._strict_mode:
                raise BrixGuardError(self.name, f"audit log write failed: {exc}") from exc
            logging.warning("ObservabilityGuard: audit log write failed: %s", exc)

    def _write_dre(
        self,
        session_id: str,
        run_id: str,
        sequence: int,
        response: CallResponse,
    ) -> None:
        try:
            sessions_dir = self._log_path / ".brix_sessions"  # type: ignore[operator]
            sessions_dir.mkdir(parents=True, exist_ok=True)

            count = self._session_record_counts.get(session_id, 0)
            session_path = sessions_dir / f"{session_id}.jsonl"

            # Rotate if at capacity
            if (
                self._max_session_records is not None
                and count > 0
                and count % self._max_session_records == 0
            ):
                ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
                rotated = sessions_dir / f"{session_id}.{ts}.jsonl"
                if session_path.exists():
                    session_path.rename(rotated)

            content = response.content
            if hasattr(content, "model_dump"):
                content_type = f"pydantic:{type(content).__name__}"
                content_serialized: Any = content.model_dump()
            else:
                content_type = "str"
                content_serialized = content

            record = {
                "run_id": run_id,
                "sequence": sequence,
                "content": content_serialized,
                "content_type": content_type,
                "usage": response.usage,
            }

            with session_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")

            self._session_record_counts[session_id] = count + 1

        except Exception as exc:
            if self._strict_mode:
                raise BrixGuardError(self.name, f"DRE session write failed: {exc}") from exc
            logging.warning("ObservabilityGuard: DRE session write failed: %s", exc)


__all__ = ["ObservabilityGuard"]
