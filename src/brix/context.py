"""ExecutionContext — shared mutable state flowing through a BRIX pipeline call.

Each BrixClient session has one ExecutionContext that persists across calls.
Guards read from and write to this context to coordinate without coupling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any
from uuid import uuid4

if TYPE_CHECKING:
    from brix.guards.protocol import CallRequest, CallResponse


@dataclass
class CallRecord:
    """Immutable record of a single completed LLM call within a session.

    Args:
        request: The CallRequest that was sent (after all pre_call transforms).
        response: The CallResponse that was received (after all post_call transforms).
        timestamp: UTC datetime when the call completed.
        latency_ms: Wall-clock latency of the call in milliseconds.
    """

    request: "CallRequest"
    response: "CallResponse"
    timestamp: datetime
    latency_ms: float


@dataclass
class ExecutionContext:
    """Mutable session state shared across all Guards in a pipeline call.

    One ExecutionContext lives on a BrixClient instance for its entire lifetime.
    Each call updates run_id, call_count, call_history, and session_cost_usd in place.

    Guards should use the ``metadata`` dict to store Guard-specific data using
    their own guard name as a namespace key to avoid collisions.

    Example::

        context.metadata["budget"] = {"tokens_used": 1234}
        context.metadata["regulated_result"] = structured_result

    Args:
        run_id: UUID4 string, unique per LLM call. Refreshed at the start of
            each BrixClient.complete() invocation.
        session_id: UUID4 string, fixed for the lifetime of the BrixClient.
        call_count: Number of complete() calls made in this session.
        session_cost_usd: Cumulative estimated cost in USD for this session.
            Guards should update this after each call.
        session_start: UTC datetime when the BrixClient was constructed.
        call_history: Ordered list of all completed calls in this session.
        metadata: Open namespace for Guard-specific data.
    """

    run_id: str
    session_id: str
    call_count: int = 0
    session_cost_usd: float = 0.0
    session_start: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    call_history: list[CallRecord] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def new_session(cls) -> "ExecutionContext":
        """Create a fresh ExecutionContext for a new BrixClient session."""
        return cls(
            run_id=str(uuid4()),
            session_id=str(uuid4()),
        )


__all__ = ["CallRecord", "ExecutionContext"]
