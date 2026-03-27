"""BrixReplayClient — deterministic replay of recorded BRIX sessions.

The 100% guarantee: given a session that was recorded with ObservabilityGuard
active and whose DRE writes succeeded, ``BrixReplayClient.complete()`` will
return the recorded responses in the exact original order, at zero LLM cost,
with no network access required.

If a response is missing for any step in the replay sequence,
:class:`~brix.exceptions.BrixReplayError` is raised with a clear explanation
— replay never silently returns wrong data.

Pydantic model reconstruction:
  When a response was originally a Pydantic model instance (written by
  SchemaGuard), the DRE record stores its ``model_dump()`` dict. To reconstruct
  the original typed instance during replay, pass ``schema=YourModel`` to
  ``BRIX.replay()``. Without ``schema=``, a ``dict`` is returned and a
  ``UserWarning`` is emitted explaining the divergence.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

from brix.exceptions import BrixReplayError

if TYPE_CHECKING:
    from pydantic import BaseModel


class BrixReplayClient:
    """Replays a recorded BRIX session without making live LLM calls.

    Do not instantiate directly — use ``BRIX.replay()`` instead.

    Args:
        session_id: The session ID of the recorded session. Shown in every
            ``ObservabilityGuard`` audit entry as ``session_id``.
        log_path: Directory where ObservabilityGuard wrote the DRE session files
            (the same ``log_path`` passed to ``BRIX.wrap()``).
        schema: Optional Pydantic model class. When provided and the recorded
            response was a Pydantic model, the dict is reconstructed into a
            typed model instance via ``schema.model_validate()``.
    """

    def __init__(
        self,
        session_id: str,
        log_path: Path,
        schema: type[BaseModel] | None = None,
    ) -> None:
        self._session_id = session_id
        self._schema = schema
        self._records = self._load_records(session_id, log_path)
        self._index = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def complete(
        self,
        messages: list[dict[str, Any]] | None = None,
        *,
        model: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Return the next recorded response in the session.

        The ``messages``, ``model``, and ``kwargs`` parameters are accepted for
        API compatibility with ``BrixClient.complete()`` but are ignored — replay
        returns the recorded response regardless of the input.

        Args:
            messages: Ignored during replay.
            model: Ignored during replay.
            **kwargs: Ignored during replay.

        Returns:
            The recorded response content. If the original response was a Pydantic
            model and ``schema=`` was provided, returns a reconstructed model instance.
            If ``schema=`` was not provided, returns the recorded ``dict`` and emits
            a ``UserWarning``.

        Raises:
            BrixReplayError: If there is no recorded response for the current call
                index (i.e., more calls were made during replay than were recorded).
        """
        if self._index >= len(self._records):
            raise BrixReplayError(
                f"no recorded response for call #{self._index} "
                f"(session '{self._session_id}' has {len(self._records)} recorded call(s))"
            )

        record = self._records[self._index]
        self._index += 1

        content_type: str = record.get("content_type", "str")
        content: Any = record["content"]

        if content_type.startswith("pydantic:"):
            if self._schema is not None:
                return self._schema.model_validate(content)
            class_name = content_type.split(":", 1)[1]
            warnings.warn(
                f"BrixReplayClient: response for call #{self._index - 1} was originally a "
                f"Pydantic model ({class_name!r}) but no schema= was provided to BRIX.replay(). "
                "Pass schema=YourModel for type-accurate reconstruction. "
                "Returning dict instead.",
                stacklevel=2,
            )

        return content

    # Alias for explicit async naming preference
    acomplete = complete

    @property
    def session_id(self) -> str:
        """The session ID being replayed."""
        return self._session_id

    @property
    def total_calls(self) -> int:
        """Total number of recorded calls available for replay."""
        return len(self._records)

    @property
    def calls_remaining(self) -> int:
        """Number of recorded calls not yet replayed."""
        return max(0, len(self._records) - self._index)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_records(session_id: str, log_path: Path) -> list[dict[str, Any]]:
        """Load and sort DRE session records from disk.

        Args:
            session_id: Session ID to look up.
            log_path: Root log directory.

        Returns:
            List of records sorted by ``sequence`` (ascending).

        Raises:
            BrixReplayError: If the session file does not exist.
        """
        session_path = log_path / ".brix_sessions" / f"{session_id}.jsonl"
        if not session_path.exists():
            raise BrixReplayError(
                f"no DRE session file found for session_id={session_id!r}. "
                f"Expected path: {session_path}"
            )

        records: list[dict[str, Any]] = []
        with session_path.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise BrixReplayError(
                        f"malformed DRE record on line {line_num} of {session_path}: {exc}"
                    ) from exc

        # Sort by sequence to handle any out-of-order writes
        records.sort(key=lambda r: r.get("sequence", 0))
        return records


__all__ = ["BrixReplayClient"]
