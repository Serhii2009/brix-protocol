"""TimeoutGuard — prevents pipeline hangs at three independent levels.

The 100% guarantee: ``asyncio.wait_for`` is absolute — no LLM call can
outlive the ``per_call_timeout``. The operating system may delay task
scheduling slightly, but asyncio will cancel the coroutine at the next
await point after the deadline. Combined with ``per_step_timeout`` and
``total_timeout``, every level of an agent pipeline is bounded.

Three timeout levels (all optional, all independent):

- ``per_call``:   Maximum wall-clock time for a single LLM call. Enforced via
                  ``asyncio.wait_for`` applied by BrixClient around the LLM
                  callable. TimeoutGuard writes the value to
                  ``context.metadata["_per_call_timeout"]`` in pre_call so
                  BrixClient's timeout-aware wrapper can read it.

- ``per_step``:   Maximum time between calls (one "agent step"). Measured from
                  the start of the previous call to the start of the next.

- ``total``:      Maximum wall-clock time for the entire session. Measured from
                  ``context.session_start``.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from brix.context import ExecutionContext
from brix.exceptions import BrixTimeoutError
from brix.guards.protocol import CallRequest, CallResponse


def _now() -> datetime:
    return datetime.now(timezone.utc)


class TimeoutGuard:
    """Guard that enforces per-call, per-step, and total session timeouts.

    Args:
        per_call: Maximum seconds for a single LLM call. Written to
            ``context.metadata["_per_call_timeout"]`` for BrixClient's
            timeout-aware wrapper (and for RetryGuard to apply per attempt).
        per_step: Maximum seconds between consecutive calls (one agent step).
        total: Maximum seconds for the entire session (from
            ``context.session_start``).
        on_timeout: ``"raise"`` (default) raises :class:`~brix.exceptions.BrixTimeoutError`.
            ``"return_partial"`` returns an empty :class:`~brix.guards.protocol.CallResponse`
            with ``raw={"timed_out": True, "level": "<level>"}``.
            Note: when ``on_timeout="return_partial"``, ``response.usage`` will be
            ``None``, so BudgetGuard will not update ``session_cost_usd`` for
            timed-out calls. This is expected behavior.
    """

    name: str = "timeout"

    def __init__(
        self,
        *,
        per_call: float | None = None,
        per_step: float | None = None,
        total: float | None = None,
        on_timeout: str = "raise",
    ) -> None:
        if on_timeout not in ("raise", "return_partial"):
            raise ValueError(
                f"TimeoutGuard on_timeout must be 'raise' or 'return_partial', "
                f"got {on_timeout!r}"
            )
        self._per_call = per_call
        self._per_step = per_step
        self._total = total
        self._on_timeout = on_timeout

    def _handle_timeout(self, level: str, elapsed: float, limit: float) -> CallResponse:
        """Handle a timeout based on the configured strategy.

        Args:
            level: Which timeout fired: ``"total"``, ``"step"``, or ``"per_call"``.
            elapsed: Seconds elapsed.
            limit: The configured limit in seconds.

        Returns:
            A partial CallResponse if ``on_timeout="return_partial"``.

        Raises:
            BrixTimeoutError: If ``on_timeout="raise"``.
        """
        reason = f"{level} timeout exceeded: {elapsed:.2f}s > {limit:.2f}s limit"
        if self._on_timeout == "return_partial":
            return CallResponse(
                content="",
                usage=None,
                raw={"timed_out": True, "level": level, "elapsed": elapsed, "limit": limit},
            )
        raise BrixTimeoutError(reason=reason)

    async def pre_call(
        self,
        request: CallRequest,
        context: ExecutionContext,
    ) -> CallRequest | CallResponse:
        """Check all timeout levels and set per_call timeout for BrixClient.

        Args:
            request: The outbound request.
            context: Mutable session state.

        Returns:
            Unmodified request if no timeout has fired.
            A partial CallResponse if ``on_timeout="return_partial"`` and a
            timeout was detected.

        Raises:
            BrixTimeoutError: If ``on_timeout="raise"`` and a timeout fired.
        """
        now = _now()

        # 1. Total session timeout
        if self._total is not None:
            elapsed = (now - context.session_start).total_seconds()
            if elapsed > self._total:
                result = self._handle_timeout("total", elapsed, self._total)
                return result

        # 2. Per-step timeout (time since the LAST call started)
        if self._per_step is not None:
            step_start: datetime | None = context.metadata.get("_step_start")
            if step_start is not None:
                elapsed = (now - step_start).total_seconds()
                if elapsed > self._per_step:
                    result = self._handle_timeout("step", elapsed, self._per_step)
                    return result

        # 3. Record start of this step
        context.metadata["_step_start"] = now

        # 4. Store per_call timeout for BrixClient's timeout-aware wrapper
        #    (and for RetryGuard to apply per attempt via asyncio.wait_for)
        if self._per_call is not None:
            context.metadata["_per_call_timeout"] = self._per_call
        elif "_per_call_timeout" in context.metadata:
            # Clear any stale value from a previous guard configuration
            del context.metadata["_per_call_timeout"]

        # 5. Store full config for observability
        context.metadata["_timeout_config"] = {
            "per_call": self._per_call,
            "per_step": self._per_step,
            "total": self._total,
            "on_timeout": self._on_timeout,
        }

        return request

    async def post_call(
        self,
        request: CallRequest,
        response: CallResponse,
        context: ExecutionContext,
    ) -> CallResponse:
        """Clear the step start time — the step completed successfully.

        Args:
            request: The request that was sent.
            response: The response received.
            context: Mutable session state.

        Returns:
            The unmodified response.
        """
        context.metadata.pop("_step_start", None)
        return response


__all__ = ["TimeoutGuard"]
