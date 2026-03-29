"""InterceptorChain — runs Guards in order around the LLM call.

The chain is the engine of the BRIX pipeline. It coordinates pre_call and
post_call hooks across all registered Guards, handles short-circuits, and
ensures post_call always runs in reverse for Guards that completed pre_call.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable

from brix.context import CallRecord, ExecutionContext
from brix.exceptions import BrixGuardBlockedError, BrixGuardError, BrixInternalError, BrixError
from brix.guards.protocol import CallRequest, CallResponse, Guard


class InterceptorChain:
    """Runs a sequence of Guards around a single LLM call.

    Execution order:
    1. pre_call runs on each Guard in registration order.
    2. If no Guard short-circuits: the LLM callable is invoked.
    3. post_call runs on each Guard in **reverse** registration order.

    If a Guard's pre_call returns None or raises BrixGuardBlockedError,
    the LLM call is skipped and post_call runs in reverse for all Guards
    that already completed pre_call.

    If a Guard's pre_call returns a CallResponse, the LLM call is skipped
    (short-circuit) and post_call runs in reverse for Guards that already
    completed pre_call (not including the short-circuiting Guard).

    Args:
        guards: Ordered list of Guards to run. An empty list is valid and
            causes the chain to pass requests directly to the LLM callable.
    """

    def __init__(self, guards: list[Guard]) -> None:
        self._guards = guards

    @property
    def guards(self) -> list[Guard]:
        """Read-only view of the registered Guards."""
        return list(self._guards)

    async def execute(
        self,
        request: CallRequest,
        context: ExecutionContext,
        llm_callable: Callable[[CallRequest], Awaitable[CallResponse]],
    ) -> CallResponse:
        """Execute the full pre_call → LLM → post_call pipeline.

        Args:
            request: The initial outbound request.
            context: Mutable session state for this call.
            llm_callable: Async callable that sends the request to the LLM.

        Returns:
            The final CallResponse after all post_call transforms.

        Raises:
            BrixGuardBlockedError: If a Guard blocks the request.
            BrixInternalError: If an unexpected error occurs inside a Guard.
        """
        guards_completed_pre: list[Guard] = []
        response: CallResponse | None = None
        blocked_error: BrixGuardBlockedError | None = None

        # --- pre_call phase ---
        try:
            for guard in self._guards:
                try:
                    result: CallRequest | CallResponse | None = await guard.pre_call(
                        request, context
                    )
                except BrixGuardBlockedError:
                    raise
                except Exception as exc:
                    raise BrixGuardError(
                        guard.name,
                        f"pre_call raised an unexpected error: {exc}",
                    ) from exc

                if result is None:
                    raise BrixGuardBlockedError(guard.name, "request blocked by guard")

                if isinstance(result, CallResponse):
                    # Short-circuit: use this response, skip LLM call.
                    # This guard is NOT added to guards_completed_pre because
                    # it did not "open" anything that needs closing in post_call.
                    response = result
                    break

                # result is a CallRequest — continue with modified request
                request = result
                guards_completed_pre.append(guard)

        except BrixGuardBlockedError as exc:
            blocked_error = exc

        # --- LLM call (skipped on block or short-circuit) ---
        if blocked_error is None and response is None:
            t0 = time.perf_counter()
            try:
                response = await llm_callable(request)
            except Exception:
                # Propagate LLM errors as-is; they are not Guard errors.
                raise
            finally:
                # Record latency even if the call failed (partial record).
                _ = time.perf_counter() - t0

        # --- post_call phase (always runs for guards_completed_pre, in reverse) ---
        if response is None and blocked_error is not None:
            # Build a stub response so post_call guards still receive something.
            # Guards should not mutate a blocked response, but they get to observe it.
            response = CallResponse(
                content="",
                usage=None,
                raw=None,
            )

        assert response is not None  # mypy narrowing

        post_call_error: Exception | None = None
        for guard in reversed(guards_completed_pre):
            try:
                response = await guard.post_call(request, response, context)
            except Exception as exc:
                # Don't wrap BrixErrors — let them propagate directly.
                if post_call_error is None:
                    if isinstance(exc, BrixError):
                        post_call_error = exc
                    else:
                        post_call_error = BrixInternalError(
                            f"[{guard.name}] post_call raised an unexpected error: {exc}"
                        )

        # Re-raise blocked error (takes priority over post_call errors)
        if blocked_error is not None:
            raise blocked_error

        if post_call_error is not None:
            raise post_call_error

        return response

    def _record_call(
        self,
        context: ExecutionContext,
        request: CallRequest,
        response: CallResponse,
        latency_ms: float,
    ) -> None:
        """Append a CallRecord to the context's call history."""
        from datetime import datetime, timezone

        context.call_history.append(
            CallRecord(
                request=request,
                response=response,
                timestamp=datetime.now(timezone.utc),
                latency_ms=latency_ms,
            )
        )


__all__ = ["InterceptorChain"]
