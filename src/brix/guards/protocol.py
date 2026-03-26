"""Guard Protocol — the interface every BRIX Guard implements.

Guards are the building blocks of the BRIX Interceptor Chain. Each Guard
solves exactly one production failure mode and is completely independent
of every other Guard.

Adding a new Guard means:
1. Create one new file implementing the Guard Protocol.
2. Register one parameter in BRIX.wrap().
Nothing else changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, runtime_checkable

from typing import Protocol

from brix.context import ExecutionContext


@dataclass
class CallRequest:
    """Represents an outbound LLM request.

    Guards receive this in pre_call and may return a modified version.

    Args:
        messages: List of message dicts in OpenAI chat format, e.g.
            [{"role": "user", "content": "Hello"}].
        model: The model identifier string (e.g. "gpt-4o", "claude-3-5-sonnet").
        kwargs: Additional keyword arguments forwarded to the LLM client
            (temperature, max_tokens, etc.).
    """

    messages: list[dict[str, Any]]
    model: str
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass
class CallResponse:
    """Represents an inbound LLM response.

    Guards receive this in post_call and may return a modified version.

    Args:
        content: The text content of the response.
        usage: Token usage dict if available (prompt_tokens, completion_tokens, etc.).
        raw: The raw response object from the LLM provider, if available.
    """

    content: str
    usage: dict[str, Any] | None = None
    raw: Any = None


@runtime_checkable
class Guard(Protocol):
    """Protocol that every BRIX Guard must implement.

    A Guard is a self-contained unit that intercepts LLM calls at two points:

    - **pre_call**: Runs before the LLM call. Can inspect or modify the request,
      short-circuit (skip the LLM call entirely by returning a CallResponse),
      or block the request (return None or raise BrixGuardBlockedError).

    - **post_call**: Runs after the LLM call (or after a short-circuit).
      Can inspect or transform the response.

    Guards in the chain run in registration order for pre_call and in reverse
    order for post_call (innermost Guard wraps the LLM call).

    Return values from pre_call:
        - ``CallRequest``: Continue with the (possibly modified) request.
        - ``CallResponse``: Short-circuit — skip the LLM call, use this as the
          response. Useful for Guards that generate the response themselves
          (e.g. RegulatedGuard, CacheGuard).
        - ``None``: Block the request. InterceptorChain raises BrixGuardBlockedError.
          Guards may also raise BrixGuardBlockedError directly to include a custom
          reason message.

    Example Guard implementation::

        class LoggingGuard:
            name = "logging"

            async def pre_call(
                self, request: CallRequest, context: ExecutionContext
            ) -> CallRequest | CallResponse | None:
                print(f"[{context.run_id}] Sending {len(request.messages)} messages")
                return request  # pass through unchanged

            async def post_call(
                self, request: CallRequest, response: CallResponse,
                context: ExecutionContext
            ) -> CallResponse:
                print(f"[{context.run_id}] Got: {response.content[:50]}")
                return response  # pass through unchanged
    """

    @property
    def name(self) -> str:
        """Unique identifier for this Guard, used in error messages and logs."""
        ...

    async def pre_call(
        self,
        request: CallRequest,
        context: ExecutionContext,
    ) -> CallRequest | CallResponse | None:
        """Hook that runs before the LLM call.

        Args:
            request: The outbound request. Modify and return to change what the
                LLM receives.
            context: Mutable session state shared across all Guards.

        Returns:
            - Modified/unmodified ``CallRequest`` to continue normally.
            - A ``CallResponse`` to short-circuit (skip the LLM call).
            - ``None`` to block the request.

        Raises:
            BrixGuardBlockedError: To block with a custom reason message.
        """
        ...

    async def post_call(
        self,
        request: CallRequest,
        response: CallResponse,
        context: ExecutionContext,
    ) -> CallResponse:
        """Hook that runs after the LLM call (or after a short-circuit).

        Runs in reverse Guard order relative to pre_call (innermost wraps the call).
        Always runs for Guards that completed pre_call, even if later Guards blocked.

        Args:
            request: The final request that was sent (after all pre_call transforms).
            response: The response to inspect or transform.
            context: Mutable session state shared across all Guards.

        Returns:
            The (possibly modified) response.
        """
        ...


__all__ = ["CallRequest", "CallResponse", "Guard"]
