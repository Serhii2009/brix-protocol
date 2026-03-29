"""BudgetGuard — prevents LLM cost overruns.

The 100% guarantee: BudgetGuard estimates the cost of a call BEFORE making it
using tiktoken token counts and a bundled price table. If the estimated cost
would push the session over the configured limit, the call is blocked before any
tokens are sent to the LLM. No money is spent on a blocked call.

Even if the actual cost differs from the estimate (e.g. due to completion length),
the pre-check uses the estimate conservatively. For a stronger bound, use
``strategy="block"`` (default) and set ``max_cost_usd`` below your hard limit.
"""

from __future__ import annotations

import asyncio
import warnings
from typing import Any

import tiktoken

from brix.context import ExecutionContext
from brix.exceptions import BrixBudgetError
from brix.guards._pricing import get_price
from brix.guards.protocol import CallRequest, CallResponse


def _count_tokens(messages: list[dict[str, Any]], model: str) -> int:
    """Count the tokens in a list of chat messages for the given model.

    Uses tiktoken. Falls back to ``cl100k_base`` for unknown models.

    Args:
        messages: List of message dicts in OpenAI chat format.
        model: Model identifier used to select the encoding.

    Returns:
        Estimated token count for the prompt.
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")

    # OpenAI's official per-message overhead formula (GPT-3.5/4 style):
    # 3 tokens per message (role + content framing) + 3 for reply primer
    num_tokens = 3  # reply primer
    for msg in messages:
        num_tokens += 3  # per-message overhead
        for value in msg.values():
            if isinstance(value, str):
                num_tokens += len(encoding.encode(value))
    return num_tokens


class BudgetGuard:
    """Guard that blocks LLM calls that would exceed the session cost limit.

    BudgetGuard estimates the prompt cost before the call and blocks if the
    estimated spend would push the session over ``max_cost_usd``. After the
    call, it updates ``context.session_cost_usd`` with the actual cost from
    ``response.usage``.

    Args:
        max_cost_usd: Maximum cumulative session cost in USD. Any call whose
            estimated cost would exceed this limit is blocked (or warned,
            depending on ``strategy``).
        strategy: ``"block"`` (default) raises :class:`~brix.exceptions.BrixBudgetError`
            when the limit would be exceeded. ``"warn"`` emits a warning and
            continues.
        warning_threshold: Fraction of ``max_cost_usd`` at which to emit a
            proactive warning regardless of strategy. Default 0.8 (80%).
    """

    name: str = "budget"

    def __init__(
        self,
        max_cost_usd: float,
        *,
        strategy: str = "block",
        warning_threshold: float = 0.8,
    ) -> None:
        if strategy not in ("block", "warn"):
            raise ValueError(f"BudgetGuard strategy must be 'block' or 'warn', got {strategy!r}")
        if not (0.0 < warning_threshold <= 1.0):
            raise ValueError(
                f"BudgetGuard warning_threshold must be in (0, 1], got {warning_threshold}"
            )
        self._max_cost = max_cost_usd
        self._strategy = strategy
        self._warning_threshold = warning_threshold
        # Lazy-initialized in post_call to avoid RuntimeError when constructing
        # outside a running event loop (asyncio.Lock() raises in that context).
        self._lock: asyncio.Lock | None = None

    async def pre_call(
        self,
        request: CallRequest,
        context: ExecutionContext,
    ) -> CallRequest:
        """Estimate prompt cost and block if the budget would be exceeded.

        Args:
            request: The outbound LLM request.
            context: Mutable session state.

        Returns:
            The unmodified request if the budget allows the call.

        Raises:
            BrixBudgetError: If ``strategy="block"`` and the estimated cost
                would exceed ``max_cost_usd``.
        """
        prompt_tokens = _count_tokens(request.messages, request.model)
        input_price, _ = get_price(request.model)
        estimated_cost = prompt_tokens * input_price

        current_spend = context.session_cost_usd
        projected_spend = current_spend + estimated_cost

        # Proactive warning at threshold (skip when max_cost is zero — zero means
        # "block everything"; emitting a threshold warning would be misleading)
        if self._max_cost > 0 and projected_spend >= self._warning_threshold * self._max_cost:
            warnings.warn(
                f"BudgetGuard: session cost ${current_spend:.4f} + estimated "
                f"${estimated_cost:.4f} = ${projected_spend:.4f} is at or above "
                f"{self._warning_threshold * 100:.0f}% of limit ${self._max_cost:.4f}.",
                stacklevel=3,
            )

        # Budget enforcement
        if projected_spend > self._max_cost:
            msg = (
                f"estimated call cost ${estimated_cost:.6f} would push session total "
                f"${current_spend:.6f} to ${projected_spend:.6f}, "
                f"exceeding limit ${self._max_cost:.6f}"
            )
            if self._strategy == "block":
                raise BrixBudgetError(reason=msg)
            else:
                warnings.warn(f"BudgetGuard (warn mode): {msg}", stacklevel=3)

        # Record estimate for post_call to compare against actuals
        context.metadata.setdefault("last_call_cost", {})["estimated"] = estimated_cost

        return request

    async def post_call(
        self,
        request: CallRequest,
        response: CallResponse,
        context: ExecutionContext,
    ) -> CallResponse:
        """Update session cost from actual token usage in the response.

        If ``response.usage`` is None (e.g. the call timed out and returned
        a partial response), this method is a no-op and ``session_cost_usd``
        is not updated for this call.

        Args:
            request: The final request that was sent.
            response: The response containing optional ``usage`` data.
            context: Mutable session state to update.

        Returns:
            The unmodified response.
        """
        if response.usage is None:
            return response

        # Lazy lock initialization: safe because asyncio is single-threaded
        if self._lock is None:
            self._lock = asyncio.Lock()

        async with self._lock:
            usage = response.usage
            prompt_tokens = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
            completion_tokens = usage.get("completion_tokens") or usage.get("output_tokens") or 0
            input_price, output_price = get_price(request.model)
            actual_cost = (prompt_tokens * input_price) + (completion_tokens * output_price)

            context.session_cost_usd += actual_cost

            # Store cost breakdown for observability
            cost_record: dict[str, Any] = {
                "estimated": context.metadata.get("last_call_cost", {}).get("estimated", 0.0),
                "actual": actual_cost,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "model": request.model,
            }
            context.metadata["last_call_cost"] = cost_record

            breakdown: list[dict[str, Any]] = context.metadata.setdefault(
                "session_cost_breakdown", []
            )
            breakdown.append(cost_record)

        return response


__all__ = ["BudgetGuard"]
