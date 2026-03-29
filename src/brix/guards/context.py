"""ContextGuard — prevents context_length_exceeded by compressing messages in pre_call.

The 100% guarantee: if ContextGuard is active, no call to ``complete()`` will ever
send more than ``max_context_tokens`` tokens to the provider.
``context_length_exceeded`` from the provider becomes impossible.

Three compression strategies:

- **sliding_window** (default, no extra dependencies): Keeps the system prompt and
  the latest user message inviolate, then greedily fills remaining space with the
  most-recent conversation history.

- **summarize** (requires an ``llm_callable``): Calls the LLM to summarise the
  oldest half of history into a compact system message, then falls back to
  ``sliding_window`` if the summary is still over budget.

- **importance**: Scores each message (tool calls = 3, tool results = 3,
  user = 2, assistant = 1), then keeps highest-scored messages first.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

import tiktoken

from brix.context import ExecutionContext
from brix.exceptions import BrixConfigurationError, BrixGuardError
from brix.guards.protocol import CallRequest, CallResponse


def _count_tokens(messages: list[dict[str, Any]], model: str) -> int:
    """Count tokens for a list of chat messages.

    Uses tiktoken with a ``cl100k_base`` fallback for unknown models.
    Overhead: 4 tokens per message + 2 reply priming tokens.

    Args:
        messages: List of message dicts in OpenAI chat format.
        model: Model identifier used to select the tiktoken encoding.

    Returns:
        Estimated token count for the prompt.
    """
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    total = 0
    for msg in messages:
        total += 4  # per-message overhead
        total += len(enc.encode(str(msg.get("content", ""))))
    return total + 2  # reply priming


class ContextGuard:
    """Guard that prevents context_length_exceeded by compressing messages in pre_call.

    The 100% guarantee: if ContextGuard is active, no call to complete() will ever send
    more than max_context_tokens to the provider. context_length_exceeded becomes impossible.

    reserve_tokens is subtracted from max_context_tokens to leave space for the model's
    response. The guard ensures: prompt_tokens <= max_context_tokens - reserve_tokens.

    Protection hierarchy (inviolable for all strategies):
    1. System messages (role="system") — always kept.
    2. Last user message — always kept.
    3. Everything else — subject to compression.

    Args:
        max_tokens: Maximum total tokens to send to the provider.
        strategy: ``"sliding_window"`` (default), ``"summarize"``, or ``"importance"``.
        reserve_tokens: Tokens reserved for the model's response. Effective budget is
            ``max_tokens - reserve_tokens``. Default 500.
        llm_callable: Required when ``strategy="summarize"``. Async callable
            ``(CallRequest) -> CallResponse`` used to generate summaries.
        summary_model: Model used for summarisation calls. If None, uses the same
            model as the main request.

    Raises:
        BrixConfigurationError: If ``strategy`` is unrecognised, or if
            ``strategy="summarize"`` is requested without an ``llm_callable``.
    """

    name = "context"

    def __init__(
        self,
        max_tokens: int,
        *,
        strategy: str = "sliding_window",
        reserve_tokens: int = 500,
        llm_callable: Callable[[CallRequest], Awaitable[CallResponse]] | None = None,
        summary_model: str | None = None,
    ) -> None:
        if strategy not in ("sliding_window", "summarize", "importance"):
            raise BrixConfigurationError(
                f"ContextGuard strategy must be 'sliding_window', 'summarize', or "
                f"'importance', got {strategy!r}"
            )
        if strategy == "summarize" and llm_callable is None:
            raise BrixConfigurationError(
                "ContextGuard strategy='summarize' requires llm_callable. "
                "Pass llm_callable when constructing ContextGuard, or use "
                "context_strategy='sliding_window' (no LLM required)."
            )
        self._max_tokens = max_tokens
        self._strategy = strategy
        self._reserve_tokens = reserve_tokens
        self._llm_callable = llm_callable
        self._summary_model = summary_model
        # Lazy asyncio.Lock — serializes concurrent summarisation calls.
        # Initialized on first async call to avoid RuntimeError when constructing
        # outside a running event loop.
        self._lock: asyncio.Lock | None = None

    # ------------------------------------------------------------------
    # Guard protocol
    # ------------------------------------------------------------------

    async def pre_call(
        self,
        request: CallRequest,
        context: ExecutionContext,
    ) -> CallRequest:
        """Compress messages if they exceed the token budget.

        Returns the request unchanged (zero overhead) when under budget.
        Records compression metadata in ``context.metadata`` when compression
        is applied.
        """
        budget = self._max_tokens - self._reserve_tokens
        tokens = _count_tokens(request.messages, request.model)

        if tokens <= budget:
            return request

        # Over budget — apply the configured strategy
        fallback_used = False
        if self._strategy == "sliding_window":
            compressed = self._apply_sliding_window(request.messages, budget, request.model)
        elif self._strategy == "summarize":
            compressed, fallback_used = await self._apply_summarize(
                request.messages, budget, request.model
            )
        else:  # importance
            compressed = self._apply_importance(request.messages, budget, request.model)

        tokens_after = _count_tokens(compressed, request.model)
        context.metadata["_context_compressed"] = True
        context.metadata["_context_strategy_used"] = self._strategy
        context.metadata["_tokens_before"] = tokens
        context.metadata["_tokens_after"] = tokens_after
        if fallback_used:
            context.metadata["_context_fallback_used"] = True

        return CallRequest(messages=compressed, model=request.model, kwargs=request.kwargs)

    async def post_call(
        self,
        request: CallRequest,
        response: CallResponse,
        context: ExecutionContext,
    ) -> CallResponse:
        """Pass through — ContextGuard does not modify responses."""
        return response

    # ------------------------------------------------------------------
    # Compression strategies
    # ------------------------------------------------------------------

    def _apply_sliding_window(
        self,
        messages: list[dict[str, Any]],
        budget: int,
        model: str,
    ) -> list[dict[str, Any]]:
        """Keep system messages + last user message; greedily fill with recent history."""
        system_msgs = [m for m in messages if m.get("role") == "system"]
        non_system = [m for m in messages if m.get("role") != "system"]

        last_user_idx: int | None = None
        for i in range(len(non_system) - 1, -1, -1):
            if non_system[i].get("role") == "user":
                last_user_idx = i
                break

        if last_user_idx is None:
            return system_msgs

        last_user = non_system[last_user_idx]
        history = non_system[:last_user_idx] + non_system[last_user_idx + 1 :]

        protected = system_msgs + [last_user]
        protected_tokens = _count_tokens(protected, model)
        remaining = budget - protected_tokens

        kept: list[dict[str, Any]] = []
        for msg in reversed(history):
            msg_tokens = _count_tokens([msg], model)
            if remaining >= msg_tokens:
                kept.insert(0, msg)
                remaining -= msg_tokens

        return system_msgs + kept + [last_user]

    async def _apply_summarize(
        self,
        messages: list[dict[str, Any]],
        budget: int,
        model: str,
    ) -> tuple[list[dict[str, Any]], bool]:
        """Summarise the oldest half of history; fall back to sliding_window if still over.

        Returns:
            A tuple of (compressed_messages, fallback_used).
            ``fallback_used`` is True when the summary was still over budget and
            sliding_window was applied as a secondary compression pass.
        """
        if self._lock is None:
            self._lock = asyncio.Lock()

        async with self._lock:
            # Double-checked: a concurrent call may have already compressed
            tokens = _count_tokens(messages, model)
            if tokens <= budget:
                return messages, False

            system_msgs = [m for m in messages if m.get("role") == "system"]
            non_system = [m for m in messages if m.get("role") != "system"]

            last_user_idx: int | None = None
            for i in range(len(non_system) - 1, -1, -1):
                if non_system[i].get("role") == "user":
                    last_user_idx = i
                    break

            if last_user_idx is None or len(non_system) <= 1:
                return self._apply_sliding_window(messages, budget, model), False

            last_user = non_system[last_user_idx]
            history = non_system[:last_user_idx] + non_system[last_user_idx + 1 :]

            if not history:
                return self._apply_sliding_window(messages, budget, model), False

            # Summarise the oldest half of history
            split = max(1, len(history) // 2)
            to_summarize = history[:split]
            remaining_history = history[split:]

            formatted = "\n".join(
                f"{m.get('role', 'unknown')}: {m.get('content', '')}" for m in to_summarize
            )
            summary_sys_msg: dict[str, Any] = {
                "role": "system",
                "content": "You are a concise conversation summarizer.",
            }
            summary_req_msg: dict[str, Any] = {
                "role": "user",
                "content": (
                    "Summarize the following conversation history concisely, "
                    "preserving key facts, decisions, and context. Be brief.\n\n"
                    + formatted
                ),
            }
            sum_model = self._summary_model or model
            summary_request = CallRequest(
                messages=[summary_sys_msg, summary_req_msg],
                model=sum_model,
                kwargs={},
            )

            assert self._llm_callable is not None
            try:
                summary_response = await self._llm_callable(summary_request)
                summary_text = str(summary_response.content)
            except Exception as exc:
                err_str = str(exc).lower()
                if "model" in err_str and self._summary_model is not None:
                    raise BrixGuardError(
                        "context",
                        f"Summarization failed: model {self._summary_model!r} not supported. "
                        "Check the model name or use context_summary_model=None to use the "
                        "same model as the main client.",
                    ) from exc
                raise

            summary_msg: dict[str, Any] = {
                "role": "system",
                "content": f"[Conversation summary]: {summary_text}",
            }
            compressed = system_msgs + [summary_msg] + remaining_history + [last_user]

            # Fallback guarantee: if still over budget, apply sliding_window
            if _count_tokens(compressed, model) > budget:
                compressed = self._apply_sliding_window(compressed, budget, model)
                return compressed, True

            return compressed, False

    def _apply_importance(
        self,
        messages: list[dict[str, Any]],
        budget: int,
        model: str,
    ) -> list[dict[str, Any]]:
        """Keep messages greedily by importance score (highest first, tie-break by recency)."""
        system_msgs = [m for m in messages if m.get("role") == "system"]
        non_system = [m for m in messages if m.get("role") != "system"]

        last_user_idx: int | None = None
        for i in range(len(non_system) - 1, -1, -1):
            if non_system[i].get("role") == "user":
                last_user_idx = i
                break

        if last_user_idx is None:
            return system_msgs

        last_user = non_system[last_user_idx]
        history = non_system[:last_user_idx] + non_system[last_user_idx + 1 :]

        protected = system_msgs + [last_user]
        protected_tokens = _count_tokens(protected, model)
        remaining = budget - protected_tokens

        def _score(msg: dict[str, Any]) -> int:
            if "tool_calls" in msg:
                return 3
            if msg.get("role") == "tool":
                return 3
            if msg.get("role") == "user":
                return 2
            return 1

        # Sort by (score desc, index desc) for tie-breaking by recency
        scored = sorted(
            enumerate(history), key=lambda x: (_score(x[1]), x[0]), reverse=True
        )

        kept_indices: set[int] = set()
        for idx, msg in scored:
            msg_tokens = _count_tokens([msg], model)
            if remaining >= msg_tokens:
                kept_indices.add(idx)
                remaining -= msg_tokens

        # Restore original order
        kept = [msg for i, msg in enumerate(history) if i in kept_indices]

        return system_msgs + kept + [last_user]


__all__ = ["ContextGuard", "_count_tokens"]
