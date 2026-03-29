"""BrixClient and BRIX factory — the public entry point for the BRIX library.

Usage::

    import openai
    from brix import BRIX

    client = BRIX.wrap(
        openai.AsyncOpenAI(),
        max_cost_usd=50.0,
        requests_per_minute=500,
        per_call_timeout=30.0,
        max_retries=3,
        response_schema=MyModel,
        log_path="./traces",
    )

    result = await client.complete([{"role": "user", "content": "Hello"}])
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from brix.chain import InterceptorChain
from brix.context import CallRecord, ExecutionContext
from brix.exceptions import BrixConfigurationError
from brix.guards.protocol import CallRequest, CallResponse, Guard
from brix.settings import get_settings
from brix.exceptions import BrixTimeoutError

if TYPE_CHECKING:
    from brix.guards.observability import ObservabilityGuard
    from brix.replay import BrixReplayClient


def build_llm_callable(
    llm_client: Any,
) -> Callable[[CallRequest], Awaitable[CallResponse]]:
    """Detect the LLM client interface and return the appropriate adapter callable.

    This module-level function is shared by BrixClient (for the chain's own LLM
    call), RetryGuard, and SchemaGuard (which call the LLM directly for their
    own loops).

    Args:
        llm_client: Any supported LLM client (OpenAI SDK, Anthropic SDK, or
            BRIX ``LLMClient`` protocol).

    Returns:
        An async callable ``(CallRequest) -> CallResponse``.

    Raises:
        BrixConfigurationError: If the client interface is not recognized.
    """
    if hasattr(llm_client, "chat") and hasattr(getattr(llm_client, "chat", None), "completions"):
        return _OpenAIAdapter(llm_client)

    if hasattr(llm_client, "messages") and hasattr(getattr(llm_client, "messages", None), "create"):
        return _AnthropicAdapter(llm_client)

    if callable(getattr(llm_client, "complete", None)):
        return _LegacyAdapter(llm_client)

    raise BrixConfigurationError(
        "llm_client does not implement a recognized interface. "
        "Expected: OpenAI SDK, Anthropic SDK, or brix.LLMClient protocol "
        "(async def complete(prompt, *, system, temperature, max_tokens) -> str)."
    )


class _OpenAIAdapter:
    """Async callable adapter for the OpenAI SDK (openai>=1.0)."""

    def __init__(self, client: Any) -> None:
        self._client = client

    async def __call__(self, request: CallRequest) -> CallResponse:
        model = request.model or "gpt-4o"
        temperature = request.kwargs.get("temperature", 0.7)
        max_tokens = request.kwargs.get("max_tokens", 1024)

        create = self._client.chat.completions.create
        raw = create(
            model=model,
            messages=request.messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        # If it's a coroutine, await it
        if asyncio.iscoroutine(raw):
            raw = await raw

        content: str = raw.choices[0].message.content or ""
        usage = None
        if hasattr(raw, "usage") and raw.usage is not None:
            usage = {
                "prompt_tokens": raw.usage.prompt_tokens,
                "completion_tokens": raw.usage.completion_tokens,
                "total_tokens": raw.usage.total_tokens,
            }
        return CallResponse(content=content, usage=usage, raw=raw)


class _AnthropicAdapter:
    """Async callable adapter for the Anthropic SDK (anthropic>=0.30)."""

    def __init__(self, client: Any) -> None:
        self._client = client

    async def __call__(self, request: CallRequest) -> CallResponse:
        model = request.model or "claude-3-5-sonnet-20241022"
        max_tokens = request.kwargs.get("max_tokens", 1024)

        system_parts = [m["content"] for m in request.messages if m.get("role") == "system"]
        non_system = [m for m in request.messages if m.get("role") != "system"]

        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": non_system,
        }
        if system_parts:
            kwargs["system"] = "\n".join(system_parts)

        create = self._client.messages.create
        raw = create(**kwargs)
        if asyncio.iscoroutine(raw):
            raw = await raw

        content = raw.content[0].text if raw.content else ""
        usage = None
        if hasattr(raw, "usage") and raw.usage is not None:
            usage = {
                "input_tokens": raw.usage.input_tokens,
                "output_tokens": raw.usage.output_tokens,
            }
        return CallResponse(content=content, usage=usage, raw=raw)


class _LegacyAdapter:
    """Adapter for the BRIX LLMClient protocol (prompt-based, not messages-based)."""

    def __init__(self, client: Any) -> None:
        self._client = client

    async def __call__(self, request: CallRequest) -> CallResponse:
        prompt = ""
        system: str | None = None

        for msg in reversed(request.messages):
            if msg.get("role") == "user" and not prompt:
                prompt = msg.get("content", "")
            if msg.get("role") == "system" and system is None:
                system = msg.get("content")

        temperature = request.kwargs.get("temperature", 0.7)
        max_tokens = request.kwargs.get("max_tokens", 1024)

        content = await self._client.complete(
            prompt,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return CallResponse(content=str(content), usage=None, raw=content)


class BrixClient:
    """Wraps any LLM client and routes calls through the Guard chain.

    Do not instantiate directly — use ``BRIX.wrap()`` instead.

    The client maintains one :class:`~brix.context.ExecutionContext` for its
    entire lifetime. Session cost, call count, and call history accumulate
    across all calls to ``complete()``.

    Args:
        llm_client: Any LLM client (OpenAI SDK, Anthropic SDK, or BRIX protocol).
        guards: Ordered list of Guards to run on every call.
        log_path: Optional path for JSONL call logging.
        obs_guard: Optional reference to ObservabilityGuard for trace access.
    """

    def __init__(
        self,
        llm_client: Any,
        *,
        guards: list[Guard],
        log_path: str | Path | None = None,
        obs_guard: ObservabilityGuard | None = None,
    ) -> None:
        self._llm = llm_client
        self._chain = InterceptorChain(guards)
        self._context = ExecutionContext.new_session()
        self._settings = get_settings()
        self._obs_guard = obs_guard
        resolved_log = log_path or self._settings.log_path
        self._log_path = Path(resolved_log) if resolved_log else None

        base_callable = build_llm_callable(llm_client)
        context = self._context

        async def _timeout_aware_callable(request: CallRequest) -> CallResponse:
            timeout: float | None = context.metadata.get("_per_call_timeout")
            if timeout is not None:
                try:
                    return await asyncio.wait_for(base_callable(request), timeout=timeout)
                except asyncio.TimeoutError as exc:
                    raise BrixTimeoutError(f"per_call_timeout={timeout}s exceeded") from exc
            return await base_callable(request)

        self._llm_callable = _timeout_aware_callable

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Send a chat request through the Guard chain and return the response.

        Args:
            messages: List of message dicts in OpenAI chat format, e.g.
                ``[{"role": "user", "content": "Hello"}]``.
            model: Model identifier. Falls back to ``BRIX_DEFAULT_MODEL``
                env var, then to an empty string (client default).
            **kwargs: Extra keyword arguments forwarded to the LLM client
                (e.g. ``temperature``, ``max_tokens``).

        Returns:
            The response content. A ``str`` normally; a validated Pydantic model
            instance when SchemaGuard is active.

        Raises:
            BrixGuardBlockedError: If a Guard blocks the request.
            BrixConfigurationError: If the LLM client interface is not recognized.
        """
        resolved_model = model or self._settings.default_model or ""
        request = CallRequest(
            messages=messages,
            model=resolved_model,
            kwargs=kwargs,
        )

        self._context.run_id = str(uuid4())
        self._context.call_count += 1

        t0 = time.perf_counter()
        response = await self._chain.execute(request, self._context, self._llm_callable)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        self._context.call_history.append(
            CallRecord(
                request=request,
                response=response,
                timestamp=datetime.now(timezone.utc),
                latency_ms=latency_ms,
            )
        )

        return response.content

    # Alias for explicit async naming preference
    acomplete = complete

    def get_traces(self) -> list[dict[str, Any]]:
        """Return in-memory trace entries from ObservabilityGuard, most-recent first.

        Returns an empty list if ObservabilityGuard is not active.
        """
        if self._obs_guard is not None:
            return self._obs_guard.get_traces()
        return []

    @property
    def context(self) -> ExecutionContext:
        """The session-scoped ExecutionContext for this client."""
        return self._context

    @property
    def chain(self) -> InterceptorChain:
        """The InterceptorChain holding all registered Guards."""
        return self._chain


class BRIX:
    """Factory for creating a :class:`BrixClient` with configured Guards.

    Usage::

        client = BRIX.wrap(
            openai.AsyncOpenAI(),
            max_cost_usd=50.0,
            requests_per_minute=500,
            per_call_timeout=30.0,
            max_retries=3,
            response_schema=MyModel,
            log_path="./traces",
        )
        result = await client.complete([{"role": "user", "content": "..."}])
    """

    @classmethod
    def wrap(
        cls,
        llm_client: Any,
        *,
        # BudgetGuard
        max_cost_usd: float | None = None,
        budget_strategy: str = "block",
        budget_warning_threshold: float = 0.8,
        # RateLimitGuard
        requests_per_minute: int | None = None,
        rate_limit_rpm: int | None = None,  # legacy alias
        adaptive_rate_limiting: bool = True,
        min_rate_floor: float = 0.1,
        rate_reduction_factor: float = 0.5,
        rate_recovery_factor: float = 1.05,
        recovery_window_seconds: float = 60.0,
        burst_capacity: int | None = None,
        # TimeoutGuard
        per_call_timeout: float | None = None,
        per_step_timeout: float | None = None,
        total_timeout: float | None = None,
        on_timeout: str = "raise",
        max_time_seconds: float | None = None,  # legacy alias for per_call_timeout
        # ObservabilityGuard (always active; buffer-only when log_path=None)
        log_path: str | Path | None = None,
        trace_buffer_size: int = 1000,
        strict_mode: bool = False,
        max_session_records: int | None = None,
        record_external_calls: bool = True,  # reserved, no-op
        # SchemaGuard
        response_schema: type | None = None,
        max_schema_retries: int = 2,
        inject_schema_in_prompt: bool = True,
        max_healing_seconds: float | None = None,
        # RetryGuard
        max_retries: int | None = None,
        retry_on: list[int] | None = None,
        backoff_base: float = 2.0,
        max_backoff: float = 60.0,
        retry_budget_seconds: float = 120.0,
        # LoopGuard
        exact_loop_detection: bool = False,
        exact_loop_threshold: int = 3,
        semantic_loop_detection: bool = False,
        semantic_loop_threshold: float = 0.92,
        on_loop: str = "inject_diversity",
        diversity_attempts: int = 2,
        loop_window: int = 10,
        loop_diversity_prompt: str | None = None,
        # ContextGuard
        max_context_tokens: int | None = None,
        context_strategy: str = "sliding_window",
        context_reserve_tokens: int = 500,
        context_summary_model: str | None = None,
        # Regulated domain guard
        regulated_spec: str | Path | Any | None = None,
    ) -> BrixClient:
        """Wrap an LLM client with BRIX reliability Guards.

        Guards are registered in this order (innermost = last before LLM call):

        1. **BudgetGuard** — cheapest pre-check; blocks over-budget calls.
        2. **RateLimitGuard** — token bucket throttling to prevent 429.
        3. **TimeoutGuard** — sets timing context before the call.
        4. **LoopGuard** — detects infinite agent loops; injects diversity prompts.
        5. **ContextGuard** — compresses conversation history to fit token budget.
        6. **ObservabilityGuard** — always active; records start time in pre_call,
           writes validated response to audit log and DRE in post_call.
        7. **SchemaGuard** — injects schema in pre_call; validates/heals in post_call.
        8. **RetryGuard** — short-circuit guard; calls LLM with retry logic.
        9. **RegulatedGuard** — domain policy enforcement (optional extra).

        The actual ``post_call`` execution order is the reverse:
        Schema → Observability → Context → Loop → Timeout → RateLimit → Budget.
        LoopGuard and ContextGuard are placed before ObservabilityGuard so their
        post_call hooks run; RetryGuard short-circuits, so any guard after it
        cannot run post_call.

        Args:
            llm_client: Any supported LLM client (see :class:`BrixClient`).
            max_cost_usd: Maximum cumulative session cost in USD (BudgetGuard).
            budget_strategy: ``"block"`` or ``"warn"`` enforcement mode.
            budget_warning_threshold: Fraction of budget at which to warn. Default 0.8.
            requests_per_minute: Target rate cap; activates RateLimitGuard.
            rate_limit_rpm: Legacy alias for ``requests_per_minute``.
            adaptive_rate_limiting: Enable auto-adjustment on 429. Default True.
            min_rate_floor: Minimum effective rate fraction of configured max. Default 0.1.
            rate_reduction_factor: Multiplicative rate reduction on 429. Default 0.5.
            rate_recovery_factor: Multiplicative rate recovery per window. Default 1.05.
            recovery_window_seconds: Seconds without 429 before rate recovery. Default 60.
            burst_capacity: Optional hard cap on token bucket capacity.
            per_call_timeout: Max seconds for a single LLM call (TimeoutGuard).
            per_step_timeout: Max seconds between consecutive calls (TimeoutGuard).
            total_timeout: Max seconds for the entire session (TimeoutGuard).
            on_timeout: ``"raise"`` or ``"return_partial"``.
            max_time_seconds: Legacy alias for ``per_call_timeout``.
            log_path: Directory for JSONL audit log and DRE session files.
                Activates disk writes for ObservabilityGuard; otherwise buffer-only.
            trace_buffer_size: Max in-memory trace entries. Default 1000.
            strict_mode: When True, log write failures raise instead of logging.
            max_session_records: Rotate DRE session file after this many records.
            record_external_calls: Reserved for future tool-call recording. No-op now.
            response_schema: Pydantic model class for structured output (SchemaGuard).
            max_schema_retries: Max self-healing re-prompt attempts. Default 2.
            inject_schema_in_prompt: Whether to inject the schema in pre_call. Default True.
            max_healing_seconds: Optional time budget for SchemaGuard's healing loop.
            max_retries: Max retry attempts for transient failures (RetryGuard).
            retry_on: Extra HTTP status codes to treat as retryable.
            backoff_base: Exponential backoff base. Default 2.0.
            max_backoff: Max backoff delay in seconds. Default 60.0.
            retry_budget_seconds: Total time budget for all retry delays. Default 120.0.
            exact_loop_detection: Enable Tier 1 SHA-256 exact loop detection (LoopGuard).
            exact_loop_threshold: Identical-response count that triggers detection. Default 3.
            semantic_loop_detection: Enable Tier 2 cosine-similarity detection. Requires
                ``pip install 'brix-protocol[semantic]'``.
            semantic_loop_threshold: Cosine similarity threshold for semantic detection.
                Default 0.92.
            on_loop: ``"inject_diversity"`` (default) or ``"raise"``.
            diversity_attempts: Max diversity injections before raising. Default 2.
            loop_window: Rolling window size for loop history. Default 10.
            loop_diversity_prompt: Custom diversity injection text. Uses built-in if None.
            max_context_tokens: Max tokens per request; activates ContextGuard.
            context_strategy: ``"sliding_window"`` (default), ``"summarize"``,
                or ``"importance"``.
            context_reserve_tokens: Tokens reserved for the model response. Default 500.
            context_summary_model: Model for summarisation calls (ContextGuard). Uses
                the main model if None.
            regulated_spec: Spec path or name for regulated-domain analysis (RegulatedGuard).

        Returns:
            A configured :class:`BrixClient` ready to use.
        """
        guards: list[Guard] = []
        settings = get_settings()

        # 1. BudgetGuard
        if max_cost_usd is not None:
            from brix.guards.budget import BudgetGuard  # noqa: PLC0415

            guards.append(
                BudgetGuard(
                    max_cost_usd,
                    strategy=budget_strategy,
                    warning_threshold=budget_warning_threshold,
                )
            )

        # 2. RateLimitGuard — requests_per_minute takes priority; rate_limit_rpm is legacy alias
        resolved_rpm = requests_per_minute if requests_per_minute is not None else rate_limit_rpm
        if resolved_rpm is not None:
            from brix.guards.rate_limit import RateLimitGuard  # noqa: PLC0415

            guards.append(
                RateLimitGuard(
                    resolved_rpm,
                    adaptive=adaptive_rate_limiting,
                    min_rate_floor=min_rate_floor,
                    rate_reduction_factor=rate_reduction_factor,
                    rate_recovery_factor=rate_recovery_factor,
                    recovery_window_seconds=recovery_window_seconds,
                    burst_capacity=burst_capacity,
                )
            )

        # 3. TimeoutGuard — per_call_timeout takes priority over legacy max_time_seconds
        resolved_per_call = per_call_timeout if per_call_timeout is not None else max_time_seconds
        if any(v is not None for v in (resolved_per_call, per_step_timeout, total_timeout)):
            from brix.guards.timeout import TimeoutGuard  # noqa: PLC0415

            guards.append(
                TimeoutGuard(
                    per_call=resolved_per_call,
                    per_step=per_step_timeout,
                    total=total_timeout,
                    on_timeout=on_timeout,
                )
            )

        # 4. LoopGuard
        if exact_loop_detection or semantic_loop_detection:
            from brix.guards.loop import LoopGuard  # noqa: PLC0415

            guards.append(
                LoopGuard(
                    exact_threshold=exact_loop_threshold,
                    semantic_detection=semantic_loop_detection,
                    semantic_threshold=semantic_loop_threshold,
                    on_loop=on_loop,
                    diversity_attempts=diversity_attempts,
                    loop_window=loop_window,
                    diversity_prompt=loop_diversity_prompt,
                )
            )

        # 5. ContextGuard
        if max_context_tokens is not None:
            from brix.guards.context import ContextGuard  # noqa: PLC0415

            guards.append(
                ContextGuard(
                    max_context_tokens,
                    strategy=context_strategy,
                    reserve_tokens=context_reserve_tokens,
                    llm_callable=build_llm_callable(llm_client)
                    if context_strategy == "summarize"
                    else None,
                    summary_model=context_summary_model,
                )
            )

        # 6. ObservabilityGuard — always active (buffer-only when log_path=None)
        # Build guard_names from guards registered so far + those coming after
        # We build this list before constructing ObservabilityGuard so it includes
        # all guard names for the audit log.  The "future" names are known statically.
        future_names = []
        if response_schema is not None:
            future_names.append("schema")
        if max_retries is not None:
            future_names.append("retry")
        if regulated_spec is not None:
            future_names.append("regulated")

        current_names = [g.name for g in guards]
        all_guard_names = current_names + ["observability"] + future_names

        resolved_log = Path(log_path) if log_path else None
        effective_buffer = (
            trace_buffer_size if trace_buffer_size > 0 else settings.trace_buffer_size
        )
        from brix.guards.observability import ObservabilityGuard  # noqa: PLC0415

        obs_guard = ObservabilityGuard(
            log_path=resolved_log,
            guard_names=all_guard_names,
            buffer_size=effective_buffer,
            strict_mode=strict_mode,
            max_session_records=max_session_records,
        )
        guards.append(obs_guard)

        # 7. SchemaGuard
        if response_schema is not None:
            from brix.guards.schema import SchemaGuard  # noqa: PLC0415

            guards.append(
                SchemaGuard(
                    build_llm_callable(llm_client),
                    response_schema,
                    max_retries=max_schema_retries,
                    inject_schema=inject_schema_in_prompt,
                    max_healing_seconds=max_healing_seconds,
                )
            )

        # 8. RetryGuard — short-circuit; calls LLM with retry logic
        if max_retries is not None:
            from brix.guards.retry import RetryGuard  # noqa: PLC0415

            guards.append(
                RetryGuard(
                    build_llm_callable(llm_client),
                    max_retries=max_retries,
                    backoff_base=backoff_base,
                    max_backoff=max_backoff,
                    retry_budget_seconds=retry_budget_seconds,
                    retry_on=retry_on,
                )
            )

        # 9. RegulatedGuard
        if regulated_spec is not None:
            try:
                from brix.regulated._guard import RegulatedGuard  # noqa: PLC0415
            except ImportError as exc:
                raise BrixConfigurationError(
                    "regulated_spec requires the 'regulated' extra. "
                    "Install with: pip install 'brix-protocol[regulated]'"
                ) from exc

            guards.append(
                RegulatedGuard(
                    llm_client=llm_client,
                    spec=regulated_spec,
                    embedding_model=settings.embedding_model,
                    log_path=resolved_log,
                )
            )

        return BrixClient(
            llm_client,
            guards=guards,
            log_path=log_path,
            obs_guard=obs_guard,
        )

    @classmethod
    def replay(
        cls,
        *,
        session_id: str,
        log_path: str | Path,
        schema: type | None = None,
    ) -> BrixReplayClient:
        """Create a replay client for a previously recorded session.

        Loads the DRE session file written by ObservabilityGuard and returns a
        :class:`~brix.replay.BrixReplayClient` that replays recorded responses
        in order, at zero LLM cost.

        Args:
            session_id: The session ID to replay. Found in every audit entry's
                ``session_id`` field.
            log_path: The same ``log_path`` that was passed to ``BRIX.wrap()``
                when the session was recorded.
            schema: Optional Pydantic model class. Required to reconstruct typed
                model instances from responses that were originally produced by
                SchemaGuard. Without it, dict values are returned with a warning.

        Returns:
            A :class:`~brix.replay.BrixReplayClient` instance.

        Raises:
            BrixReplayError: If no session file is found for ``session_id``.
        """
        from brix.replay import BrixReplayClient  # noqa: PLC0415

        return BrixReplayClient(
            session_id=session_id,
            log_path=Path(log_path),
            schema=schema,
        )

    @classmethod
    def purge_sessions(
        cls,
        log_path: str | Path,
        *,
        older_than_days: int = 7,
    ) -> int:
        """Delete DRE session files older than the specified number of days.

        This is a management utility for long-running applications. Call it
        from a cleanup job to reclaim disk space from old session files.

        Args:
            log_path: The ``log_path`` directory where session files are stored.
            older_than_days: Delete files whose last modification time is older
                than this many days. Default 7.

        Returns:
            The number of session files deleted.
        """
        import time as _time  # noqa: PLC0415

        cutoff = _time.time() - older_than_days * 86400
        sessions_dir = Path(log_path) / ".brix_sessions"
        if not sessions_dir.exists():
            return 0
        deleted = 0
        for f in sessions_dir.glob("*.jsonl"):
            try:
                if f.stat().st_mtime < cutoff:
                    f.unlink()
                    deleted += 1
            except OSError:
                pass
        return deleted


__all__ = ["BRIX", "BrixClient", "build_llm_callable"]
